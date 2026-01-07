#!/bin/bash

# Configuration
SERVER="192.168.86.227"
APP_DIR="/opt/sudoku"
SERVICE_NAME="sudoku"
EXTRACTION_SERVICE_NAME="sudoku-extraction"
DOMAIN="sudoku.seavey.dev"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Deployment to $SERVER...${NC}"

# 0. Run Tests
echo -e "${GREEN}[0/9] Running Backend Tests...${NC}"
cd backend
go test ./...
if [ $? -ne 0 ]; then
    echo -e "${RED}Tests failed. Deployment aborted.${NC}"
    exit 1
fi
cd ..

# 1. Build Backend
echo -e "${GREEN}[1/9] Building Backend (Go)...${NC}"
cd backend
GOOS=linux GOARCH=amd64 go build -o sudoku-server .
if [ $? -ne 0 ]; then
    echo -e "${RED}Backend build failed.${NC}"
    exit 1
fi
cd ..

# 2. Build Frontend
echo -e "${GREEN}[2/9] Building Frontend (Vue.js)...${NC}"
cd frontend
npm install
npm run build
if [ $? -ne 0 ]; then
    echo -e "${RED}Frontend build failed.${NC}"
    exit 1
fi
cd ..

# 3. Prepare Remote Directory
echo -e "${GREEN}[3/9] Preparing Remote Directory ($APP_DIR)...${NC}"
ssh $SERVER "sudo systemctl stop $SERVICE_NAME || true"
ssh $SERVER "sudo systemctl stop $EXTRACTION_SERVICE_NAME || true"
ssh $SERVER "sudo mkdir -p $APP_DIR && sudo chown \$(whoami) $APP_DIR"
ssh $SERVER "mkdir -p $APP_DIR/extraction_service"

# 4. Transfer Files
echo -e "${GREEN}[4/9] Transferring Files...${NC}"
scp backend/sudoku-server $SERVER:$APP_DIR/
scp -r frontend/dist $SERVER:$APP_DIR/
scp extraction_service/app.py $SERVER:$APP_DIR/extraction_service/
scp extraction_service/requirements.txt $SERVER:$APP_DIR/extraction_service/

# 4.5. Setup Environment
echo -e "${GREEN}[4.5/9] Setting up Environment Variables...${NC}"
ssh $SERVER "echo GOOGLE_API_KEY=\$(cat /home/cody/gemini-api-key) > $APP_DIR/config.env"

# 5. Setup Python Environment for Extraction Service
echo -e "${GREEN}[5/9] Setting up Python Environment for Extraction Service...${NC}"
ssh $SERVER "cd $APP_DIR/extraction_service && python3 -m venv venv"
ssh $SERVER "cd $APP_DIR/extraction_service && ./venv/bin/pip install --upgrade pip"
ssh $SERVER "cd $APP_DIR/extraction_service && ./venv/bin/pip install -r requirements.txt"

# 6. Setup Systemd Services
echo -e "${GREEN}[6/9] Configuring Systemd Services...${NC}"
scp deployment/sudoku.service $SERVER:/tmp/
scp deployment/sudoku-extraction.service $SERVER:/tmp/
ssh $SERVER "sudo mv /tmp/sudoku.service /etc/systemd/system/"
ssh $SERVER "sudo mv /tmp/sudoku-extraction.service /etc/systemd/system/"
ssh $SERVER "sudo systemctl daemon-reload"
ssh $SERVER "sudo systemctl enable $SERVICE_NAME"
ssh $SERVER "sudo systemctl enable $EXTRACTION_SERVICE_NAME"

# 7. Start Extraction Service First (Go server depends on it)
echo -e "${GREEN}[7/9] Starting Extraction Service...${NC}"
ssh $SERVER "sudo systemctl start $EXTRACTION_SERVICE_NAME"
sleep 5

# Check if extraction service is running
ssh $SERVER "curl -s http://127.0.0.1:5001/health" | grep -q "ok"
if [ $? -ne 0 ]; then
    echo -e "${RED}Warning: Extraction service may not be running properly${NC}"
    ssh $SERVER "sudo systemctl status $EXTRACTION_SERVICE_NAME"
fi

# 8. Start Main Service
echo -e "${GREEN}[8/9] Starting Main Service...${NC}"
ssh $SERVER "sudo systemctl start $SERVICE_NAME"

# 9. Setup Nginx
echo -e "${GREEN}[9/9] Configuring Nginx...${NC}"
scp deployment/sudoku.seavey.dev.conf $SERVER:/tmp/
ssh $SERVER "sudo mv /tmp/sudoku.seavey.dev.conf /etc/nginx/sites-available/$DOMAIN.conf"
ssh $SERVER "sudo ln -sf /etc/nginx/sites-available/$DOMAIN.conf /etc/nginx/sites-enabled/"
ssh $SERVER "sudo nginx -t && sudo systemctl reload nginx"

echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}App should be live at https://$DOMAIN${NC}"
echo ""
echo "Service Status:"
ssh $SERVER "sudo systemctl status $SERVICE_NAME --no-pager | head -5"
echo ""
ssh $SERVER "sudo systemctl status $EXTRACTION_SERVICE_NAME --no-pager | head -5"
