#!/usr/bin/env python3
"""
Custom CNN digit classifier for sudoku cells.

A lightweight CNN optimized for recognizing digits 0-9 in sudoku cell images.
Designed to be faster and more accurate than general-purpose OCR for this
specific task.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os


class SudokuDigitCNN(nn.Module):
    """LeNet-5 inspired CNN for sudoku digit classification.

    Input: 64x64 grayscale image (single channel)
    Output: 10-class logits (digits 0-9)
    """

    def __init__(self, dropout_rate=0.5):
        super().__init__()

        # Convolutional layers
        # Input: 1x64x64
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)   # -> 32x64x64
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)                              # -> 32x32x32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # -> 64x32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)                              # -> 64x16x16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> 128x16x16
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)                              # -> 128x8x8

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten and fully connected
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNNDigitClassifier:
    """Inference wrapper for the CNN digit classifier.

    Provides a simple interface for predicting digits from cell images,
    with optional GPU acceleration.
    """

    def __init__(self, model_path=None, device=None):
        """Initialize the classifier.

        Args:
            model_path: Path to trained model weights (.pth file)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = SudokuDigitCNN()

        # Load weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        self.model.to(self.device)
        self.model.eval()

        print(f"CNNDigitClassifier initialized on {self.device}")

    def load_model(self, model_path):
        """Load trained model weights.

        Args:
            model_path: Path to .pth file
        """
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"Loaded model weights from {model_path}")

    def preprocess(self, cell_image):
        """Preprocess cell image for inference.

        Args:
            cell_image: Grayscale or BGR image (any size)

        Returns:
            Tensor ready for model input (1, 1, 64, 64)
        """
        # Convert to grayscale if needed
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image

        # Resize to 64x64
        resized = cv2.resize(gray, (64, 64))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Convert to tensor and add batch + channel dimensions
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, cell_image):
        """Predict digit from cell image.

        Args:
            cell_image: Grayscale or BGR numpy array (any size)

        Returns:
            tuple: (predicted_digit, confidence)
                - predicted_digit: int 0-9
                - confidence: float 0-1 (softmax probability)
        """
        tensor = self.preprocess(cell_image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)

        return predicted.item(), confidence.item()

    def predict_batch(self, cell_images):
        """Predict digits for multiple cells at once.

        Args:
            cell_images: List of cell images

        Returns:
            List of (digit, confidence) tuples
        """
        if not cell_images:
            return []

        # Preprocess all images
        tensors = []
        for img in cell_images:
            tensors.append(self.preprocess(img))

        # Stack into batch
        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = probs.max(1)

        return list(zip(predictions.cpu().numpy(), confidences.cpu().numpy()))

    def predict_with_alternatives(self, cell_image, top_k=3):
        """Get top-k predictions with confidences.

        Useful for debugging or ensemble approaches.

        Args:
            cell_image: Cell image
            top_k: Number of top predictions to return

        Returns:
            List of (digit, confidence) tuples, sorted by confidence descending
        """
        tensor = self.preprocess(cell_image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = probs.topk(top_k, dim=1)

        results = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            results.append((int(idx), float(prob)))

        return results


def get_model_path():
    """Get default model path relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'models', 'digit_cnn.pth')


# Global classifier instance (lazy-loaded)
_classifier = None


def get_classifier():
    """Get or create the global CNN classifier instance.

    Returns:
        CNNDigitClassifier instance
    """
    global _classifier
    if _classifier is None:
        model_path = get_model_path()
        if os.path.exists(model_path):
            _classifier = CNNDigitClassifier(model_path)
        else:
            print(f"WARNING: Model file not found at {model_path}")
            print("CNN classifier will not be available until model is trained.")
            return None
    return _classifier


if __name__ == '__main__':
    # Test model architecture
    print("Testing SudokuDigitCNN architecture...")

    model = SudokuDigitCNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with random input
    x = torch.randn(1, 1, 64, 64)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test classifier wrapper
    print("\nTesting CNNDigitClassifier...")
    classifier = CNNDigitClassifier()

    # Create dummy image
    dummy_img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    digit, conf = classifier.predict(dummy_img)
    print(f"Prediction: digit={digit}, confidence={conf:.4f}")

    # Test batch prediction
    dummy_batch = [dummy_img] * 5
    results = classifier.predict_batch(dummy_batch)
    print(f"Batch predictions: {len(results)} results")

    print("\nModel architecture test passed!")
