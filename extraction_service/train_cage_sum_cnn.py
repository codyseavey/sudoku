#!/usr/bin/env python3
"""
Train CNN for cage sum recognition (1-45).

Tuned settings for optimal performance:
- Label smoothing for better generalization
- Gradient clipping to prevent exploding gradients
- Data augmentation during training
- Learning rate warmup
- Class-weighted loss for imbalanced data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import json
import argparse

class CageSumDataset(Dataset):
    """Dataset for cage sum images."""

    def __init__(self, data_dir, transform=None, label_to_idx=None, augment=False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.samples = []

        # Build label mapping if not provided
        if label_to_idx is None:
            all_labels = set()
            for label_dir in sorted(self.data_dir.glob('label_*')):
                label = int(label_dir.name.split('_')[1])
                all_labels.add(label)

            # Create mapping: cage sum value -> class index
            sorted_labels = sorted(all_labels)
            self.label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
            self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        else:
            self.label_to_idx = label_to_idx
            self.idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        # Load all images
        for label_dir in sorted(self.data_dir.glob('label_*')):
            label = int(label_dir.name.split('_')[1])

            for img_path in sorted(label_dir.glob('*.png')):
                self.samples.append((str(img_path), label))

        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        print(f"Label mapping: {len(self.label_to_idx)} unique labels")

        # Training-time augmentation transforms
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=5,           # Rotation
                    translate=(0.05, 0.05),  # Translation
                    scale=(0.95, 1.05),  # Scaling
                    shear=3              # Shear
                ),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),  # Cutout-style
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to tensor
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension

        # Apply training-time augmentation
        if self.augment and hasattr(self, 'aug_transform'):
            img = self.aug_transform(img)

        if self.transform:
            img = self.transform(img)

        # Convert label to class index
        class_idx = self.label_to_idx[label]

        return img, class_idx

class CageSumCNN(nn.Module):
    """CNN for cage sum recognition (1-45)."""

    def __init__(self, num_classes=45):
        super(CageSumCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)

        # Fully connected layers
        # After 3 pooling layers: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Conv block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Conv block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

def compute_class_weights(dataset):
    """Compute inverse frequency weights for each class."""
    label_counts = {}
    for _, label in dataset.samples:
        label_counts[label] = label_counts.get(label, 0) + 1

    total = len(dataset.samples)
    weights = {}
    for label in range(len(dataset.label_to_idx)):
        cage_sum = dataset.idx_to_label[label]
        count = label_counts.get(cage_sum, 1)
        weights[label] = total / (len(label_counts) * count)

    return torch.FloatTensor([weights[i] for i in range(len(weights))])

def main():
    parser = argparse.ArgumentParser(description='Train cage sum CNN')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-dir', type=str,
                        default='training_data/cage_sum_cnn',
                        help='Training data directory')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor (default: 0.1)')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable training-time augmentation')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data directories
    base_dir = Path(__file__).parent / args.data_dir
    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'

    # Create datasets (train with augmentation, val without)
    use_augment = not args.no_augment
    train_dataset = CageSumDataset(train_dir, augment=use_augment)
    val_dataset = CageSumDataset(val_dir, label_to_idx=train_dataset.label_to_idx, augment=False)

    print(f"Training augmentation: {'enabled' if use_augment else 'disabled'}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    # Get number of classes from metadata
    metadata_path = base_dir / 'dataset_metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)

    num_classes = metadata['num_classes']
    print(f"Number of classes: {num_classes}")

    # Create model
    model = CageSumCNN(num_classes=num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute class weights for balanced loss
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Using class-weighted loss (inverse frequency weighting)")
    print(f"Label smoothing: {args.label_smoothing}")

    # Loss with label smoothing and class weights for better generalization
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Cosine annealing scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_model_path = Path(__file__).parent / 'models' / 'cage_sum_cnn.pth'
    best_model_path.parent.mkdir(exist_ok=True)

    # Early stopping
    patience_counter = 0

    print("\nStarting training...")
    print("=" * 80)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler (cosine annealing steps every epoch)
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'label_to_idx': train_dataset.label_to_idx,
                'idx_to_label': train_dataset.idx_to_label
            }, best_model_path)
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {best_model_path}")

    # Save training history
    history_path = Path(__file__).parent / 'models' / 'cage_sum_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to: {history_path}")

if __name__ == '__main__':
    main()
