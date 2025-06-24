import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from utils.dataset import ExpressionDataset
from utils.transforms import get_transforms
from models.cnn_model import EmotionCNN
import config

def train():
    print("🚀 Starting training process...")

    # Set seed
    torch.manual_seed(config.SEED)

    # Transforms
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    # Load datasets
    print("📦 Loading datasets...")
    train_dataset = ExpressionDataset(config.TRAIN_CSV, config.TRAIN_IMG_DIR, transform=train_transforms)
    val_dataset = ExpressionDataset(config.VAL_CSV, config.VAL_IMG_DIR, transform=val_transforms)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    # Model
    print("🧠 Initializing model...")
    model = EmotionCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_val_acc = 0.0

    print("🔁 Training loop starting...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_losses = []
        all_preds = []
        all_labels = []

        print(f"\n📅 Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Step {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_acc = accuracy_score(all_labels, all_preds)
        print(f"✅ Train Loss: {sum(train_losses)/len(train_losses):.4f} | Train Accuracy: {train_acc:.4f}")

        # Evaluation
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                outputs = model(images)
                preds = torch.argmax(outputs, 1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"📊 Validation Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"💾 Best model saved with accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    train()
