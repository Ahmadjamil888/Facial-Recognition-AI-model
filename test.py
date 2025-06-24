import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from models.cnn_model import EmotionCNN
from utils.dataset import ExpressionDataset
from utils.transforms import get_transforms
from utils.helpers import calculate_accuracy, load_model, plot_confusion_matrix, print_classification_report
from config import *

# Define labels
emotion_labels = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


if __name__ == "__main__":
    # Load test data
    val_dataset = ExpressionDataset(
        csv_file=VAL_CSV,
        img_dir=VAL_IMG_DIR,
        transform=get_transforms("test")
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = EmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model = load_model(model, BEST_MODEL_PATH, DEVICE)

    # Run evaluation
    preds, labels = evaluate(model, val_loader)

    # Accuracy
    acc = sum([p == l for p, l in zip(preds, labels)]) / len(labels)
    print(f"\n✅ Validation Accuracy: {acc:.4f}")

    # Classification report
    print_classification_report(labels, preds, emotion_labels)

    # Confusion matrix
    plot_confusion_matrix(labels, preds, emotion_labels, save_path=CONFUSION_MATRIX_PATH)
