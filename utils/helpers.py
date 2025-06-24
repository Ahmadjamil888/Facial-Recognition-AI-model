import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def calculate_accuracy(preds, labels):
    """
    Returns accuracy given model predictions and true labels.
    """
    _, pred_classes = torch.max(preds, 1)
    correct = (pred_classes == labels).sum().item()
    return correct / labels.size(0)


def save_model(model, path):
    """
    Saves the PyTorch model to a file.
    """
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """
    Loads model weights from file.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plots confusion matrix using seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def print_classification_report(y_true, y_pred, class_names):
    """
    Prints classification report.
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
