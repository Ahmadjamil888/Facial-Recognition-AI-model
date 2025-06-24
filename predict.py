import torch
from PIL import Image
import argparse
import requests
from io import BytesIO
import torchvision.transforms as transforms

from models.cnn_model import EmotionCNN
from utils.helpers import load_model
from config import DEVICE, NUM_CLASSES

label_map = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load model
model = EmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
model = load_model(model, "checkpoints/best_model.pth", DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_url(image_url):
    # Download image from URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("L")  # Grayscale

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return label_map[predicted.item()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True, help='URL of the image to predict')
    args = parser.parse_args()

    prediction = predict_url(args.url)
    print(f"Predicted Emotion: {prediction}")
