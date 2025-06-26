# Facial Expression Recognition using CNN

This repository contains a deep learning model built with PyTorch for facial expression recognition. The model classifies images of human faces into one of seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Prediction](#prediction)  
- [Model Architecture](#model-architecture)  
- [Results](#results)  
- [License](#license)

---

## Project Overview

Facial expression recognition is a fundamental task in human-computer interaction and affective computing. This project implements a Convolutional Neural Network (CNN) trained on a labeled dataset of facial images to classify emotions accurately.
<img src="https://raw.githubusercontent.com/Ahmadjamil888/Facial-Recognition-AI-model/refs/heads/main/Screenshot%202025-06-26%20134324.png">


---

## Features

- Custom CNN architecture optimized for grayscale 48x48 pixel images.
- Data augmentation during training for improved generalization.
- Training and validation pipelines with accuracy and loss tracking.
- Model checkpointing for best validation accuracy.
- Inference script to predict emotion from an image URL.
- Modular and configurable codebase.

---

## Dataset

The dataset consists of labeled facial images categorized into seven emotions:  
**Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.**

The dataset should be organized with accompanying CSV files specifying image paths and labels. Images are expected to be grayscale and resized to 48x48 pixels during training and inference.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/face-expression-recognition.git
   cd face-expression-recognition
   ```

2. Create and activate a Python virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training the Model

Run the training script:

```bash
python train.py
```

The script will load the dataset, train the CNN model, and save the best performing model to the `checkpoints/` directory.

### Predicting Emotion from Image URL

Use the prediction script to classify an emotion from an image URL:

```bash
python predict.py --url "https://example.com/image.jpg"
```

The script outputs the predicted emotion label.

---

## Training

- The model trains using CrossEntropyLoss and Adam optimizer.
- Training and validation data loaders use data augmentation and normalization.
- Training progress is displayed with loss and accuracy metrics per epoch.
- The best model based on validation accuracy is saved automatically.

---

## Evaluation

Validation accuracy is printed after each epoch. Typical performance achieves approximately 50% accuracy on validation data, which can be improved with further tuning and data.

---

## Model Architecture

The CNN model consists of:

- 3 convolutional layers with batch normalization and max pooling.
- Dropout layer to reduce overfitting.
- Fully connected layers mapping to 7 output emotion classes.

Input images are grayscale of size 48x48 pixels.

---

## Results

Example metrics after training:

- **Train Loss:** 1.38
- **Train Accuracy:** 46.6%
- **Validation Accuracy:** 51.3%

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
