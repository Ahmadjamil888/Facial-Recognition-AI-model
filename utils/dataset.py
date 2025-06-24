import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class ExpressionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file with 'filename' and 'label' columns.
            img_dir (str): Path to base directory where images are stored in subfolders by label.
            transform (callable, optional): Optional transform to apply on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.annotations["filename"] = self.annotations["filename"].astype(str)  # Ensure filenames are strings
        self.img_dir = img_dir
        self.transform = transform

        # Create label-to-index mapping, e.g. {'angry': 0, 'happy': 1, ...}
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.annotations['label'].unique()))}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_name = str(row["filename"])  # Ensure filename is a string
        label_str = row["label"]
        label = self.label_to_idx[label_str]

        # Construct image path: e.g., "data/raw/train/happy/img1.jpg"
        img_path = os.path.join(self.img_dir, label_str, img_name)

        # Load and convert to grayscale
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label
