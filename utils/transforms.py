from torchvision import transforms

def get_transforms(phase="train"):
    """
    Returns appropriate torchvision transforms for training or evaluation.

    Args:
        phase (str): "train" or "val"

    Returns:
        torchvision.transforms.Compose object
    """
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # For grayscale images
        ])
    else:
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
