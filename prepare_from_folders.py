import os
import pandas as pd

def prepare_csv_from_folders(data_dir, output_csv_path):
    data = []

    # Loop through each label folder (e.g., happy, sad)
    for label in os.listdir(data_dir):
        label_folder = os.path.join(data_dir, label)
        if not os.path.isdir(label_folder):
            continue
        for img_name in os.listdir(label_folder):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                data.append([img_name, label])  # Save filename and label

    # Create DataFrame
    df = pd.DataFrame(data, columns=["filename", "label"])
    df["filename"] = df["filename"].astype(str)

    # Save CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved: {output_csv_path} ({len(df)} samples)")

if __name__ == "__main__":
    prepare_csv_from_folders("data/raw/train", "data/processed/train.csv")
    prepare_csv_from_folders("data/raw/test", "data/processed/val.csv")
