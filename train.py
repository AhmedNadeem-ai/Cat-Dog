import os
import shutil
from ultralytics import YOLO

def train_model():
    yaml_path = "dataset/data.yaml"

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Could not find '{yaml_path}'. Please ensure your dataset folder and data.yaml exist."
        )

    # Load base pre-trained model architecture
    model = YOLO("yolov8s.pt")

    # Train on your local dataset
    # Set device=0 for GPU, or device='cpu' if training on standard processor
    print("🚀 Starting local training...")
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # Change to 'cpu' if you don't have an Nvidia GPU
    )

    # Copy output weights to your weights/ folder for main.py to use
    os.makedirs("weights", exist_ok=True)
    best_weights_src = "runs/detect/train/weights/best.pt"
    destination_path = "weights/YoloCatDog.pt"

    shutil.copy(best_weights_src, destination_path)
    print(f"Training complete! Saved trained model to '{destination_path}'.")

if __name__ == "__main__":
    train_model()