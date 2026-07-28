# Cat & Dog Object Detector

A Streamlit web application powered by YOLO for detecting cats and dogs in uploaded images.

## Project Structure

```text
Cat-Dog/
├── assets/                   # Screenshot images for README
│   ├── screenshot-home.png
│   ├── screenshot-result.png
│   └── screenshot-result2.png
├── dataset/                  # Dataset folder (used for local training)
│   ├── images/               # Image files (.jpg, .png, etc.)
│   │   ├── train/
│   │   └── val/
│   ├── labels/               # YOLO annotation files (.txt)
│   │   ├── train/
│   │   └── val/
│   └── data.yaml             # YOLO dataset configuration file
├── weights/
│   └── YoloCatDog.pt         # Trained model weights
├── main.py                   # Streamlit web application
├── train.py                  # Model training script
└── requirements.txt          # Python dependencies
```

## 📷 Screenshots

### Home Page
![Home Page](assets/screenshot-home.png)

### Detection Outputs
![Detection Result 1](assets/screenshot-result.png)

![Detection Result 2](assets/screenshot-result2.png)

## How to Run Locally

1. Open your terminal in the project root directory.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit web application:
   ```bash
   streamlit run main.py
   ```
