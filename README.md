# Real-Time Handwritten Character Recognition (0–9, A–Z)

## 1. Overview

### Project Purpose

The goal of this project is to develop a real-time handwritten character recognition system capable of identifying both digits (0–9) and uppercase letters (A–Z) using a convolutional neural network (CNN).

This project bridges the gap between offline model training and real-world deployment, where input data is captured dynamically from a webcam rather than clean, preprocessed datasets.

### Objectives

- Train a CNN model to classify 36 character classes (0–9, A–Z)
- Fine-tune the model specifically for uppercase handwritten letters
- Implement a real-time recognition system using OpenCV
- Design an image preprocessing pipeline (thresholding, contour detection, normalization)
- Improve prediction stability using a multi-frame stabilization mechanism
...

## 2. Code Structure

```text

ECE570_AI_Project/
│
├── app.py                  # Real-time webcam recognition (main demo)
├── inference.py            # Model loading and prediction function
├── model.py                # CNN architecture (RealTimeCharCNN)
├── class_map.py            # Mapping of class indices to characters
│
├── train.py                # Training script for 36-class model (0–9, A–Z)
├── finetune_letters.py     # Fine-tuning model on A–Z dataset
│
├── test_one_letter.py      # Quick sanity check (1 sample per class)
├── test_10_letters.py      # Random evaluation (multiple samples per class)
│
├── dataset.py              # Dataset loading utilities
├── csv_to_images.py        # Convert dataset from CSV to images
│
├── weights/                # Model weights (not included)
├── az_dataset/             # Custom dataset (not included)
├── outputs/                # Saved predictions and debug images
└── README.md

```

## 3. Dependencies
Install required packages:
```text
pip install torch torchvision opencv-python numpy
```
...

## 4. How to Run

1. Train model (optional)
```text
python train.py
```
2. Fine-tune on A–Z dataset
```text
python finetune_letters.py
```
3. Run real-time webcam demo
```text
python app.py
```
Controls:
Press q → Quit
Press s → Save prediction
4. Run evaluation scripts
```text
python test_one_letter.py
python test_10_letters.py
```

## 5. Dataset / Model

### Dataset

This project uses a combination of standard datasets and a custom subset:

- **Digits (0–9):**  
  Trained using the **MNIST dataset**

- **Letters (A–Z):**  
  Fine-tuned using a handwritten letters dataset from **Kaggle (A–Z Handwritten Alphabets dataset)**

Due to local computational and runtime constraints, a subset of the dataset was used:

- **Each letter class was limited to ~1000 samples**
- Total dataset size was reduced to ensure feasible training time

This subset still provides sufficient diversity for model learning while maintaining efficiency.

---

### Dataset Availability

The dataset (`az_dataset/`) is not included in this repository due to size constraints.

To reproduce results:

- Download:
  - MNIST (automatically handled by PyTorch)
  - A–Z dataset from Kaggle (A–Z Handwritten Alphabets)
- Generate images using:
  - `csv_to_images.py`

---

### Model Weights

Pretrained model weights (`.pt`) are not included due to GitHub size limitations.

To obtain weights:

- Run:
  ```bash
  python train.py
  python finetune_letters.py
  ```
  
## Code Attribution

### Written by me

- `app.py`  
  Real-time webcam pipeline, preprocessing, and stability mechanism  

- `finetune_letters.py`  
  Dataset selection and fine-tuning logic  

- `test_one_letter.py`, `test_10_letters.py`  
  Evaluation scripts  

- `class_map.py`  
  Class mapping  

---

### Adapted from prior code

- `model.py`  
  CNN architecture inspired by LeNet-style models  

- `train.py`  
  Standard PyTorch training loop  

---

### AI Assistance

Some parts of the implementation and debugging process were assisted by AI tools (e.g., ChatGPT), including:

- Debugging code and fixing runtime issues  
- Improving preprocessing pipeline (thresholding, contour handling)  
- Enhancing real-time stability (multi-frame prediction smoothing)  
- Structuring and refining README documentation  

All AI-generated suggestions were reviewed, modified, and integrated by the author.

---

###  External references

- PyTorch tutorials  
- OpenCV documentation  
- EMNIST dataset  

No code was directly copied without modification.

---

##  Modifications to Prior Code

### `app.py`

- Added real-time webcam pipeline  
- Added ROI cropping (lines ~30–40)  
- Implemented thresholding and contour detection (lines ~50–100)  
- Added multi-frame stabilization  
- Modified input size to 28×28  

---

### `inference.py`

- Updated to load fine-tuned weights  
- Added confidence threshold  

---

### `finetune_letters.py`

- Selected A–Z subset  
- Remapped labels to 36 classes  
- Implemented fine-tuning  
### Limitations
- Sensitive to lighting conditions
- Performance varies with handwriting style
- Domain gap between training dataset and webcam input

## 6. Results

### Single Sample per Class (Sanity Check)

A quick evaluation was performed by selecting one random sample from each letter class (A–Z):

- Accuracy: **84.62%**

This serves as a fast sanity check to verify that the model is functioning correctly.

---

### Multi-Sample Evaluation (10 Samples per Class)

A more reliable evaluation was conducted by testing **10 random samples per class**:

- Accuracy: **82.69%**

---

## 7. Notes
Dataset and model weights are not included due to GitHub size limits
This project is for educational purposes only
...
