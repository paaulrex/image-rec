# Brain Tumor MRI Detection using Convolutional Neural Networks
## Overview
This project implements a Convolutional Neural Network (CNN) to classify brain MRI images as either:

Tumor present (yes)
No tumor (no)

The model is trained using a small MRI dataset and demonstrates a complete machine learning workflow including:

dataset preprocessing
model training
validation
evaluation using classification metrics
prediction on new images

The goal of the project is to explore how deep learning models can be applied to **medical image classification tasks**.

## Dataset
The dataset consists of 253 brain MRI images divided into two classes:
```
dataset/
   yes/   → MRI images containing tumors
   no/    → MRI images without tumors
```
Dataset statistics:

| **Class** | **Images**|
|-----------|-----------|
|Tumor (`yes`)|155|
|No Tumor (`no`)|98
|**Total**|253|

The images vary in resolution and format (`.jpg`, .`jpeg`), so preprocessing standardizes them before training.

## Project Structure

```
brain_tumor_cnn/
│
├── dataset/
│   ├── yes/
│   └── no/
│
├── notebooks/
│   └── tumor_cnn.ipynb
│
├── src/ (optional)
│   └── training/testing scripts
│
├── results/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   └── confusion_matrix.png
│
└── README.md
```

## Model Architecture
The model is a custom CNN designed for grayscale MRI images.

Input shape:
```
224 x 224 x 1
```

Architecture:
```
Conv2D (32 filters)
MaxPooling

Conv2D (64 filters)
MaxPooling

Conv2D (128 filters)
MaxPooling

Flatten
Dense (128)
Dropout
Dense (1) → Sigmoid
```

Key design decisions:
- Grayscale input since MRI scans do not require RGB channels
- Sigmoid output for binary classification
- Binary Cross Entropy loss
- Adam optimizer

## Data Preprocessing

Before training, images are processed as follows:

1. Resizing

All images are resized to:
```
224 x 224
```

2. Normalization

Pixel values are scaled:
```
pixel / 255
```

3. Automatic dataset split

The dataset is split into:
```
80% Training
20% Validation
```

using Tensorflow's `image_dataset_from_directory`.

## Training
Training is performed in the notebook.

Typical parameters:
```
Batch size: 16
Epochs: 15
Optimizer: Adam
Loss: Binary Crossentropy
```

Training output includes:
- training accuracy
- validation accuracy
- training loss
- validation loss

## Evaluation Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Example output:
```
precision    recall  f1-score   support

no       0.81      0.72      0.76        18
yes      0.85      0.91      0.88        32

accuracy                           0.84        50
```

Interpretation:

- The model performs well at detecting tumor images
- Slightly weaker performance occurs when identifying non-tumor cases\

## Visualizing Model Performance
The notebook generates visualizations including:

### Training curves
- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

These help identify issues like overfitting.
 
### Confusion Matrix

Shows the distribution of:
- True Positives
- True Negatives
- False Positives
- False Negatives

## Identifying Misclassified Images
The project includes analysis to identify incorrect predictions.

Misclassified images can be displayed with:
- predicted label
- true label
- prediction confidence

This helps analyze model weaknesses.

## Running the Project
1. Install dependencies
```bash
pip install tensorflow numpy matplotlib scikit-learn pillow pandas
```

2. Launch the notebook
Open:
```
tumor_cnn.ipynb
```

Run cells sequentially to:
- load the dataset
- train the CNN
- evaluate the results

## Testing New Images
New MRI images can be evaluated by:
1. Loading the trained model
2. Preprocessing the image
3. Running prediction

Steps include:
```
resize image -> normalize -> predict -> threshold at 0.5
```

Prediction output example:
```
Prediction: yes (tumor)
Confidence: 0.91
```

## Potential Improvements
No known future improvements planned.

## Educational Purpose
This project demonstrates:
- CNN design for medical image classification
- dataset preprocessing
- model evaluation techniques
- debugging misclassifications

It is intended for academic and research learning purposes only, not clinical use.