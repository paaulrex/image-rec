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
There are two datasets that are in this repo, a training dataset under `/dataset` and evaluation dataset under `/eval_dataset`
```
dataset/      → Training dataset folder
├── yes/      → MRI images containing tumors
└── no/       → MRI images without tumors

eval_dataset/  → Evaluation dataset folder
├── yes/      → MRI images containing tumors
└── no/       → MRI images without tumors
```
## Dataset statistics:
**Training Dataset**
|  **Class**   |  **Images**  |
|--------------|--------------|
|Tumor (`yes`) |  2,513       |
|No Tumor (`no`)| 2,087       |
|**Total**     |  4,600       |

Kaggle Link: <https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset>

**Evaluation Dataset**
|  **Class**   |  **Images**  |
|--------------|--------------|
|Tumor (`yes`) |  155         |
|No Tumor (`no`)| 98          |
|**Total**     |  253         |

Kaggle Link: <https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection>

The images vary in resolution and format (`.jpg`, .`jpeg`, `.tif`, `.png`), so preprocessing standardizes them before training.

## Project Structure

```
brain_tumor_cnn/
│
├── dataset/
│   ├── yes/
│   └── no/
│
├── eval_dataset/
│   ├── yes/
│   └── no/
│
├── models/
│   └── *.keras (generated after training model)
│
├── notebooks/
│   └── tumor_cnn.ipynb
│
├── src/ 
│   ├── model.py
│   ├── train_model.py
│   └── eval_model.py
│
├── results/ (not implemented yet)
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   └── confusion_matrix.png
│
└── README.md
```

## Source Files
### 1. `model.py`
A prototype training script that builds, trains, and evaluates the CNN in a single run

**What it does:**
- Scans `dataset` and collects original image sizes using PIL
- Loads training (80%) and validation (20%) splits from `dataset`
- Normalize pixel values to [0,1]
- Prompts the user to select activation functions at runtime
- Builds an trains the CNN for 15 epochs
- Prints a classification report on the validation set
- Collects misclassified validation images with their true/predicted labesl and confidence scores

**How to run:**
```bash
cd src
python model.py
```

**Interactive prompts**
```
Enter Conv activation (relu/tanh/elu):
Enter Dense activation (relu/tanh/elu):
Enter Output activation (sigmoid/softmax):
```
Invalid inpus fall back to defaults (as follows): `relu`, `relu`, `sigmoid`

**Output activation affects loss function:**
|  **Output**  |  **Units**   |  **Loss** |
|--------------|--------------|-----------|
|`sigmoid`     |  1           |`binary_crossentropy`|
|`softmax`     |  2           |`sparse_categorical_crossentropy`|

**Key parameters:**
```
IMG_SIZE:   224 x 224
BATCH_SIZE: 16
SEED:       123
EPOCHS:     15
Dropout:    0.4
```
---
### 2. `train_model.py`
A structured training script with data augmentation, class weighting, and dynamic model naming.

**What it does:**
- Loads training (80%) and validation (20%) splits from `../dataset/`
- Normalizes pixel values to [0, 1]
- Applies data augmentation to training images (horizontal flip, rotation, zoom)
- Prompts user for activation functions, then builds and trains the CNN
- Applies class weighting (`no`: 1.5, `yes`: 1.0) to handle class imbalance
- Saves the trained model to `../models/` with a name based on chosen activation

**How to run:**
```bash
cd src
python train_model.py
```

**Optional CLI arguments:**
```
--data   Path to training dataset (default: ../dataset)
--epoch  Number of epochs (default: 15)
```

**Model naming:**
```
tumor_cnn_{conv}_{dense}_{output}.keras
e.g. tumor_cnn_relu_relu_sigmoid.keras
```

**Key parameters:**
```
IMG_SIZE:      224 x 224
BATCH_SIZE:    16
SEED:          123
EPOCHS:        15 (overrideable, see CLI arguments)
Dropout:       0.4
Class weight:  no=1.5, yes=1.0
Augmentation:  horizontal flip, ±10% rotation, ±10% zoom
```


---

### 3. `eval_model.py`
Loads a saved model and runs a full evaluation against the eval dataset.

**What it does:**
- Loads images from `../eval_dataset/`
- Lists all `.keras` / `.h5` models in `../models/` and prompts user to select one
- Evaluations the selected model (loss + accuracy)
- Prints a full classification report (precision, recall, F1)
- Diplays a confusion matrix plot
- Identified and prints all misclassified images with true/predicted labels
- Diplays all misclassified images in a matplotlib grid

**How to run:**
```bash
cd src
python eval_model.py
```

**Interactive Prompt:**
```
Available models:
   [1] tumor_cnn_relu_relu_sigmoid.keras
   [2] tumor_tanh_tanh_softmax.keras
Select a model:
```

**Key parameters:**
```
IMG_SIZE:            224 x 224
BATCH_SIZE:          16
Sigmoid threshold:   0.35
```

**Output:**
- Eval loss and accuracy
- Per-class precisioin, recall, F1-score
- Confusion matrix (visual)
- List + grid plot of all misclassified images

---

## Jupyter Notebook contents
### Model Architecture
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

### Data Preprocessing

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

### Training
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

### Evaluation Metrics
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

### Visualizing Model Performance
The notebook generates visualizations including:

#### Training curves
- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

These help identify issues like overfitting.
 
#### Confusion Matrix

Shows the distribution of:
- True Positives
- True Negatives
- False Positives
- False Negatives

### Identifying Misclassified Images
The project includes analysis to identify incorrect predictions.

Misclassified images can be displayed with:
- predicted label
- true label
- prediction confidence

This helps analyze model weaknesses.

### Running the Project
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

### Testing New Images
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

## Educational Purpose
This project demonstrates:
- CNN design for medical image classification
- dataset preprocessing
- model evaluation techniques
- debugging misclassifications

It is intended for academic and research learning purposes only, not clinical use.