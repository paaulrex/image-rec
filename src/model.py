import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from collections import Counter
from sklearn.metrics import classification_report

# Define the target image size for all images after resizing
IMG_SIZE = (224, 224)

# Define the number of images processed in one batch during training
BATCH_SIZE = 16

# Set a random seed so the train/validation split is reproducible
SEED = 123

# Define dataset path
dataset_path = "..\\dataset"

# Lists to store image sizes and classification results
sizes = []
y_true = []
y_pred = []

# Read images from each class folder and collect original image sizes
for label in ["yes", "no"]:
    folder = os.path.join(dataset_path, label)

    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)

        try:
            # Open image and store its original width and height
            with Image.open(filepath) as img:
                sizes.append(img.size)

        except Exception as e:
            # Skip unreadable or corrupted image files
            print("Skipped:", file, "| Error:", e)

# print("Total Images:", len(sizes))

# Count how many images exist for each original image size
size_counts = Counter(sizes)

# Uncomment below if you want to inspect the image size distribution
# for size, count in size_counts.most_common():
#     print(f"{size} : {count}")

# Create the training dataset from the directory
# validation_split=0.2 means 80% training and 20% validation
# color_mode="grayscale" loads images with one channel instead of RGB
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

# Create the validation dataset using the same split settings
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

# Store class labels based on folder names
class_names = train_ds.class_names
# print(class_names)

# Normalize pixel values from the range [0, 255] to [0, 1]
# This helps improve model stability and training performance
normalization_layer = tf.keras.layers.Rescaling(1. / 255)

# Apply normalization to training and validation images
# Labels remain unchanged
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Let TensorFlow automatically optimize data loading performance
AUTOTUNE = tf.data.AUTOTUNE

# Optimize the training pipeline:
# cache() stores data in memory after first load for faster reuse
# shuffle(1000) randomizes the training data order
# prefetch() prepares the next batch while the current one is being processed
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# Optimize the validation pipeline:
# cache() speeds up repeated access
# prefetch() improves evaluation efficiency
# No shuffle is used so validation remains consistent
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Ask user for activation functions
conv_activation = input("Enter Conv activation (relu/tanh/elu): ").lower()
dense_activation = input("Enter Dense activation (relu/tanh/elu): ").lower()
output_activation = input("Enter Output activation (sigmoid/softmax): ").lower()

#Validate Inputs
valid_hidden = ['relu', 'tanh', 'elu']
valid_output = ['sigmoid', 'softmax']

if conv_activation not in valid_hidden:
    print("Invalid Conv activation → using relu")
    conv_activation = 'relu'

if dense_activation not in valid_hidden:
    print("Invalid Dense activation → using relu")
    dense_activation = 'relu'

if output_activation not in valid_output:
    print("Invalid Output activation → using sigmoid")
    output_activation = 'sigmoid'

#Handle output layer
if output_activation == 'softmax':
    output_units = 2
    loss_fn = 'sparse_categorical_crossentropy'
else:
    output_units = 1
    loss_fn = 'binary_crossentropy'

# Define the CNN model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation=conv_activation, padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation=conv_activation, padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation=conv_activation, padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=dense_activation),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(output_units, activation=output_activation)
])

# Compile the model
# Adam is used as the optimizer
# loss function is choosen based on the activation function
# Accuracy is used to monitor training performance
model.compile(
    optimizer="adam",
    loss=loss_fn,
    metrics=["accuracy"]
)

# Display the model architecture summary
model.summary()

# Train the model using the training dataset
# Evaluate performance on the validation dataset after each epoch
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

# Store training and validation accuracy/loss values for later plotting if needed
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Evaluate final model performance on the validation dataset
val_loss, val_acc = model.evaluate(val_ds)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)

# Generate predictions for the validation set and collect true/predicted labels
for images, labels in val_ds:
    predictions = model.predict(images)

    # Convert predicted probabilities into class labels using 0.5 threshold
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    # Store actual and predicted labels for classification report
    y_true.extend(labels.numpy().astype(int).flatten())
    y_pred.extend(predicted_labels)

# Print precision, recall, f1-score, and support for each class
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# Lists to store misclassified images and related information
misclassified_images = []
misclassified_true = []
misclassified_pred = []
misclassified_scores = []

# Identify validation images the model classified incorrectly
for images, labels in val_ds:
    predictions = model.predict(images, verbose=0).flatten()
    predicted_labels = (predictions > 0.5).astype(int)
    true_labels = labels.numpy().astype(int).flatten()

    for i in range(len(images)):
        if predicted_labels[i] != true_labels[i]:
            # Store the misclassified image
            misclassified_images.append(images[i].numpy())

            # Store the true class label
            misclassified_true.append(true_labels[i])

            # Store the predicted class label
            misclassified_pred.append(predicted_labels[i])

            # Store the model's predicted probability score
            misclassified_scores.append(predictions[i])

# Print the total number of misclassified validation images
print("Total misclassified images:", len(misclassified_images))