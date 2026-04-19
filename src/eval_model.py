import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

dataset = "../eval_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

def main():
  eval_ds = tf.keras.utils.image_dataset_from_directory(
    dataset,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=False
  )

  class_names = eval_ds.class_names

  norm = tf.keras.layers.Rescaling(1./255)
  eval_ds = eval_ds.map(lambda x, y: (norm(x), y))
  eval_ds = eval_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

  models_dir = "../models"
  model_files = [f for f in os.listdir(models_dir) if f.endswith(".keras") or f.endswith(".h5")]

  if not model_files:
    print("No model files found in ../models, please run train_model.py first")
    return
  
  print("Available models:")
  for i, name in enumerate(model_files):
    print(f"  [{i + 1}] {name}")

  choice = int(input("Select a model: "))
  selected_model = os.path.join(models_dir, model_files[choice - 1])
  print(f"Loaded model: {selected_model}")

  model = tf.keras.models.load_model(selected_model)

  loss, acc = model.evaluate(eval_ds)
  print(f"Eval Loss: {loss:.4f}")
  print(f"Eval Accuracy: {acc:.4f}")

  y_true = []
  y_pred = []

  for images, labels in eval_ds:
    predictions = model.predict(images, verbose=0)
    print("Raw prediction scores:", predictions[:5])
    print("True labels:", labels.numpy()[:5])
    if predictions.shape[-1] == 1:
      predicted_labels = (predictions > 0.35).astype(int).flatten()
    else:
      predicted_labels = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy().astype(int).flatten())
    y_pred.extend(predicted_labels)

  print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

  cm = confusion_matrix(y_true, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp.plot(cmap="Blues")
  plt.title("Confusion Matrix")
  plt.show()

  image_files = []
  for class_name in class_names:
    class_dir = os.path.join(dataset, class_name)
    for fname in sorted(os.listdir(class_dir)):
      image_files.append((os.path.join(class_dir, fname), class_name))

  print("\nMisclassified images:")
  for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
    if true != pred:
      path, _ = image_files[idx]
      print(f"  {path} | True: {class_names[true]} | Pred: {class_names[pred]}")

if __name__ == "__main__":
  main()