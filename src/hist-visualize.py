import tensorflow as tf
import matplotlib.pyplot as plt

def get_pixel_values (dataset_path):
  ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(224, 224),
    batch_size=32,
    color_mode="grayscale",
    shuffle=False
  )

  values = []
  
  for images, _ in ds.take(5):
    values.extend(images.numpy().flatten())

  return values

train_pixels = get_pixel_values("../dataset")
eval_pixels = get_pixel_values("../eval_dataset")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(train_pixels, bins=50, color="steelblue", edgecolor="none")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Pixel Intensity Distribution (Training Dataset)")

plt.subplot(1, 2, 2)
plt.hist(eval_pixels, bins=50, color="navy", edgecolor="none")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Pixel Intensity Distribution (Evaluation Dataset)")

plt.tight_layout()
plt.show()