import tensorflow as tf
import os
import argparse

dataset = "../dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 123

def build_model(conv_activation, dense_activation, output_activation):
  if output_activation == "softmax":
    output_units = 2
    loss_fn = "sparse_categorical_crossentropy"
  else:
    output_units = 1
    loss_fn = "binary_crossentropy"

  model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation=conv_activation, padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation=conv_activation, padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3,3), activation=conv_activation, padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=dense_activation),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(output_units, activation=output_activation)
  ])

  model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
  return model

def main():
  tf.keras.utils.set_random_seed(42)

  parser = argparse.ArgumentParser()
  parser.add_argument("--data", default=dataset, help="Path to training dataset folder")
  parser.add_argument("--output", default="../models/tumor_cnn.keras", help="Path to save trained model")
  parser.add_argument("--epoch", type=int, default=15)
  args = parser.parse_args()

  conv_act = input("Enter Conv activation (relu/tanh/elu): ").lower()
  dense_act = input("enter Dense activation (relu/tanh/elu): ").lower()
  out_act = input("Enter Output activation (sigmoid/softmax): ").lower()

  if conv_act not in ["relu", "tanh", "elu"]:
    conv_act = "relu"
  if dense_act not in ["relu", "tanh", "elu"]:
    dense_act = "relu"
  if out_act not in ["sigmoid", "softmax"]:
    out_act = "sigmoid"

  train_ds = tf.keras.utils.image_dataset_from_directory(
    args.data,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
  )

  val_ds = tf.keras.utils.image_dataset_from_directory(
    args.data,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
  )

  norm = tf.keras.layers.Rescaling(1./255)
  train_ds = train_ds.map(lambda x, y: (norm(x), y))
  val_ds = val_ds.map(lambda x, y: (norm(x), y))

  AUTOTUNE = tf.data.AUTOTUNE
  train_ds = train_ds.cache().shuffle(1000)

  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
  ])

  train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y)).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

  model = build_model(conv_act, dense_act, out_act)
  model.summary()

  class_weight = {0: 1.5, 1: 1.0}

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epoch,
    verbose=1,
    class_weight=class_weight
  )

  print(f"\nFinal train accuracy: {history.history['accuracy'][-1]:.4f}")
  print(f"Final val accuracy:   {history.history['val_accuracy'][-1]:.4f}")
  print(f"Final train loss:     {history.history['loss'][-1]:.4f}")
  print(f"Final val loss:       {history.history['val_loss'][-1]:.4f}")

  model_name = f"tumor_cnn_{conv_act}_{dense_act}_{out_act}.keras"
  output_path = os.path.join(os.path.dirname(args.output), model_name)
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  model.save(output_path)
  print(f"Model saved to {output_path}")

if __name__ == "__main__":
  main()