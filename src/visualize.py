import matplotlib.pyplot as plt
import numpy as np
import os

def count_classes(dataset_path):
  no_count = len(os.listdir(os.path.join(dataset_path, "no")))
  yes_count = len(os.listdir(os.path.join(dataset_path, "yes")))
  return no_count, yes_count

train_no, train_yes = count_classes("../dataset")
eval_no, eval_yes = count_classes("../eval_dataset")

datasets = ["Training", "Evaluation"]
no_counts = [train_no, eval_no]
yes_counts = [train_yes, eval_yes]

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, no_counts, width, label="No Tumor", color="steelblue")
bars2 = ax.bar(x + width/2, yes_counts, width, label="Tumor", color="navy")

ax.set_xlabel("Dataset")
ax.set_ylabel("Image Count")
ax.set_title("Class Distribution Across Datasets")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()

for bar in bars1 + bars2:
  ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
          str(int(bar.get_height())), ha="center", va="bottom")

plt.tight_layout()
plt.show()