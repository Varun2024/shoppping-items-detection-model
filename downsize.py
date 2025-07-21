import os
import random
import shutil

# === CONFIGURATION ===
subset_size = 1000  #
base_dir = "D:/projects/new_shop_model/SKU110K_fixed"
original_images_dir = os.path.join(base_dir, "train")
original_labels_dir = os.path.join(base_dir, "labels")

subset_images_dir = os.path.join(base_dir, "train/train_subset")
subset_labels_dir = os.path.join(base_dir, "labels/train_subset")

# === SETUP DIRECTORIES ===
os.makedirs(subset_images_dir, exist_ok=True)
os.makedirs(subset_labels_dir, exist_ok=True)

# === GET .jpg FILES ONLY ===
image_files = [f for f in os.listdir(original_images_dir) if f.endswith(".jpg")]

# === RANDOMLY SAMPLE IMAGES ===
selected_images = random.sample(image_files, min(subset_size, len(image_files)))

# === COPY IMAGES AND CORRESPONDING LABELS ===
for image in selected_images:
    # Copy image
    shutil.copy(os.path.join(original_images_dir, image), os.path.join(subset_images_dir, image))

    # Copy label
    label_file = image.replace(".jpg", ".txt")
    src_label = os.path.join(original_labels_dir, label_file)
    dst_label = os.path.join(subset_labels_dir, label_file)

    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)

print(f"✅ Copied {len(selected_images)} images and labels to 'train_subset'")
