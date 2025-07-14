import os
import pandas as pd
from tqdm import tqdm

# Set your dataset root directory
root_dir = "D:/projects/new_shop_model/SKU110K_fixed"

splits = ['train', 'val']
for split in splits:
    # Build paths
    csv_path = os.path.join(root_dir, f'annotations/annotations_{split}.csv')
    img_dir = os.path.join(root_dir, f'images/{split}')
    label_dir = os.path.join(root_dir, f'labels/{split}')
    os.makedirs(label_dir, exist_ok=True)

    df = pd.read_csv(csv_path, header=None, names=[
        "image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"
    ])
    df.columns = df.columns.str.strip()  
    grouped = df.groupby('image_name') 


    for image_name, group in tqdm(grouped):
        label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))

        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                x1 = row['x1']
                y1 = row['y1']
                x2 = row['x2']
                y2 = row['y2']

                w = row['image_width']
                h = row['image_height']

                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h
                x_center = (x1 + (x2 - x1) / 2) / w
                y_center = (y1 + (y2 - y1) / 2) / h

                f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")

print("✅ Conversion to YOLO format complete!")
