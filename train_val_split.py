import os
import shutil
import random
from pathlib import Path

# Set manually in Colab
data_path = '/content/custom_data'  # Should contain 'images' and 'labels' subfolders
train_percent = 0.9

# Input paths
input_image_path = os.path.join(data_path, 'images')
input_label_path = os.path.join(data_path, 'labels')

# Output paths
train_img_path = '/content/data/train/images'
train_lbl_path = '/content/data/train/labels'
val_img_path = '/content/data/validation/images'
val_lbl_path = '/content/data/validation/labels'

# Create output folders
for dir_path in [train_img_path, train_lbl_path, val_img_path, val_lbl_path]:
    os.makedirs(dir_path, exist_ok=True)

# Get list of images
img_file_list = list(Path(input_image_path).rglob('*.*'))
random.shuffle(img_file_list)

train_num = int(len(img_file_list) * train_percent)
train_files = img_file_list[:train_num]
val_files = img_file_list[train_num:]

print(f'Train images: {len(train_files)}')
print(f'Val images: {len(val_files)}')

def copy_files(file_list, img_dest, lbl_dest):
    for img_path in file_list:
        img_fn = img_path.name
        base_fn = img_path.stem
        txt_fn = base_fn + '.txt'
        txt_path = os.path.join(input_label_path, txt_fn)

        shutil.copy(img_path, os.path.join(img_dest, img_fn))
        if os.path.exists(txt_path):
            shutil.copy(txt_path, os.path.join(lbl_dest, txt_fn))

copy_files(train_files, train_img_path, train_lbl_path)
copy_files(val_files, val_img_path, val_lbl_path)
