import os
import shutil
import random

# Need to manually split Self Driving Car Dataset this created the initial test, train and validation dataset folders.
# Reference: Splitting the Self Driving Dataset with assistance from Lab 06 Notebook File

# Set the path to your "export" folder
export_path = "./export"

# Set the path to the new folders (train, val, test)
train_path = "./train"
val_path = "./valid"
test_path = "./test"

# Set the split ratios (adjust as needed)
train_ratio = 0.8  # 80% for training
val_ratio = 0.1    # 10% for validation
test_ratio = 0.1   # 10% for testing

# Create the new folders if they don't exist
os.makedirs(os.path.join(train_path, "images"), exist_ok=True)
os.makedirs(os.path.join(train_path, "labels"), exist_ok=True)
os.makedirs(os.path.join(val_path, "images"), exist_ok=True)
os.makedirs(os.path.join(val_path, "labels"), exist_ok=True)
os.makedirs(os.path.join(test_path, "images"), exist_ok=True)
os.makedirs(os.path.join(test_path, "labels"), exist_ok=True)

# Get the list of image files in the "images" folder
image_folder_path = os.path.join(export_path, "images")
image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.png'))]

# Randomly shuffle the list of image files
random.shuffle(image_files)

# Calculate the number of images for each split
num_images = len(image_files)
num_train = int(train_ratio * num_images)
num_val = int(val_ratio * num_images)
num_test = int(test_ratio * num_images)

# Split the image files
train_images = image_files[:num_train]
val_images = image_files[num_train:num_train + num_val]
test_images = image_files[num_train + num_val:]

# Move the images to their respective folders
for img in train_images:
    shutil.move(os.path.join(image_folder_path, img), os.path.join(os.path.join(train_path, "images"), img))

for img in val_images:
    shutil.move(os.path.join(image_folder_path, img), os.path.join(os.path.join(val_path, "images"), img))

for img in test_images:
    shutil.move(os.path.join(image_folder_path, img), os.path.join(os.path.join(test_path, "images"), img))

# Repeat the same process for the "labels" folder

label_folder_path = os.path.join(export_path, "labels")

# Move the label files to their respective folders
for lbl in train_images:
    lbl = lbl.replace('.jpg', '.txt')
    shutil.move(os.path.join(label_folder_path, lbl), os.path.join(os.path.join(train_path, "labels"), lbl))

for lbl in val_images:
    lbl = lbl.replace('.jpg', '.txt')
    shutil.move(os.path.join(label_folder_path, lbl), os.path.join(os.path.join(val_path, "labels"), lbl))

for lbl in test_images:
    lbl = lbl.replace('.jpg', '.txt')
    shutil.move(os.path.join(label_folder_path, lbl), os.path.join(os.path.join(test_path, "labels"), lbl))

# ===Modification of the labels in cones and potholes dataset===
old_class_index = 0

def rename_index(input_folder, old_class_index, new_class_index):
    # Loop through all text files in input
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)
            
            # Open the annotation file and read all lines
            with open(file_path, 'r') as f:
                lines = f.readlines()

            updated_lines = []

            # Loop through each label line in the file
            for line in lines:
                # Split the line into components (class, x_center, y_center, width, height)
                components = line.strip().split()

                # Check if the current class index is the one to be renamed
                if int(components[0]) == old_class_index:
                    components[0] = str(new_class_index)  # Update the class index
                updated_lines.append(" ".join(components)) # Merge the lines

            # Write to file
            with open(file_path, 'w') as f:
                f.writelines("\n".join(updated_lines) + "\n")
            print(f"Updated {file_name}")

# Define the directories and their respective class indices
directories = [
    ("Pothole.v1-raw.yolov8/train/labels", 12),
    ("Pothole.v1-raw.yolov8/test/labels", 12),
    ("Pothole.v1-raw.yolov8/valid/labels", 12),
    ("ConeTest.v1i.yolov8/train/labels", 11),
    ("ConeTest.v1i.yolov8/valid/labels", 11),
    ("ConeTest.v1i.yolov8/test/labels", 11)
]

# Use enumerate to iterate over the directories
for idx, (directory, new_class_index) in enumerate(directories):
    rename_index(directory, old_class_index, new_class_index)
    
# After this, the folders were manually combined into a single folder for training, testing, and validation.