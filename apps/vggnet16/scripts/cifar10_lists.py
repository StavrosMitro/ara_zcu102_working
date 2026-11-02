import os
import random

# Define paths
dataset_dir = os.path.expanduser('~/scripts/cifar10_images')
train_output = 'images.list'
val_output = 'image_val.list'
split_ratio = 0.8  # 80% Train, 20% Val

def generate_splits(data_dir, train_file, val_file, split_ratio):
    # Sort class folders for deterministic label indexing (0 to 9)
    try:
        class_folders = sorted([
            d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))
        ])
    except FileNotFoundError:
        print(f"Error: The directory '{data_dir}' was not found.")
        return

    train_lines = []
    val_lines = []

    for label_index, class_folder in enumerate(class_folders):
        class_dir = os.path.join(data_dir, class_folder)
        
        # Get all .jpg images in this specific class
        images = [img for img in os.listdir(class_dir) if img.endswith('.jpg')]
        
        # 1. Shuffle the class images to ensure a random 80/20 selection
        random.shuffle(images)
        
        # 2. Calculate the exact index to split the list
        split_idx = int(len(images) * split_ratio)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # 3. Format and append paths to the global lists
        for img in train_images:
            img_path = os.path.join(data_dir, class_folder, img)
            train_lines.append(f"{label_index} {img_path}\n")
            
        for img in val_images:
            img_path = os.path.join(data_dir, class_folder, img)
            val_lines.append(f"{label_index} {img_path}\n")

    # 4. Shuffle the global lists so batches are thoroughly mixed
    random.shuffle(train_lines)
    random.shuffle(val_lines)

    # 5. Write the shuffled lists to their respective files
    with open(train_file, 'w') as f:
        f.writelines(train_lines)
        
    with open(val_file, 'w') as f:
        f.writelines(val_lines)

    # Print summary
    print("Success!")
    print(f"Total classes mapped: {len(class_folders)}")
    print(f"Saved {len(train_lines)} shuffled training images to '{train_file}'")
    print(f"Saved {len(val_lines)} shuffled validation images to '{val_file}'")

if __name__ == '__main__':
    generate_splits(dataset_dir, train_output, val_output, split_ratio)