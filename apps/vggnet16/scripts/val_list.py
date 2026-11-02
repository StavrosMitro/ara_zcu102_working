import os

# Define paths using expanduser to handle the '~' correctly
train_dir = os.path.expanduser('~/tiny-imagenet-200/train')
val_dir = os.path.expanduser('~/tiny-imagenet-200/val')
val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
output_file = 'image_val.list'

def generate_val_labels(train_dir, val_annotations_file, output_path):
    # STEP 1: Build the exact same mapping used for the training set
    try:
        class_folders = sorted([
            d for d in os.listdir(train_dir) 
            if os.path.isdir(os.path.join(train_dir, d))
        ])
    except FileNotFoundError:
        print(f"Error: Train directory '{train_dir}' not found.")
        return

    # Create a dictionary mapping: {'n01443537': 0, 'n01629819': 1, ...}
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_folders)}

    # STEP 2: Read val annotations and map to label indices
    count = 0
    with open(output_path, 'w') as out_f:
        with open(val_annotations_file, 'r') as in_f:
            for line in in_f:
                # Split the line into columns (handles tabs or spaces)
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                    
                image_name = parts[0]  # e.g., 'val_0.JPEG'
                class_name = parts[1]  # e.g., 'n03444034'
                
                # Look up the integer label
                if class_name in class_to_idx:
                    label_index = class_to_idx[class_name]
                    
                    # Construct the relative path to match your layout
                    image_path = os.path.join('tiny-imagenet-200/val/images', image_name)
                    
                    # Write '<label_index> <image_path>'
                    out_f.write(f"{label_index} {image_path}\n")
                    count += 1
                else:
                    print(f"Warning: Class {class_name} found in val but not in train!")

    print(f"Success! Wrote {count} validation image paths to {output_path}.")

if __name__ == '__main__':
    generate_val_labels(train_dir, val_annotations_file, output_file)