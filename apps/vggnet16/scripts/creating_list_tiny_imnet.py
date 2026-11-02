import os
import random  # Don't forget to import random

# Define paths
# Adjust this if you are running the script from outside ~/tiny-imagenet-200
dataset_dir = os.path.expanduser('~/tiny-imagenet-200/train')
output_file = 'images.list'

def generate_label_file(data_dir, output_path):
    # Get all class directories and sort them for deterministic label indexing
    try:
        class_folders = sorted([
            d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))
        ])
    except FileNotFoundError:
        print(f"Error: The directory '{data_dir}' was not found.")
        return

    all_lines = []  # Create an empty list to store the lines
    
    for label_index, class_folder in enumerate(class_folders):
        images_dir = os.path.join(data_dir, class_folder, 'images')
        
        # Skip if the images subfolder doesn't exist
        if not os.path.exists(images_dir):
            continue
            
        # Loop through all JPEGs in the images folder
        for image_file in sorted(os.listdir(images_dir)):
            if image_file.endswith('.JPEG'):
                # Construct the relative path to the image
                image_path = os.path.join(data_dir, class_folder, 'images', image_file)
                
                # Append the formatted string to our list instead of writing to a file
                all_lines.append(f"{label_index} {image_path}\n")
                
    # Shuffle the list of strings
    # Note: If you want exactly the same shuffle every time you run the script,
    # you can add `random.seed(42)` right before the shuffle line.
    random.shuffle(all_lines)
    
    # Write the shuffled lines to the output file all at once
    with open(output_path, 'w') as f:
        f.writelines(all_lines)
                
    print(f"Success! Wrote {len(all_lines)} shuffled image paths to {output_path}.")
    print(f"Total classes found: {len(class_folders)}")

if __name__ == '__main__':
    generate_label_file(dataset_dir, output_file)