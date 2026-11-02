import os
import sys
from PIL import Image

def convert_image_to_stbi_format(image_path, output_c_path="image_data.c", array_name="img_data"):
    expanded_path = os.path.expanduser(image_path)
    
    if not os.path.exists(expanded_path):
        print(f"Error: Could not find image at {expanded_path}")
        return
        
    try:
        img = Image.open(expanded_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    width, height = img.size
    channels = 3
    
    print(f"Processing image: {width}x{height}, {channels} channels...")

    # stbi_load returns interleaved data (RGB, RGB, RGB...)
    pixels = list(img.getdata())
    flat_pixels = []
    for r, g, b in pixels:
        # Keep as integers 0-255 to match stbi_load's unsigned char output
        flat_pixels.extend([r, g, b])

    with open(output_c_path, 'w') as f:
        f.write(f"// Generated from: {os.path.basename(expanded_path)}\n")
        f.write(f"// Format: Interleaved (RGB RGB RGB...)\n")
        f.write(f"// Matches stbi_load output exactly.\n\n")
        
        f.write(f"const int {array_name}_w = {width};\n")
        f.write(f"const int {array_name}_h = {height};\n")
        f.write(f"const int {array_name}_c = {channels};\n\n")
        
        # Using unsigned char to perfectly replace the stbi_load data pointer
        f.write(f"const unsigned char {array_name}[] = {{\n")
        
        lines = []
        for i in range(0, len(flat_pixels), 15):
            chunk = flat_pixels[i:i+15]
            line = "    " + ", ".join(str(p) for p in chunk)
            lines.append(line)
        
        f.write(",\n".join(lines))
        f.write("\n};\n")

    print(f"Success! Wrote C array to {output_c_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python img_to_c.py <path_to_image.jpg> [output_file.c]")
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else "image.c"
        convert_image_to_stbi_format(in_path, out_path)