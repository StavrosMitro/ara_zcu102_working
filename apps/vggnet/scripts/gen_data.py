import os
import struct
from PIL import Image

# Ρυθμίσεις
LIST_FILE = "cifar10/images10k.list"
OUT_DATA = "src/cifar10_data.bin"
OUT_OFFSETS = "src/cifar10_offsets.bin"
OUT_LABELS = "src/cifar10_labels.bin"

def generate_binary():
    with open(LIST_FILE, 'r') as f:
        lines = f.readlines()

    data_bytes = bytearray()
    labels = []
    offsets = [0]
    current_offset = 0

    print(f"Converting {len(lines)} images to binary...")

    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        label = int(parts[0])
        img_path = parts[1]
        
        img = Image.open(img_path).convert('RGB')
        raw_pixels = img.tobytes() # 3072 bytes για 32x32x3
        
        data_bytes.extend(raw_pixels)
        labels.append(label)
        
        current_offset += len(raw_pixels)
        offsets.append(current_offset)

    # Αποθήκευση Pixels
    with open(OUT_DATA, 'wb') as f:
        f.write(data_bytes)

    # Αποθήκευση Offsets (ως unsigned int - 4 bytes το καθένα)
    with open(OUT_OFFSETS, 'wb') as f:
        for offset in offsets:
            f.write(struct.pack('I', offset))

    # Αποθήκευση Labels (ως int - 4 bytes το καθένα)
    with open(OUT_LABELS, 'wb') as f:
        for label in labels:
            f.write(struct.pack('i', label))

    print(f"Success! Generated binary files in src/")

if __name__ == "__main__":
    generate_binary()