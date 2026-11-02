"""
prepare_cifar100.py — Convert CIFAR-100 binary to the format used by VGGMini.

CIFAR-100 binary record layout:
    [1 byte coarse_label][1 byte fine_label][3072 bytes RGB pixels in CHW order]
    CHW: first 1024 bytes = Red channel (row-major), then Green, then Blue.

Output (in kernel/):
    cifar100_data.bin    — pixels converted to HWC interleaved (matches data.c src_idx)
    cifar100_labels.bin  — fine labels as int32 LE (4 bytes per image)
    cifar100_offsets.bin — byte offset of each image in cifar100_data.bin (uint32 LE)

data.c loads images with: src_idx = k + c*i + c*w*j  (HWC / interleaved RGB)
The raw CIFAR-100 binary is CHW (planar), so we convert CHW→HWC here.

Usage:
    python3 scripts/prepare_cifar100.py cifar-100-binary/test.bin   [10000 images]
    python3 scripts/prepare_cifar100.py cifar-100-binary/train.bin  [50000 images]
"""

import sys
import struct
import os
import numpy as np

RECORD_SIZE = 3074   # 1 coarse + 1 fine + 3072 pixels
IMAGE_SIZE  = 3072
W = H = 32
C = 3

def prepare(bin_path, out_dir):
    with open(bin_path, 'rb') as f:
        raw = f.read()

    n = len(raw) // RECORD_SIZE
    assert len(raw) == n * RECORD_SIZE, "File size not a multiple of record size"
    print(f"Found {n} images in {bin_path}")

    pixel_data = bytearray()
    labels     = []

    for i in range(n):
        base = i * RECORD_SIZE
        fine_label = raw[base + 1]          # byte 1 = fine label (0-99)
        # CIFAR binary is CHW (planar); convert to HWC (interleaved) for data.c
        chw = np.frombuffer(raw[base + 2 : base + 2 + IMAGE_SIZE], dtype=np.uint8)
        chw = chw.reshape(C, H, W)
        hwc = chw.transpose(1, 2, 0)        # CHW → HWC
        labels.append(fine_label)
        pixel_data.extend(hwc.tobytes())

    prefix = os.path.join(out_dir, "cifar100")

    with open(f"{prefix}_data.bin", "wb") as f:
        f.write(pixel_data)

    with open(f"{prefix}_labels.bin", "wb") as f:
        for l in labels:
            f.write(struct.pack("<i", l))    # int32 little-endian (matches cifar10)

    with open(f"{prefix}_offsets.bin", "wb") as f:
        for i in range(n):
            f.write(struct.pack("<I", i * IMAGE_SIZE))   # uint32 LE byte offset

    print(f"Wrote kernel/cifar100_{{data,labels,offsets}}.bin  ({n} images)")

    # Update cifar10_dataset.h with the correct image count
    header_path = os.path.join(out_dir, "cifar10_dataset.h")
    total_bytes  = n * IMAGE_SIZE
    header = f"""#ifndef CIFAR10_CIFAR10_DATASET_H
#define CIFAR10_CIFAR10_DATASET_H

#include <stddef.h>

// Σταθερές (hardcoded εφόσον ξέρουμε το dataset)
#define cifar10_count {n}
#define cifar10_w 32
#define cifar10_h 32
#define cifar10_c 3
#define cifar10_image_bytes 3072
#define cifar10_total_bytes {total_bytes}

// Τα σύμβολα που έρχονται από την Assembly
extern const unsigned char cifar10_data[];
extern const unsigned int cifar10_offsets[];
extern const int cifar10_labels[];

#endif
"""
    with open(header_path, "w") as f:
        f.write(header)
    print(f"Updated kernel/cifar10_dataset.h  (cifar10_count={n})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    from dataset_root import resolve_raw
    bin_path = resolve_raw(sys.argv[1])
    out_dir  = os.path.join(os.path.dirname(__file__), "..", "kernel")
    out_dir  = os.path.normpath(out_dir)
    prepare(bin_path, out_dir)

    print()
    print("Next steps:")
    print("  python3 scripts/gen_zero_weights.py 100")
    print("  make -B DATASET=cifar100 BATCHSIZE=128 EPOCHS=50")
