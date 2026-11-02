"""
prepare_cifar100c.py — Convert CIFAR-100-C (corrupted) .npy files to the
binary format used by VGGMini (same as CIFAR-100).

CIFAR-100-C format:
    labels.npy          : (10000,)        int64  — CIFAR-100 test labels
    <corruption>.npy    : (50000, 32,32,3) uint8  — 5 severity levels stacked
                          [0:10000]  = severity 1
                          [10000:20000] = severity 2  ... etc.

Output (in kernel/):
    cifar100_data.bin    — raw pixel bytes, 3072 per image
    cifar100_labels.bin  — fine labels as int32 LE
    cifar100_offsets.bin — byte offsets into cifar100_data.bin

Usage:
    # single corruption, single severity
    python3 scripts/prepare_cifar100c.py CIFAR-100-C/ gaussian_noise --severity 3

    # mix all corruptions at a given severity (recommended for fine-tuning)
    python3 scripts/prepare_cifar100c.py CIFAR-100-C/ --all --severity 2

    # mix all corruptions at all severities
    python3 scripts/prepare_cifar100c.py CIFAR-100-C/ --all --severity all

Severity guide:
    1 = very mild   (good starting point for fine-tuning)
    2 = mild
    3 = moderate    (recommended for robust training)
    4 = severe
    5 = extreme
"""

import sys
import os
import struct
import argparse
import numpy as np

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "spatter", "saturate", "gaussian_blur", "speckle_noise",
]

IMAGE_SIZE = 3072  # 32 * 32 * 3


def severity_indices(sev):
    """Return (start, end) slice for a severity level (1-5)."""
    return (sev - 1) * 10000, sev * 10000


def load_corruption(cifar_c_dir, name, severities):
    path = os.path.join(cifar_c_dir, f"{name}.npy")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    data = np.load(path)          # (50000, 32, 32, 3) uint8
    chunks = []
    for sev in severities:
        s, e = severity_indices(sev)
        chunks.append(data[s:e])  # (10000, 32, 32, 3)
    return np.concatenate(chunks, axis=0)


def write_bins(images, labels, out_dir):
    """
    images : (N, 32, 32, 3) uint8  HWC — matches what data.c expects (src_idx = k + c*i + c*w*j)
    labels : (N,) int
    """
    n = len(images)
    assert len(labels) == n

    # Images are already HWC — flatten directly, no channel reorder needed
    flat = images.reshape(n, IMAGE_SIZE)

    with open(os.path.join(out_dir, "cifar100_data.bin"), "wb") as f:
        f.write(flat.tobytes())

    with open(os.path.join(out_dir, "cifar100_labels.bin"), "wb") as f:
        for l in labels:
            f.write(struct.pack("<i", int(l)))

    with open(os.path.join(out_dir, "cifar100_offsets.bin"), "wb") as f:
        for i in range(n):
            f.write(struct.pack("<I", i * IMAGE_SIZE))

    print(f"Wrote {n} images → kernel/cifar100_{{data,labels,offsets}}.bin")

    # Update cifar10_dataset.h so cifar10_count matches the embedded data
    header_path = os.path.join(out_dir, "cifar10_dataset.h")
    with open(header_path, "w") as f:
        f.write(f"""#ifndef CIFAR10_CIFAR10_DATASET_H
#define CIFAR10_CIFAR10_DATASET_H

#include <stddef.h>

#define cifar10_count {n}
#define cifar10_w 32
#define cifar10_h 32
#define cifar10_c 3
#define cifar10_image_bytes 3072
#define cifar10_total_bytes {n * IMAGE_SIZE}

extern const unsigned char cifar10_data[];
extern const unsigned int cifar10_offsets[];
extern const int cifar10_labels[];

#endif
""")
    print(f"Updated kernel/cifar10_dataset.h  (cifar10_count={n})")


def main():
    parser = argparse.ArgumentParser(description="Prepare CIFAR-100-C for VGGMini fine-tuning")
    parser.add_argument("cifar_c_dir", help="Path to CIFAR-100-C directory")
    parser.add_argument("corruption", nargs="?", help="Corruption name (e.g. gaussian_noise)")
    parser.add_argument("--all", action="store_true", help="Use all 19 corruption types")
    parser.add_argument("--severity", default="3",
                        help="Severity level 1-5, or 'all' for all levels (default: 3)")
    parser.add_argument("--out", default="kernel", help="Output directory (default: kernel)")
    args = parser.parse_args()

    if args.severity == "all":
        severities = [1, 2, 3, 4, 5]
    else:
        severities = [int(args.severity)]

    from dataset_root import resolve_raw
    args.cifar_c_dir = resolve_raw(args.cifar_c_dir)

    labels_path = os.path.join(args.cifar_c_dir, "labels.npy")
    all_labels_npy = np.load(labels_path).astype(np.int32)  # (50000,) or (10000,)
    # Normalise to base 10000-entry block (same labels repeat across severity levels)
    base_labels = all_labels_npy[:10000]

    corruptions = ALL_CORRUPTIONS if args.all else [args.corruption]
    if not args.all and args.corruption is None:
        parser.error("Provide a corruption name or --all")

    print(f"Corruptions : {corruptions}")
    print(f"Severities  : {severities}")

    all_images = []
    all_labels = []

    for name in corruptions:
        imgs = load_corruption(args.cifar_c_dir, name, severities)
        if imgs is None:
            continue
        n_imgs = len(imgs)
        # tile base 10000 labels to match number of loaded images
        lbl = np.tile(base_labels, len(severities))[:n_imgs]
        all_images.append(imgs)
        all_labels.append(lbl)
        print(f"  loaded {name}: {n_imgs} images")

    if not all_images:
        print("No data loaded. Check the directory path.")
        sys.exit(1)

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Shuffle so different corruption types are interleaved in training
    idx = np.random.permutation(len(images))
    images = images[idx]
    labels = labels[idx]

    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", args.out))
    write_bins(images, labels, out_dir)

    print()
    print("Next steps:")
    print(f"  python3 scripts/gen_zero_weights.py small 100")
    print(f"  make -B VARIANT=small DATASET=cifar100 FINETUNE=1 BATCHSIZE=16 EPOCHS=20")


if __name__ == "__main__":
    main()
