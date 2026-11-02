#ifndef CIFAR10_CIFAR10_DATASET_H
#define CIFAR10_CIFAR10_DATASET_H

#include <stddef.h>

// Σταθερές (hardcoded εφόσον ξέρουμε το dataset)
// NOTE: temporarily reduced to a 256-image subset for fast Spike bring-up.
// The full 50000-image bins are backed up as kernel/cifar100_*.bin.full; restore
// them (and set cifar10_count 50000 / cifar10_total_bytes 153600000) for a full
// run, or regenerate via scripts/prepare_cifar100c.py for CIFAR-100-C adaptation.
#define cifar10_count 64
#define cifar10_w 32
#define cifar10_h 32
#define cifar10_c 3
#define cifar10_image_bytes 3072
#define cifar10_total_bytes 196608

// Τα σύμβολα που έρχονται από την Assembly
extern const unsigned char cifar10_data[];
extern const unsigned int cifar10_offsets[];
extern const int cifar10_labels[];

#endif
