#ifndef CIFAR10_CIFAR10_DATASET_H
#define CIFAR10_CIFAR10_DATASET_H

#include <stddef.h>

// Σταθερές (hardcoded εφόσον ξέρουμε το dataset)
#define cifar10_count 10000
#define cifar10_w 32
#define cifar10_h 32
#define cifar10_c 3
#define cifar10_image_bytes 3072
#define cifar10_total_bytes 30720000

// Τα σύμβολα που έρχονται από την Assembly
extern const unsigned char cifar10_data[];
extern const int cifar10_labels[];

#endif