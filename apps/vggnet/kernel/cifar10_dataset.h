#ifndef CIFAR10_CIFAR10_DATASET_H
#define CIFAR10_CIFAR10_DATASET_H

#include <stddef.h>

#define cifar10_count 400
#define cifar10_w 32
#define cifar10_h 32
#define cifar10_c 3
#define cifar10_image_bytes 3072
#define cifar10_total_bytes 1228800

extern const unsigned char cifar10_data[];
extern const unsigned int cifar10_offsets[];
extern const int cifar10_labels[];

#endif
