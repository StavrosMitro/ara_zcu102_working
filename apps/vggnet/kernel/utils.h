//
// File:        utils.h
// Description: Shared element-wise helpers used across the whole network.
//
// This build of vggnet is the fully vectorized (Ara / RVV) version, so these are
// simply THE memset/memcpy/scale/accumulate primitives every layer uses -- conv,
// fc, batchnorm, pooling, dropout and the training loop. They were previously
// duplicated as file-local statics in convolution_layer.c / fc_layer.c /
// train.c (copied from fc_layer32 and conv_layer32); collected here so there is
// one implementation.
//
// All routines are fp32 and use LMUL=8 (m8) unit-strided accesses, the widest
// and most bandwidth-efficient form on Ara. Lengths need no alignment or
// multiple-of-VL padding -- the vsetvli stripmining loop handles the tail.
//
#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

// dst[0..n) = 0
void memset_vectorized_zero_f32(float *dst, size_t n_elements);

// dst[0..n) = value
void memset_vectorized_value_f32(float *dst, float value, size_t n_elements);

// dst[0..n) = src[0..n)   (non-overlapping)
void memcpy_vectorized_f32(float *dst, const float *src, size_t n_elements);

// buf[0..n) = 0   (int-length variant, matches the older zero_f32_vec callers)
void zero_f32_vec(float *buf, int n);

// vec[i] *= scale
void vector_scale_f32(float *vec, float scale, int length);

// dst[i] += src[i]
void vec_add_f32(float *dst, const float *src, size_t n_elements);

// dst[i] = dst[i] * a + b   (in-place affine; e.g. BatchNorm's gamma/beta)
void vec_affine_f32(float *dst, float a, float b, size_t n_elements);

#endif /* UTILS_H */
