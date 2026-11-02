//
// File:        matrix.h
// Description: interface of matrix computation
// Author:      Haris Wang
//
// #include <stdlib.h>

void matrix_multiply(const float *a, const float *b, float *c, const int M, const int N, const int K);
// Same GEMM with an explicit skip threshold on `a`. matrix_multiply()'s 1e-5
// cutoff is an ABSOLUTE magnitude, only meaningful at the scale it was tuned
// for. Gradient GEMMs now carry a 1/batchsize factor (applied once in
// cross_entropy_loss), so they must scale the cutoff to match.
void matrix_multiply_eps(const float *a, const float *b, float *c, const int M, const int N, const int K, const float eps);
void matrix_transpose(float *x, int m, int n);

// ---------------------------------------------------------------------------
// Vectorized (Ara/RVV) FP32 matmul family, implemented in matrix_vec.c.
// Used by the vectorized fc_layer.c. Scratch pools are shared across calls.
// ---------------------------------------------------------------------------
void matrix_multiply_32(const float *a, const float *b, float *c, const int M, const int N, const int K);
void matrix_multiply_fused_32(const float *a, const float *b, const float *bias,
                           float *c, const int M, const int N, const int K);
void matrix_multiply_nt_32(const float *a, const float *b, float *c,
                        const int M, const int N, const int K);
void matrix_multiply_tn_32(const float *a, const float *b, float *c,
                        const int M, const int N, const int K);
void matrix_multiply_nt_deferred_32(const float *a, const float *b, float *c,
                                 const int M, const int N, const int K);
int matrix_multiply_nt_verify_32(const float *a, const float *b,
                              const int M, const int N, const int K,
                              const float eps);
int matrix_multiply_tn_verify_32(const float *a, const float *b,
                              const int M, const int N, const int K,
                              const float eps);
void matrix_transpose_32(float *x, int m, int n);

extern float fmatmul_c_scratch_32[];
extern float shared_memory_pool_32[];
