//
// File:        matrix.h
// Description: interface of matrix computation
// Author:      Haris Wang
//
// #include <stdlib.h>

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
void matrix_multiply_scalar(const float *a, const float *b, float *c,
                                   const int M, const int N, const int K);

// extern float fmatmul_a_scratch[];
extern float fmatmul_c_scratch_32[];
extern float shared_memory_pool_32[];