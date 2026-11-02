//
// File:        matrix.h
// Description: interface of matrix computation
// Author:      Haris Wang
//
// #include <stdlib.h>

void matrix_multiply(const _Float16 *a, const _Float16 *b, float *c, const int M, const int N, const int K);
void matrix_multiply_fused(const _Float16 *a, const _Float16 *b, const float *bias,
						   float *c, const int M, const int N, const int K);
void matrix_multiply_nt(const _Float16 *a, const _Float16 *b, float *c,
						const int M, const int N, const int K);
// FP16-output d_input (c = a * b^T, c is _Float16). Overwrite semantics.
void matrix_multiply_nt_f16out(const _Float16 *a, const _Float16 *b, _Float16 *c,
                               const int M, const int N, const int K);
void matrix_multiply_tn(const _Float16 *a, const _Float16 *b, float *c,
                        const int M, const int N, const int K);
void matrix_multiply_nt_deferred(const _Float16 *a, const _Float16 *b, float *c,
                                 const int M, const int N, const int K);
int matrix_multiply_nt_verify(const _Float16 *a, const _Float16 *b,
							  const int M, const int N, const int K,
							  const float eps);
int matrix_multiply_tn_verify(const _Float16 *a, const _Float16 *b,
                              const int M, const int N, const int K,
                              const float eps);
void matrix_transpose(float *x, int m, int n);
void matrix_multiply_scalar(const _Float16 *a, const _Float16 *b, float *c,
                                   const int M, const int N, const int K);

// extern float fmatmul_a_scratch[];
extern float fmatmul_c_scratch[];
extern _Float16 fmatmul_c_scratch_f16[];
extern float shared_memory_pool[];