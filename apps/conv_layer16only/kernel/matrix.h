//
// File:        matrix.h
// Description: interface of matrix computation
// Author:      Haris Wang
//
// #include <stdlib.h>

void matrix_multiply(const _Float16 *a, const _Float16 *b, _Float16 *c, const int M, const int N, const int K);
void matrix_multiply_fused(const _Float16 *a, const _Float16 *b, const _Float16 *bias,
						   _Float16 *c, const int M, const int N, const int K);
void matrix_multiply_nt(const _Float16 *a, const _Float16 *b, _Float16 *c,
						const int M, const int N, const int K);
void matrix_multiply_tn(const _Float16 *a, const _Float16 *b, _Float16 *c,
                        const int M, const int N, const int K);
void matrix_multiply_nt_deferred(const _Float16 *a, const _Float16 *b, _Float16 *c,
                                 const int M, const int N, const int K);
int matrix_multiply_nt_verify(const _Float16 *a, const _Float16 *b,
							  const int M, const int N, const int K,
							  const _Float16 eps);
int matrix_multiply_tn_verify(const _Float16 *a, const _Float16 *b,
                              const int M, const int N, const int K,
                              const _Float16 eps);
void matrix_transpose(_Float16 *x, int m, int n);
void matrix_multiply_scalar(const _Float16 *a, const _Float16 *b, _Float16 *c,
                                   const int M, const int N, const int K);

// extern _Float16 fmatmul_a_scratch[];
extern _Float16 fmatmul_c_scratch[];
extern _Float16 shared_memory_pool[];