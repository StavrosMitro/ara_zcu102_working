//
// File:        matrix.h
// Description: Pure FP16 matrix computation interface
// Author:      Haris Wang
// FP16 refactor: Stavros Mitropoulos, NTUA
//

void matrix_multiply_32(const _Float16 *a, const _Float16 *b, _Float16 *c,
                        const int M, const int N, const int K);

void matrix_multiply_fused_32(const _Float16 *a, const _Float16 *b,
                               const _Float16 *bias, _Float16 *c,
                               const int M, const int N, const int K);

void matrix_multiply_nt_32(const _Float16 *a, const _Float16 *b, _Float16 *c,
                            const int M, const int N, const int K);

void matrix_multiply_tn_32(const _Float16 *a, const _Float16 *b, _Float16 *c,
                            const int M, const int N, const int K);

void matrix_multiply_nt_deferred_32(const _Float16 *a, const _Float16 *b,
                                    _Float16 *c,
                                    const int M, const int N, const int K);

int matrix_multiply_nt_verify_32(const _Float16 *a, const _Float16 *b,
                                  const int M, const int N, const int K,
                                  const _Float16 eps);

int matrix_multiply_tn_verify_32(const _Float16 *a, const _Float16 *b,
                                  const int M, const int N, const int K,
                                  const _Float16 eps);

void matrix_transpose_32(_Float16 *x, int m, int n);

extern _Float16 fmatmul_c_scratch_32[];
extern _Float16 shared_memory_pool_32[];
