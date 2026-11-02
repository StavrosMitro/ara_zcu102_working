// Kernels under comparison, both taken verbatim from apps/fc_layer32/kernel/fmatmul.c.
//
//   fmatmul_tn_32        C[N,P] = (A^T * B) / M     A=[M,N]  B=[M,P]
//                        Strided scalar broadcast of A + vfmacc.vf; the reduction
//                        over M happens in the accumulator registers. Also folds
//                        in the 1/M batch scaling at the end.
//
//   fmatmul_4x4_deferred_32
//                        C[M,P] += A * B^T          A=[M,N]  B=[P,N]
//                        Both operands read row-contiguous, vfmacc.vv into 16
//                        accumulator vectors, one vredsum.vs per output element
//                        at the very end (the "deferred" reduction).
//
// The two compute different products, so main.c feeds the deferred kernel the
// pre-transposed operands (A^T, B^T); both then produce the same C[N,P] and do
// the same 2*M*N*P FLOPs.

#ifndef FMATMUL_CMP_H
#define FMATMUL_CMP_H

// ---- TN: C = A^T * B, scaled by 1/M ----
void fmatmul_tn_32(float *c, const float *a, const float *b,
                   const unsigned long int M, const unsigned long int N,
                   const unsigned long int P);

void fmatmul_4x4_tn_32(float *c, const float *a, const float *b,
                       const unsigned long int M, const unsigned long int N,
                       const unsigned long int P);
void fmatmul_8x8_tn_32(float *c, const float *a, const float *b,
                       const unsigned long int M, const unsigned long int N,
                       const unsigned long int P);
void fmatmul_16x16_tn_32(float *c, const float *a, const float *b,
                         const unsigned long int M, const unsigned long int N,
                         const unsigned long int P);

void fmatmul_vec_4x4_slice_init_32();
void fmatmul_vec_8x8_slice_init_32();
void fmatmul_vec_16x16_slice_init_32();

void fmatmul_vec_4x4_tn_32(float *c, const float *a, const float *b,
                           const unsigned long int N, const unsigned long int P,
                           const unsigned long int lda);
void fmatmul_vec_8x8_tn_32(float *c, const float *a, const float *b,
                           const unsigned long int N, const unsigned long int P,
                           const unsigned long int lda);
void fmatmul_vec_16x16_tn_32(float *c, const float *a, const float *b,
                             const unsigned long int N,
                             const unsigned long int P,
                             const unsigned long int lda);

// ---- Deferred: C += A * B^T ----
void fmatmul_4x4_deferred_32(float *c, const float *a, const float *b,
                             const unsigned long int M,
                             const unsigned long int N,
                             const unsigned long int P);

#endif // FMATMUL_CMP_H
