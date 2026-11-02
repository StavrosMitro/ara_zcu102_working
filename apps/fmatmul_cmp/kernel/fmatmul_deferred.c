// Deferred-reduction kernel (C += A * B^T), extracted VERBATIM from
// apps/fc_layer32/kernel/fmatmul.c (fmatmul_4x4_deferred_32 + its vec helper).
//
// Original authors: Matheus Cavalcante / Samuel Riedel, ETH Zurich
//                   modified versions of fmatmul_32: Stavros Mitropoulos, NTUA

#include <stddef.h>
#include "fmatmul_cmp.h"

// ---------------
// 4x4 deferred (A * B^T)
// ---------------

static inline void fmatmul_vec_4x4_deferred(float *c, const float *a,
                                            const float *b,
                                            const unsigned long int N,
                                            const unsigned long int ldc) {
  const float *a0 = a;
  const float *a1 = a + N;
  const float *a2 = a + 2 * N;
  const float *a3 = a + 3 * N;

  const float *b0 = b;
  const float *b1 = b + N;
  const float *b2 = b + 2 * N;
  const float *b3 = b + 3 * N;

  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(N));
  asm volatile("vmv.v.i v16, 0");
  asm volatile("vmv.v.i v17, 0");
  asm volatile("vmv.v.i v18, 0");
  asm volatile("vmv.v.i v19, 0");
  asm volatile("vmv.v.i v20, 0");
  asm volatile("vmv.v.i v21, 0");
  asm volatile("vmv.v.i v22, 0");
  asm volatile("vmv.v.i v23, 0");
  asm volatile("vmv.v.i v24, 0");
  asm volatile("vmv.v.i v25, 0");
  asm volatile("vmv.v.i v26, 0");
  asm volatile("vmv.v.i v27, 0");
  asm volatile("vmv.v.i v28, 0");
  asm volatile("vmv.v.i v29, 0");
  asm volatile("vmv.v.i v30, 0");
  asm volatile("vmv.v.i v31, 0");

  unsigned long int k = N;
  while (k > 0) {
    unsigned long int vl;
    asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(k));

    asm volatile("vle32.v v0, (%0)" : : "r"(a0));
    a0 += vl;
    asm volatile("vle32.v v1, (%0)" : : "r"(a1));
    a1 += vl;
    asm volatile("vle32.v v2, (%0)" : : "r"(a2));
    a2 += vl;
    asm volatile("vle32.v v3, (%0)" : : "r"(a3));
    a3 += vl;

    asm volatile("vle32.v v4, (%0)" : : "r"(b0));
    b0 += vl;
    asm volatile("vle32.v v5, (%0)" : : "r"(b1));
    b1 += vl;
    asm volatile("vle32.v v6, (%0)" : : "r"(b2));
    b2 += vl;
    asm volatile("vle32.v v7, (%0)" : : "r"(b3));
    b3 += vl;

    asm volatile("vfmacc.vv v16, v0, v4");
    asm volatile("vfmacc.vv v17, v0, v5");
    asm volatile("vfmacc.vv v18, v0, v6");
    asm volatile("vfmacc.vv v19, v0, v7");

    asm volatile("vfmacc.vv v20, v1, v4");
    asm volatile("vfmacc.vv v21, v1, v5");
    asm volatile("vfmacc.vv v22, v1, v6");
    asm volatile("vfmacc.vv v23, v1, v7");

    asm volatile("vfmacc.vv v24, v2, v4");
    asm volatile("vfmacc.vv v25, v2, v5");
    asm volatile("vfmacc.vv v26, v2, v6");
    asm volatile("vfmacc.vv v27, v2, v7");

    asm volatile("vfmacc.vv v28, v3, v4");
    asm volatile("vfmacc.vv v29, v3, v5");
    asm volatile("vfmacc.vv v30, v3, v6");
    asm volatile("vfmacc.vv v31, v3, v7");

    k -= vl;
  }

  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(N));
  asm volatile("vmv.v.i v8, 0");

  asm volatile("vredsum.vs v16, v16, v8");
  asm volatile("vredsum.vs v17, v17, v8");
  asm volatile("vredsum.vs v18, v18, v8");
  asm volatile("vredsum.vs v19, v19, v8");
  asm volatile("vredsum.vs v20, v20, v8");
  asm volatile("vredsum.vs v21, v21, v8");
  asm volatile("vredsum.vs v22, v22, v8");
  asm volatile("vredsum.vs v23, v23, v8");
  asm volatile("vredsum.vs v24, v24, v8");
  asm volatile("vredsum.vs v25, v25, v8");
  asm volatile("vredsum.vs v26, v26, v8");
  asm volatile("vredsum.vs v27, v27, v8");
  asm volatile("vredsum.vs v28, v28, v8");
  asm volatile("vredsum.vs v29, v29, v8");
  asm volatile("vredsum.vs v30, v30, v8");
  asm volatile("vredsum.vs v31, v31, v8");

  float *c_ptr = c;
  float res;

  asm volatile("vfmv.f.s %0, v16" : "=f"(res));
  c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v17" : "=f"(res));
  c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v18" : "=f"(res));
  c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v19" : "=f"(res));
  c_ptr[3] += res;

  c_ptr += ldc;
  asm volatile("vfmv.f.s %0, v20" : "=f"(res));
  c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v21" : "=f"(res));
  c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v22" : "=f"(res));
  c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v23" : "=f"(res));
  c_ptr[3] += res;

  c_ptr += ldc;
  asm volatile("vfmv.f.s %0, v24" : "=f"(res));
  c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v25" : "=f"(res));
  c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v26" : "=f"(res));
  c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v27" : "=f"(res));
  c_ptr[3] += res;

  c_ptr += ldc;
  asm volatile("vfmv.f.s %0, v28" : "=f"(res));
  c_ptr[0] += res;
  asm volatile("vfmv.f.s %0, v29" : "=f"(res));
  c_ptr[1] += res;
  asm volatile("vfmv.f.s %0, v30" : "=f"(res));
  c_ptr[2] += res;
  asm volatile("vfmv.f.s %0, v31" : "=f"(res));
  c_ptr[3] += res;
}

void fmatmul_4x4_deferred_32(float *c, const float *a, const float *b,
                          const unsigned long int M, const unsigned long int N,
                          const unsigned long int P) {
  // C = A * B^T
  // A: [M x N], B: [P x N], C: [M x P]
  for (unsigned long int m = 0; m < M; m += 4) {
    for (unsigned long int p = 0; p < P; p += 4) {
      const float *a_block = a + m * N;
      const float *b_block = b + p * N;
      float *c_block = c + m * P + p;

      fmatmul_vec_4x4_deferred(c_block, a_block, b_block, N, P);
    }
  }
}
