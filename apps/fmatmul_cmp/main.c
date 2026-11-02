// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Cycle comparison of the two FP32 matmul strategies in fc_layer32:
//   fmatmul_tn_32            -- scalar-broadcast (vfmacc.vf), reduction in regs
//   fmatmul_4x4_deferred_32  -- vector-vector (vfmacc.vv), deferred vredsum
//
// Both are driven to compute the same product, C[N,P] = A^T * B, over the same
// 2*M*N*P FLOPs. The deferred kernel computes a * b^T, so it is handed the
// pre-transposed operands (A^T, B^T) that script/gen_data.py emits.

#include <string.h>

#include "kernel/fmatmul_cmp.h"
#include "runtime.h"
#include "util.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// C = A^T B with A=[MxN], B=[MxP], C=[NxP]
extern uint64_t M;
extern uint64_t N;
extern uint64_t P;

extern float a[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float b[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
// Pre-transposed operands for the deferred kernel
extern float at[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float bt[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

extern float c_tn[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float c_df[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
// Gold results. fmatmul_tn_32 folds a 1/M scaling into the kernel, the deferred
// one does not, hence two golds.
extern float g_tn[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern float g_df[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

#define THRESHOLD 0.01f

// Peak FP32 throughput: 4 FLOP/cycle/lane (2 elements per 64-bit lane, 1 FMA each)
#define PEAK_FLOP_PER_CYCLE (4.0 * NR_LANES)

#define MAX_SWEEP 16

static int verify_matrix(const float *result, const float *gold, size_t R,
                         size_t C, float threshold) {
  for (uint64_t i = 0; i < R; ++i) {
    for (uint64_t j = 0; j < C; ++j) {
      uint64_t idx = i * C + j;
      if (!similarity_check_32b(result[idx], gold[idx], threshold)) {
        return (i + j) == 0 ? -1 : (int)idx;
      }
    }
  }
  return 0;
}

int main() {
  int64_t cyc_tn[MAX_SWEEP];
  int64_t cyc_df[MAX_SWEEP];
  uint64_t size[MAX_SWEEP];
  int iter = 0;

  printf("\n");
  printf("=========================\n");
  printf("=  FMATMUL TN vs DEFER  =\n");
  printf("=========================\n");
  printf("\n");
  printf("C[N,P] = A^T * B   (A=[M,N], B=[M,P])\n");
  printf("  tn      : fmatmul_tn_32(c, a, b, s, s, s)\n");
  printf("  deferred: fmatmul_4x4_deferred_32(c, a^T, b^T, s, s, s)\n");
  printf("\n");

#ifdef VCD_DUMP
  // Measure only the full-size matmul
  for (uint64_t s = M; s <= M; s *= 2) {
#else
  for (uint64_t s = 4; s <= M && iter < MAX_SWEEP; s *= 2) {
#endif
    printf("------------------------------------------------------------\n");
    printf("(%lu x %lu)^T x (%lu x %lu) -> (%lu x %lu)\n", s, s, s, s, s, s);
    printf("------------------------------------------------------------\n");

    // ---- TN: reduction lives in the accumulator registers ----
    // The kernel overwrites c, so no zeroing is needed.
    start_timer();
    fmatmul_tn_32(c_tn, a, b, s, s, s);
    stop_timer();
    int64_t runtime_tn = get_timer();

    // ---- Deferred: vfmacc.vv + a final vredsum per output element ----
    // This kernel accumulates (c += a*b^T), so c must be zeroed first. The
    // zeroing is left OUT of the timed region: it is a property of how the
    // kernel is used, not of the kernel itself.
    for (uint64_t i = 0; i < s * s; ++i)
      c_df[i] = 0.0f;

    start_timer();
    fmatmul_4x4_deferred_32(c_df, at, bt, s, s, s);
    stop_timer();
    int64_t runtime_df = get_timer();

    float flop = 2.0 * s * s * s;
    float perf_tn = flop / runtime_tn;
    float perf_df = flop / runtime_df;

    printf("  tn      : %ld cycles, %f FLOP/cycle (%f%% util)\n", runtime_tn,
           perf_tn, 100 * perf_tn / PEAK_FLOP_PER_CYCLE);
    printf("  deferred: %ld cycles, %f FLOP/cycle (%f%% util)\n", runtime_df,
           perf_df, 100 * perf_df / PEAK_FLOP_PER_CYCLE);
    printf("  deferred/tn = %f\n", (float)runtime_df / (float)runtime_tn);
    printf("\n");

    size[iter] = s;
    cyc_tn[iter] = runtime_tn;
    cyc_df[iter] = runtime_df;
    iter++;

    // Only the full size uses the whole gold matrix: for s < M the leading
    // s x s block of a^T is not the transpose of the leading s x s block of a,
    // so the small sizes are timing-only.
    if (s == M) {
      printf("Verifying results...\n");
      int e_tn = verify_matrix(c_tn, g_tn, s, s, THRESHOLD);
      if (e_tn != 0) {
        printf("tn: error at index %d\n", e_tn);
        return e_tn;
      }
      printf("  tn      : passed.\n");

      int e_df = verify_matrix(c_df, g_df, s, s, THRESHOLD);
      if (e_df != 0) {
        printf("deferred: error at index %d\n", e_df);
        return e_df;
      }
      printf("  deferred: passed.\n");
      printf("\n");
    }
  }

  printf("============================================\n");
  printf("  size     tn cycles   deferred cycles   ratio\n");
  printf("============================================\n");
  for (int i = 0; i < iter; ++i) {
    printf("  %4lu     %9ld   %15ld   %f\n", size[i], cyc_tn[i], cyc_df[i],
           (float)cyc_df[i] / (float)cyc_tn[i]);
  }
  printf("\n");

  return 0;
}
