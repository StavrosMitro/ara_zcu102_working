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

// Author: Matheus Cavalcante, ETH Zurich
//         Samuel Riedel, ETH Zurich
//         Matteo Perotti, ETH Zurich

#include <string.h>

#include "kernel/fmatmul.h"
#include "runtime.h"
#include "util.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// DIAG (temporary): silence ALL printf so we can test whether everything EXCEPT
// printing works. If fmatmul now runs to completion (kernel checkpoints advance,
// EOC set), printf is conclusively the only problem. Revert by deleting this line.
#define printf(...) ((void)0)

// Define Matrix dimensions:
// C = AB with A=[MxN], B=[NxP], C=[MxP]
extern uint64_t M;
extern uint64_t N;
extern uint64_t P;

extern double a[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double b[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern double c[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
// Gold results
extern double g[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

#define THRESHOLD 0.001

// Bring-up progress marker: a DIRECT store to DDR (bypasses printf/_putchar).
// ARM reads it at 0x50000070 to see how far main got when it hangs.
//   0x100 main entered      0x200 header printfs done (=> printf works)
//   0x3xx before kernel size xx     0x4xx kernel size xx done     0x500 cycles printed
#define DBG(v) do { *(volatile unsigned int *)0x50000070UL = (unsigned int)(v); __sync_synchronize(); } while (0)

// ARM-controlled checkpoint bisect. ARM writes the checkpoint number to 0x50000060.
// We publish the most recent checkpoint at 0x50000064 and halt when it matches CTRL.
#define CTRL    (*(volatile unsigned int *)0x50000060UL)
#define CKPTREG (*(volatile unsigned int *)0x50000064UL)
#define CYCTAB  ((volatile unsigned int *)0x50000090UL)
static void llc_flush_halt(void) {
    volatile unsigned int *llc = (volatile unsigned int *)0x03001000UL;
    unsigned int ways = llc[0x28/4];
    unsigned int mask = (ways >= 32) ? 0xFFFFFFFFu : ((1u << ways) - 1u);
    llc[0x8/4] = mask; llc[0xc/4] = 0; llc[0x10/4] = 1;   // flush all ways -> DDR
    // Poll CFG_FLUSH (0x8), NOT FLUSHED (0x18): FLUSHED is a live "ways with
    // no valid tags" bitmap, and axi_llc_config.sv FsmEndFlush overwrites it
    // with cfg_spm (=0) on the LAST way, so it never reaches the mask and the
    // old `while (llc[0x18/4] != mask)` spun forever. CFG_FLUSH is cleared by
    // hardware exactly when the flush FSM returns to idle.
    while (llc[0x8/4] != 0) { }                            // wait writeback done
    // No UN-FLUSH needed: FsmEndFlush already restores flushed = cfg_spm, so
    // the ways are re-enabled automatically. (The old un-flush was based on
    // the same misreading of FLUSHED and polled a condition that was already
    // true, so it was a no-op at best.)
    for (;;) { }                                           // halt
}

#define CKPT(n) do { CKPTREG = (unsigned int)(n); __sync_synchronize(); \
           if (CTRL == (unsigned int)(n)) llc_flush_halt(); } while (0)

// Verify the matrix
int verify_matrix(double *result, double *gold, size_t R, size_t C,
                  double threshold) {
  for (uint64_t i = 0; i < R; ++i) {
    for (uint64_t j = 0; j < C; ++j) {
      uint64_t idx = i * C + j;
      if (!similarity_check(result[idx], gold[idx], threshold)) {
        return (i + j) == 0 ? -1 : idx;
      }
    }
  }
  return 0;
}

int main() {
  // DIAG (zero-bitstream experiment): disable the CVA6<->Ara memory
  // consistency mode (custom CSR 0x7C2, resets to 1). This turns OFF the
  // axi_inval_filter, so Ara stores no longer wait for CVA6 to accept L1
  // invalidations -- the prime suspect for the core-complex deadlock (ILA
  // showed ALL AXI paths idle during the wedge; the inval handshake is the
  // one blocking dependency that is invisible to every AXI probe).
  // If the kernel now completes, the invalidation deadlock is CONFIRMED.
  // Risk: verify_matrix could read stale c[] lines from L1 (wrong result,
  // NOT a hang -- distinguishable; c[] is never read before the kernel, so
  // it is likely not even cached and verify should still pass).
  asm volatile("csrwi 0x7C2, 0");
  DBG(0x100);   // main entered (before any printf)
  CKPT(1);      // reached main
  printf("\n");
  printf("=============\n");
  printf("=  FMATMUL  =\n");
  printf("=============\n");
  printf("\n");
  printf("\n");
  DBG(0x200);   // header printfs done => printf works
  CKPT(2);      // header printfs done

  unsigned int ckpt_num = 3;   // 3=before first kernel, then 4=after kernel, 5=metrics printed, ...
  int iter = 0;

#ifdef VCD_DUMP
  // Measure only the full-size matmul
  for (uint64_t s = M; s <= M; s *= 2) {
#else
  for (uint64_t s = 4; s <= M; s *= 2) {
#endif
    printf("\n");
    printf("------------------------------------------------------------\n");
    printf("Calculating a (%d x %d) x (%d x %d) matrix multiplication...\n", s,
           s, s, s);
    printf("------------------------------------------------------------\n");
    printf("\n");

    // Matrices are initialized --> Start calculating
    printf("Calculating fmatmul...\n");
    DBG(0x300 + s);   // BEFORE kernel of size s (outside timed region)
  CKPT(ckpt_num++);  // about to enter this kernel
    start_timer();
    fmatmul(c, a, b, s, s, s);
    stop_timer();
    DBG(0x400 + s);   // AFTER kernel of size s completed (outside timed region)
  CKPT(ckpt_num++);  // kernel of this size done

    // Metrics
    int64_t runtime = get_timer();
    float performance = 2.0 * s * s * s / runtime;
    float utilization = 100 * performance / (2.0 * NR_LANES);

    printf("The execution took %d cycles.\n", runtime);
    printf("The performance is %f FLOP/cycle (%f%% utilization).\n",
           performance, utilization);

    // ARM-controlled bisection: log (size, cycles) and halt after size 'ctrl'.
    CYCTAB[2 * iter]     = (unsigned int)s;
    CYCTAB[2 * iter + 1] = (unsigned int)runtime;
    __sync_synchronize();
    iter++;
    CKPT(ckpt_num++);  // metrics printed

    // Verify the result only for s == M (to keep it simple)
    if (s == M) {
      printf("Verifying result...\n");
      int error = verify_matrix(c, g, s, s, THRESHOLD);
      if (error != 0) {
        printf("Error code %d\n", error);
        printf("c[%d]=%d\n", error, c[error]);
        return error;
      } else {
        printf("Passed.\n");
      }
    }
  }

  return 0;
}
