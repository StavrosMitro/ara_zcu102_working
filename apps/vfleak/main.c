// vfleak -- minimal Verilator reproducer for the FPGA ".vf leak" wedge.
//
// FPGA observation (Cheshire + Ara, NrLanes=2, VLEN=2048, ILA-confirmed twice,
// 2026-07-28): every executed vfmul.vf permanently consumes one slot of a
// depth-4 structure (MfpuInsnQueueDepth / the sequencer's per-VFU counter /
// the hazard table -- not yet discriminated). All .vv forms free their slots.
// Concretely: 4 x vfmul.vf complete, then in iteration 5 the vle64 is accepted
// but never generates an address/AXI read (no load_complete), and CVA6 blocks
// forever with acc_req_valid=0. Interleaved .vv ops do NOT drain the leak.
//
// The passing sims that seemed to contradict this ran an 8-LANE build; this
// test must run with config=2_lanes to match the FPGA.
//
// Behavior here if the bug reproduces in simulation:
//   prints "vf 1 ok" ... "vf 4 ok", then hangs before "vf 5 ok" (sim timeout).
// If all 12 lines + PASS print, the 2-lane RTL is clean in the TB environment
// and the fault needs the Cheshire integration / memory latency to manifest.

#include <stdio.h>

#define N 32 /* exactly one m1 vector at e64 with VLEN=2048: vl = 32 */

static volatile double A[N], B[N], C[N];

/* one vfmul.vf per call: scalar operand crosses CVXIF (the leaking form) */
static long mul_vf(volatile double *a, volatile double *b,
                   volatile double *c, long n) {
  long d = 0;
  __asm__ volatile(
    "1: vsetvli t0,%4,e64,m1,ta,ma; vle64.v v16,(%2); fld ft0,0(%1);"
    "   vfmul.vf v0,v16,ft0; vse64.v v0,(%3);"
    "   slli t1,t0,3; add %2,%2,t1; add %3,%3,t1; add %0,%0,t0;"
    "   sub %4,%4,t0; bnez %4,1b"
    : "+r"(d), "+r"(a), "+r"(b), "+r"(c), "+r"(n)
    :: "t0","t1","ft0","v0","v16","memory");
  return d;
}

/* one vfmul.vv per call: control -- this form frees its slot on the FPGA */
static long mul_vv(volatile double *x, volatile double *b,
                   volatile double *c, long n) {
  long d = 0;
  __asm__ volatile(
    "  vsetvli t0,%4,e64,m1,ta,ma; vle64.v v8,(%1)\n"
    "1: vsetvli t0,%4,e64,m1,ta,ma; vle64.v v16,(%2);"
    "   vfmul.vv v0,v8,v16; vse64.v v0,(%3);"
    "   slli t1,t0,3; add %2,%2,t1; add %3,%3,t1; add %0,%0,t0;"
    "   sub %4,%4,t0; bnez %4,1b"
    : "+r"(d), "+r"(x), "+r"(b), "+r"(c), "+r"(n)
    :: "t0","t1","v0","v8","v16","memory");
  return d;
}

int main(void) {
  for (int i = 0; i < N; i++) { A[i] = 1.5; B[i] = 2.25; C[i] = 0.0; }
  printf("init ok\n");

  /* consecutive .vf: FPGA dies inside #5 */
  for (int k = 1; k <= 8; k++) {
    mul_vf(A, B, C, N);
    printf("vf %d ok\n", k);
  }
  printf("consecutive done\n");

  /* interleaved .vf/.vv: FPGA also dies at the 5th cumulative .vf */
  for (int k = 1; k <= 4; k++) {
    mul_vf(A, B, C, N);
    mul_vv(A, B, C, N);
    printf("interleaved %d ok\n", k);
  }

  /* light sanity so results cannot be optimized away */
  double s = 0;
  for (int i = 0; i < N; i++) s += C[i];
  printf("PASS checksum %d\n", (int)s); /* 32 * 1.5*2.25 = 108 */
  return 0;
}
