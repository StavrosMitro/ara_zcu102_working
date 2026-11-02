// sldprobe -- time ONE vector load and ONE vector slide of the same size, in a
// long steady loop, so an ILA capture on the PROBE_VEC sequencer points (and,
// for the load, the PROBE_MEM AXI points) is unambiguous.
//
//   SLD_CASE  instruction                         trigger the ILA on
//   --------  ----------------------------------  ------------------------
//      1      vle16.v        v8, (buf)            dbg_q_iss_ld == 1
//      2      vfslide1down.vf v8, v16, f          dbg_q_iss_sl == 1
//
// Both run at e16 (FP16 -- the fpu=16 bitstreams) and SLD_LMUL in {1,2,4,8},
// with vl = VLMAX, so a case moves exactly VLEN*LMUL/8 bytes and the load and
// the slide of the same LMUL move the SAME number of bytes. Build with
// -DSLD_SEW=32 for the fpu=32 bitstreams (vfslide1down.vf needs an FPU that
// supports the element width).
//
// THE ONE NON-OBVIOUS THING: HOW A SLIDE IS ISOLATED
// ---------------------------------------------------
// The barrier `sd zero,(scratch); fence; csrr` stalls at issue only while Ara
// has a pending LOAD or STORE (acc_dispatcher.sv). A slide is neither, so the
// barrier does NOT wait for it: back-to-back slides would overlap and the ILA
// would see throughput, not the latency of one instruction. So in case 2 every
// slide is followed by `vse16.v v8` -- a store that READS the slide's result.
// It cannot finish before the slide has, and the barrier does wait for it, so
// exactly one slide is in flight at a time. The ILA times the slide on its own
// (issue to SLDU -> SLDU done); the store is a separate, later instruction in
// the trace and does not contaminate that interval.
//
// What software can and cannot tell you here:
//   case 1: bar->vle->bar is the load's full latency (same as latprobe).
//   case 2: bar->slide->store->bar is slide PLUS a dependent store; with
//           chaining the two overlap, so "minus store-alone" is only a bound.
//           The exact slide time comes from the ILA, nowhere else.
//
// The loop length is scaled at runtime so every config runs ~SLD_TARGET_S
// seconds whatever VLEN/LMUL is -- that is the window for arming the ILA
// MID-RUN (arming before the reset release triggers on boot traffic and
// wastes the pre-trigger buffer; see memory ila-arm-midrun).
//
// Build: ENV_DEFINES="-DSLD_CASE=<1|2> -DSLD_LMUL=<1|2|4|8>"  (build_variants.sh)

#include <stdint.h>
#include "printf.h"

#ifndef SLD_CASE
#define SLD_CASE 1
#endif
#ifndef SLD_LMUL
#define SLD_LMUL 1
#endif
#ifndef SLD_SEW
#define SLD_SEW 16
#endif
#ifndef SLD_TARGET_S
#define SLD_TARGET_S 150            // seconds of steady loop = the ILA arming window
#endif
#define CLK_HZ 50000000UL

#define XSTR(x) STR(x)
#define STR(x)  #x
#define VTYPE   "e" XSTR(SLD_SEW) ", m" XSTR(SLD_LMUL) ", ta, ma"
#if SLD_SEW == 16
  typedef uint16_t elem_t;  typedef _Float16 scal_t;
  #define VLE  "vle16.v"
  #define VSE  "vse16.v"
#elif SLD_SEW == 32
  typedef uint32_t elem_t;  typedef float    scal_t;
  #define VLE  "vle32.v"
  #define VSE  "vse32.v"
#else
  #error "SLD_SEW must be 16 or 32"
#endif

// --- markers: flushed to DDR so a hang still shows how far it got ----------
#define LLC_BASE 0x03001000UL
#define A_STEP   0x50000100UL   // phase reached
#define A_RES    0x50000110UL   // results, see the end of main
#define SCRATCH  0x44008000UL   // barrier store target (same as latprobe)

// Big enough for VLEN 8192, LMUL 8, e16: 8192*8/16 = 4096 elements.
static elem_t ld_buf[8192] __attribute__((aligned(64)));
static elem_t st_buf[8192] __attribute__((aligned(64), unused));

static inline void llc_flush(void) {
  uint32_t ways = *(volatile uint32_t *)(LLC_BASE + 0x28);
  uint32_t mask = (ways >= 32) ? 0xFFFFFFFFu : ((1u << ways) - 1u);
  *(volatile uint32_t *)(LLC_BASE + 0x08) = mask;
  *(volatile uint32_t *)(LLC_BASE + 0x0c) = 0;
  *(volatile uint32_t *)(LLC_BASE + 0x10) = 1;
  while (*(volatile uint32_t *)(LLC_BASE + 0x08)) { }
}
static inline void mark(uintptr_t a, uint32_t v) {
  *(volatile uint32_t *)a = v; __asm__ volatile("fence" ::: "memory"); llc_flush();
}

static inline int64_t bar(void) {
  int64_t c;
  __asm__ volatile("sd zero, 0(%1)\n\t fence\n\t csrr %0, cycle"
                   : "=r"(c) : "r"(SCRATCH) : "memory");
  return c;
}

// The measured body. Kept in one asm per case so the compiler cannot move
// anything between the vector instructions.
static inline uint64_t one_iter(scal_t s) {
  (void)s;
  int64_t t0 = bar();
#if SLD_CASE == 1
  __asm__ volatile(VLE " v8, (%0)" :: "r"(ld_buf) : "memory");
#elif SLD_CASE == 2
  __asm__ volatile("vfslide1down.vf v8, v16, %1\n\t"
                   VSE " v8, (%0)"           // completion fence: reads v8
                   :: "r"(st_buf), "f"(s) : "memory");
#else
  #error "SLD_CASE must be 1 (load) or 2 (slide)"
#endif
  int64_t t1 = bar();
  return (uint64_t)(t1 - t0);
}

int main(void) {
  size_t vl, vlmax_e8;
  scal_t s = (scal_t)1.5f;
  mark(A_STEP, 1);

  __asm__ volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(vlmax_e8) :: "memory");
  __asm__ volatile("vsetvli %0, zero, " VTYPE : "=r"(vl) :: "memory");
  unsigned long vlen  = (unsigned long)vlmax_e8 * 8;
  unsigned long bytes = (unsigned long)vl * (SLD_SEW / 8);

  for (unsigned i = 0; i < sizeof(ld_buf) / sizeof(ld_buf[0]); i++) ld_buf[i] = (elem_t)(i * 3 + 1);
  __asm__ volatile("fence" ::: "memory");
  // warm the source: ld_buf LLC-resident, and v16 = the slide's input
  for (int w = 0; w < 8; w++) __asm__ volatile(VLE " v8, (%0)" :: "r"(ld_buf) : "memory");
  __asm__ volatile(VLE " v16, (%0)" :: "r"(ld_buf) : "memory");
  (void)bar();

  printf_("\n===== sldprobe  SLD_CASE=%d (%s)  e%d m%d =====\n", SLD_CASE,
          SLD_CASE == 1 ? "vle LOAD" : "vfslide1down SLIDE", SLD_SEW, SLD_LMUL);
  printf_("VLEN=%lu  vl=%lu  -> %lu bytes per instruction\n", vlen, (unsigned long)vl, bytes);

  // ---- barrier cost, exactly as memmatrix/latprobe ------------------------
  uint64_t acc = 0;
  for (int i = 0; i < 64; i++) { int64_t a = bar(); int64_t b = bar(); acc += (uint64_t)(b - a); }
  uint64_t bar_cost = acc / 64;

  // ---- store-alone reference (case 2 only) ---------------------------------
  uint64_t st_alone = 0;
#if SLD_CASE == 2
  acc = 0;
  for (int i = 0; i < 64; i++) {
    int64_t a = bar();
    __asm__ volatile(VSE " v8, (%0)" :: "r"(st_buf) : "memory");
    int64_t b = bar();
    acc += (uint64_t)(b - a);
  }
  st_alone = acc / 64;
#endif

  // ---- size the loop so it lasts ~SLD_TARGET_S seconds ----------------------
  uint64_t c0 = (uint64_t)bar();
  for (int i = 0; i < 1024; i++) (void)one_iter(s);
  uint64_t per_iter = ((uint64_t)bar() - c0) / 1024;
  if (per_iter == 0) per_iter = 1;
  uint64_t iters = (uint64_t)SLD_TARGET_S * CLK_HZ / per_iter;
  if (iters < 100000UL)    iters = 100000UL;
  if (iters > 400000000UL) iters = 400000000UL;
  mark(A_STEP, 2);

  printf_("barrier = %lu cycles%s\n", (unsigned long)bar_cost,
          SLD_CASE == 2 ? "  |  store-alone (vse of the same size) = see below" : "");
  printf_("loop: %lu iterations x ~%lu cycles = ~%lu s\n", (unsigned long)iters,
          (unsigned long)per_iter, (unsigned long)(iters * per_iter / CLK_HZ));
  printf_("ILA trig: %s == 1   -- arm it MID-RUN, ~20 s after release\n",
          SLD_CASE == 1 ? "dbg_q_iss_ld" : "dbg_q_iss_sl");
  mark(A_STEP, 3);

  // ---- the steady loop ------------------------------------------------------
  acc = 0;
  for (uint64_t i = 0; i < iters; i++) acc += one_iter(s);
  uint64_t avg = acc / iters;
  mark(A_STEP, 4);

  printf_("\nsoftware mean : %lu cycles raw, %lu after the barrier\n",
          (unsigned long)avg, (unsigned long)(avg > bar_cost ? avg - bar_cost : 0));
#if SLD_CASE == 1
  printf_("  = the load's full latency (CVA6 dispatch .. data in VRF .. retired)\n");
#else
  uint64_t st_net = st_alone > bar_cost ? st_alone - bar_cost : 0;
  printf_("  = slide + dependent store. store alone = %lu after the barrier\n", (unsigned long)st_net);
  printf_("  slide ~ %lu cycles -- a BOUND only (the store chains on the slide).\n",
          (unsigned long)((avg > bar_cost + st_net) ? avg - bar_cost - st_net : 0));
  printf_("  The exact slide time is the ILA interval dbg_q_iss_sl -> dbg_q_done_sl.\n");
#endif

  // machine-readable copy for devmem, in case the console is lost
  mark(A_RES + 0x00, (uint32_t)SLD_CASE | ((uint32_t)SLD_LMUL << 8) | ((uint32_t)SLD_SEW << 16));
  mark(A_RES + 0x04, (uint32_t)vlen);
  mark(A_RES + 0x08, (uint32_t)vl);
  mark(A_RES + 0x0c, (uint32_t)avg);
  mark(A_RES + 0x10, (uint32_t)bar_cost);
  mark(A_RES + 0x14, (uint32_t)st_alone);
  mark(A_STEP, 99);
  printf_("===== sldprobe done =====\n");
  return 0;
}
