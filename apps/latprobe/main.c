// latprobe -- one homogeneous access pattern in a long steady loop, so an ILA
// capture on the memory-hierarchy probes is unambiguous.
//
// WHY A SEPARATE APP
// ------------------
// memmatrix issues thousands of MIXED accesses -- hit and miss, unit and
// strided, four sizes, three element widths. One ILA window catches them
// interleaved and no amount of post-processing separates them reliably. Each
// case here does exactly ONE kind of access, for long enough that the trigger
// cannot land anywhere else.
//
// WHY 16 BYTES
// ------------
// 16 B is one LLC line at BOTH geometries (128 B at 4 lanes, 64 B at 2). One
// line means exactly one AR and one R_LAST at every probe point, so request
// and response match by position and no ar.id bits are needed -- which is what
// kept the probe budget at 29 nets. A 2 KiB access would issue 16 refill ARs
// at L3 and the trace would need ID matching.
//
// WHY THE BARRIER IS STILL HERE
// -----------------------------
// `sd zero,(scratch); fence; csrr` stalls at issue until Ara has no pending
// load or store (acc_dispatcher.sv:128). That guarantees ONE outstanding
// transaction, which is what makes the trace readable. It also makes the
// software number printed below directly comparable to memmatrix.
//
//   LAT_CASE  pattern                    trigger the ILA on
//   --------  -------------------------  ---------------------------
//      1      vector 16 B, LLC hit       dbg_a1_arv && dbg_a1_arr
//      2      vector 16 B, LLC miss      dbg_a1_arv && dbg_a1_arr   (L3 = DDR)
//      3      scalar  8 B, L1 miss ->    dbg_c1_arv && dbg_c1_arr
//                          LLC hit
//      4      scalar  8 B, L1-D hit      dbg_d1_req && dbg_d1_gnt
//                                        ** no AXI traffic at all **
//
// Build: -DFPGA=1 -DNO_BOOT_LLC_FLUSH=1 -DLAT_CASE=<n>

#include <stdint.h>
#include "printf.h"

#ifndef LAT_CASE
#define LAT_CASE 1
#endif

// Long enough to arm and trigger the ILA by hand: at ~60 cycles per iteration
// and 50 MHz this is a window of roughly two seconds.
#ifndef LAT_ITERS
#define LAT_ITERS 100000000UL
#endif

#define L1_BUF    0x44000000UL     // 4 KiB, fits the 8 KiB L1 D$
#define SCRATCH   0x44008000UL     // barrier store target
#define HOT_BUF   0x45000000UL     // LLC-resident working set
#define HOT_WIN   (128UL * 1024UL) // 1/8 of a 1 MiB LLC -- cannot self-evict
#define COLD_BASE 0x48000000UL
#define COLD_SIZE (128UL * 1024UL * 1024UL)   // 128x the LLC: a wrap still misses

#define LLC_BASE       0x03001000UL
#define LLC_SET_ASSO   (LLC_BASE + 0x28)
#define LLC_NUM_LINES  (LLC_BASE + 0x30)
#define LLC_NUM_BLOCKS (LLC_BASE + 0x38)
#ifndef NR_LANES
#define NR_LANES 4
#endif
#define BLOCK_BYTES ((NR_LANES * 32 > 64 ? NR_LANES * 32 : 64) / 8)

static uint64_t sink __attribute__((unused));
static size_t   vl_now;

static inline int64_t bar(void) {
  int64_t c;
  __asm__ volatile("sd zero, 0(%1)\n\t fence\n\t csrr %0, cycle"
                   : "=r"(c) : "r"(SCRATCH) : "memory");
  return c;
}

// Install a range in the LLC using ARA. Ara has no L1, so this leaves CVA6's
// L1 untouched -- the only way to set up "L1 misses, LLC hits" (case 3).
__attribute__((unused)) static void ara_touch(uintptr_t base, uint64_t bytes) {
  __asm__ volatile("vsetvli %0, %1, e64, m8, ta, ma"
                   : "=r"(vl_now) : "r"((size_t)4096) : "memory");
  uint64_t span = (uint64_t)vl_now * 8UL;
  for (uint64_t o = 0; o < bytes; o += span)
    __asm__ volatile("vle64.v v8, (%0)" :: "r"(base + o) : "memory");
  __asm__ volatile("fence" ::: "memory");
}

__attribute__((unused)) static void prefill(uintptr_t base, uint64_t bytes) {
  volatile uint64_t *p = (volatile uint64_t *)base;
  for (uint64_t i = 0; i < bytes / 8; i++) p[i] = i;
  __asm__ volatile("fence" ::: "memory");
}

int main(void) {
  printf_("\n===== latprobe  LAT_CASE=%d  iters=%lu =====\n",
          LAT_CASE, (unsigned long)LAT_ITERS);

  uint32_t sa = *(volatile uint32_t *)LLC_SET_ASSO;
  uint32_t nl = *(volatile uint32_t *)LLC_NUM_LINES;
  uint32_t nb = *(volatile uint32_t *)LLC_NUM_BLOCKS;
  uint64_t line = (uint64_t)nb * BLOCK_BYTES;
  printf_("LLC %lu ways x %lu lines x %lu B = %lu KiB | NR_LANES=%d\n",
          (unsigned long)sa, (unsigned long)nl, (unsigned long)line,
          (unsigned long)((uint64_t)sa * nl * line / 1024), NR_LANES);

  // ---- calibrate the barrier, exactly as memmatrix does ------------------
  uint64_t acc = 0;
  for (int i = 0; i < 64; i++) { int64_t a = bar(); int64_t b = bar(); acc += (uint64_t)(b - a); }
  uint64_t bar_cost = acc / 64;
  printf_("barrier = %lu cycles (subtracted below)\n", (unsigned long)bar_cost);

  uintptr_t base, step, wrap;
  const char *what, *trig;

#if LAT_CASE == 1
  prefill(HOT_BUF, HOT_WIN);
  base = HOT_BUF; step = 4096; wrap = HOT_WIN;
  what = "vector 16 B, LLC HIT";   trig = "dbg_a1_arv && dbg_a1_arr";
  __asm__ volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl_now) : "r"((size_t)8) : "memory");
#elif LAT_CASE == 2
  base = COLD_BASE; step = 4096; wrap = COLD_SIZE;
  what = "vector 16 B, LLC MISS";  trig = "dbg_a1_arv && dbg_a1_arr  (L3 = DDR)";
  __asm__ volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl_now) : "r"((size_t)8) : "memory");
#elif LAT_CASE == 3
  // Step 4096 puts every address in the SAME L1 set (8 KiB / 4 ways / 32 B =
  // 64 sets, index bits [10:5]), so with 4 ways every access misses L1.
  // ara_touch then guarantees the line is in the LLC but not in L1.
  prefill(HOT_BUF, HOT_WIN);
  ara_touch(HOT_BUF, HOT_WIN);
  base = HOT_BUF; step = 4096; wrap = HOT_WIN;
  what = "scalar 8 B, L1 MISS -> LLC HIT"; trig = "dbg_c1_arv && dbg_c1_arr";
#else
  prefill(L1_BUF, 4096);
  base = L1_BUF; step = 8; wrap = 4096;
  what = "scalar 8 B, L1-D HIT";   trig = "dbg_d1_req && dbg_d1_gnt  (NO AXI)";
#endif

#if LAT_CASE <= 2
  printf_("vl=%lu -> %lu B per access\n", (unsigned long)vl_now,
          (unsigned long)(vl_now * 2));
#endif
  printf_("pattern : %s\n", what);
  printf_("ILA trig: %s\n", trig);
  printf_("running %lu iterations -- arm and trigger the ILA NOW\n",
          (unsigned long)LAT_ITERS);

  // ---- the steady loop ---------------------------------------------------
  acc = 0;
  uintptr_t off = 0;
  for (uint64_t i = 0; i < LAT_ITERS; i++) {
    uintptr_t a = base + off;
    off += step; if (off >= wrap) off = 0;
    int64_t t0 = bar();
#if LAT_CASE <= 2
    __asm__ volatile("vle16.v v8, (%0)" :: "r"(a) : "memory");
#else
    { uint64_t v_; __asm__ volatile("ld %0, 0(%1)" : "=r"(v_) : "r"(a) : "memory"); sink += v_; }
#endif
    int64_t t1 = bar();
    acc += (uint64_t)(t1 - t0);
  }

  uint64_t avg = acc / LAT_ITERS;
  printf_("\nsoftware mean : %lu cycles raw, %lu after subtracting the barrier\n",
          (unsigned long)avg,
          (unsigned long)(avg > bar_cost ? avg - bar_cost : 0));
  printf_("  compare this with the ILA interval for the same case.\n");
  printf_("  the DIFFERENCE is dispatch+retire, which no probe can see from\n");
  printf_("  software and no software number can separate on its own.\n");
  printf_("===== latprobe done =====\n");
  return 0;
}
