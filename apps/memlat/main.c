// memlat -- memory latency seen by Ara's vector register file.
//
// WHAT IT MEASURES
//   LLC  -> VRF   a 128-bit vector load whose line is resident in the LLC
//   DDR  -> VRF   the same load when the line must be refilled from DRAM
//   DDR  -> LLC   derived as (DDR->VRF) - (LLC->VRF).  This one is a
//                 SUBTRACTION, not a measurement: the refill and the return
//                 to Ara overlap in part, so read it as a lower bound.
//
// WHY THE FENCE MATTERS
//   Ara is a decoupled accelerator: CVA6 dispatches a vector instruction and
//   carries on.  Reading `cycle` straight after `vle64.v` therefore measures
//   the DISPATCH, a handful of cycles, and tells you nothing about memory.
//   common/runtime.h's get_cycle_count() issues `fence` before `csrr`, which
//   waits for Ara to go idle.  Every number below depends on that.
//
// The cost of the fence pair is itself measured and reported, and must be
// subtracted before the results mean anything.
//
// Written for the Cheshire+Ara ZCU102 investigation, 2026-09-09.
// See AGENT_NOTES_FP16_4LANE.md sections 52-55.

#include <stdint.h>
#include <string.h>

#ifdef SPIKE
#include "util.h"
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#include "runtime.h"

// 128 bits per access, whatever the lane count: 2 elements of e64.
#define ACCESS_BITS 128
#define ELEMS       (ACCESS_BITS / 64)

// Stride between samples. The LLC line is 64 B at 2 lanes and 128 B at 4
// (line = LlcNumBlocks * AxiDataWidth/8, see notes 52.1), so 256 B guarantees
// a distinct line on both, with a gap so no access can share a line with the
// previous one.
#define STRIDE_B 256

// Cold buffer must comfortably exceed the 1 MiB LLC so that a walk misses.
#ifdef SPIKE
#define COLD_BYTES (256 * 1024)
#else
#define COLD_BYTES (8 * 1024 * 1024)
#endif
// Warm buffer must sit well inside it so a re-walk always hits.
#define WARM_BYTES (32 * 1024)

#define SAMPLES 128

// The cold buffer is addressed directly rather than declared in .bss.
// `objcopy -O binary` materialises .bss into the flat image, so an 8 MiB array
// would add 8 MiB to a binary that has to be shipped over the DDR mailbox.
//
// Memory map it has to stay clear of (common/link.fpga.ld and the console
// protocol): the image is linked at 0x4000_0000 and is a few hundred KiB; the
// harness owns 0x5000_0004 (console length), 0x5000_0050 (trap flag),
// 0x5000_2000 (console buffer) and 0x5FFF_F010 (sentinel). 0x4400_0000 is
// 64 MiB into the region, far above the image and far below the mailbox.
//
// On Spike none of this applies: riscv-tests' test.ld links at 0x8000_0000 and
// 0x4400_0000 is unmapped, so there the buffer is a small .bss array. Spike
// models no caches, so its size is irrelevant there anyway.
#ifdef SPIKE
static uint64_t cold_storage[COLD_BYTES / 8] __attribute__((aligned(256)));
static volatile uint64_t *const cold_buf = cold_storage;
#else
#define COLD_BASE 0x44000000UL
static volatile uint64_t *const cold_buf = (volatile uint64_t *)COLD_BASE;
#endif

static uint64_t warm_buf[WARM_BYTES / 8] __attribute__((aligned(256)));
static uint64_t sink_buf[STRIDE_B * 4 / 8] __attribute__((aligned(256)));

static int64_t samp[SAMPLES];

// ---------------------------------------------------------------- helpers

static void sort_samples(int64_t *a, int n) {
  for (int i = 1; i < n; ++i) {
    int64_t v = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; --j; }
    a[j + 1] = v;
  }
}

static void report(const char *what, int64_t *a, int n, int64_t overhead) {
  sort_samples(a, n);
  // min is the right statistic for a latency: it is the sample that was not
  // disturbed. median is reported next to it so interference is visible.
  int64_t mn = a[0] - overhead;
  int64_t md = a[n / 2] - overhead;
  int64_t mx = a[n - 1] - overhead;
  printf("  %-22s min %4ld   median %4ld   max %5ld\n", what,
         (long)mn, (long)md, (long)mx);
}

// Each timed region contains exactly one thing. No branches inside, so the
// measurement is not polluted by control flow.
static inline int64_t t_nothing(void) {
  start_timer();
  stop_timer();
  return get_timer();
}

static inline int64_t t_load(const void *src) {
  start_timer();
  asm volatile("vle64.v v0, (%0)" ::"r"(src) : "memory");
  stop_timer();
  return get_timer();
}

static inline int64_t t_load_store(const void *src, void *dst) {
  start_timer();
  asm volatile("vle64.v v0, (%0)" ::"r"(src) : "memory");
  asm volatile("vse64.v v0, (%0)" ::"r"(dst) : "memory");
  stop_timer();
  return get_timer();
}

static inline int64_t t_store(void *dst) {
  start_timer();
  asm volatile("vse64.v v0, (%0)" ::"r"(dst) : "memory");
  stop_timer();
  return get_timer();
}

// ---------------------------------------------------------------- main

int main(void) {
  unsigned long vl;

  ENABLE_VEC;
  HW_CNT_READY;

  asm volatile("vsetvli %0, %1, e64, m1, ta, ma" : "=r"(vl) : "r"(ELEMS));

  printf("\n=== memlat: memory latency to the vector register file ===\n");
  printf("  access   : %d bits  (vl=%lu, e64)\n", ACCESS_BITS, vl);
  printf("  stride   : %d B\n", STRIDE_B);
  printf("  cold buf : %d KiB   warm buf : %d KiB\n",
         COLD_BYTES / 1024, WARM_BYTES / 1024);
  printf("  samples  : %d\n\n", SAMPLES);

  // Touch everything once so the pages and the .bss are materialised, and so
  // the cold buffer's first-ever access is not being timed.
  for (int i = 0; i < COLD_BYTES / 8; i += STRIDE_B / 8) cold_buf[i] = i;
  for (int i = 0; i < WARM_BYTES / 8; i += STRIDE_B / 8) warm_buf[i] = i;
  for (unsigned i = 0; i < sizeof(sink_buf) / 8; ++i) sink_buf[i] = 0;

  // ---- 1. the cost of the fence pair itself -------------------------------
  for (int i = 0; i < SAMPLES; ++i) samp[i] = t_nothing();
  sort_samples(samp, SAMPLES);
  int64_t overhead = samp[0];
  printf("  fence pair overhead (subtracted below): min %ld  median %ld\n\n",
         (long)samp[0], (long)samp[SAMPLES / 2]);

  // ---- 2. LLC hit ---------------------------------------------------------
  // Walk the small buffer twice: the first walk pulls it into the LLC, the
  // second is what we time.
  for (int r = 0; r < 2; ++r)
    for (int i = 0; i < WARM_BYTES / STRIDE_B; ++i)
      (void)t_load(&warm_buf[i * (STRIDE_B / 8)]);

  for (int i = 0; i < SAMPLES; ++i)
    samp[i] = t_load(&warm_buf[(i % (WARM_BYTES / STRIDE_B)) * (STRIDE_B / 8)]);
  report("LLC hit  -> VRF", samp, SAMPLES, overhead);
  int64_t llc_min = samp[0] - overhead;

  // ---- 3. DDR, sequential lines ------------------------------------------
  // Every sample lands on a line the walk has not touched since the last
  // eviction. 8 MiB against a 1 MiB LLC, so nothing survives a pass.
  {
    int stride_words = STRIDE_B / 8;
    int lines = COLD_BYTES / STRIDE_B;
    for (int i = 0; i < SAMPLES; ++i) {
      int line = (i * (lines / SAMPLES)) % lines;
      samp[i] = t_load(&cold_buf[line * stride_words]);
    }
  }
  report("DDR seq  -> VRF", samp, SAMPLES, overhead);
  int64_t ddr_seq_min = samp[0] - overhead;

  // ---- 4. DDR, scattered lines -------------------------------------------
  // A large odd stride so consecutive samples land in different DRAM rows.
  // Sequential access benefits from open pages; this is the unfriendly case.
  {
    int stride_words = STRIDE_B / 8;
    int lines = COLD_BYTES / STRIDE_B;
    int step = 1021;  // prime, larger than any row's worth of lines
    int line = 0;
    for (int i = 0; i < SAMPLES; ++i) {
      line = (line + step) % lines;
      samp[i] = t_load(&cold_buf[line * stride_words]);
    }
  }
  report("DDR scatter -> VRF", samp, SAMPLES, overhead);
  int64_t ddr_sct_min = samp[0] - overhead;

  // ---- 5. what the store costs on top ------------------------------------
  for (int i = 0; i < SAMPLES; ++i)
    samp[i] = t_load_store(&warm_buf[(i % (WARM_BYTES / STRIDE_B)) *
                                     (STRIDE_B / 8)],
                           &sink_buf[0]);
  report("LLC load+store", samp, SAMPLES, overhead);

  for (int i = 0; i < SAMPLES; ++i) samp[i] = t_store(&sink_buf[0]);
  report("store alone", samp, SAMPLES, overhead);

  // ---- summary ------------------------------------------------------------
  printf("\n  derived, lower bound only:\n");
  printf("    DDR->LLC refill (seq)     = %ld - %ld = %ld cycles\n",
         (long)ddr_seq_min, (long)llc_min, (long)(ddr_seq_min - llc_min));
  printf("    DDR->LLC refill (scatter) = %ld - %ld = %ld cycles\n",
         (long)ddr_sct_min, (long)llc_min, (long)(ddr_sct_min - llc_min));
  printf("\n  NB the refill figure is a subtraction. The refill and the return\n");
  printf("  to Ara overlap, so the true DDR->LLC time is at least this.\n\n");

  HW_CNT_NOT_READY;
  return 0;
}
