// memmatrix -- raw memory-latency matrix for the Cheshire+CVA6+Ara ZCU102.
//
// ============================================================================
//  THE INSTRUMENT
// ============================================================================
// Everything below rests on four facts that were read out of the RTL, not
// assumed. If any of them stops being true, the numbers stop meaning anything.
//
//  (1) A plain `fence` does NOT wait for vector LOADS.
//        acc_dispatcher.sv:337
//          wait_acc_store_d = (wait_acc_store_q | commit_st_barrier_i)
//                             & acc_resp_i.acc_resp.store_pending;
//          ctrl_halt_o      = wait_acc_store_q;
//      Only `store_pending`. Timing a `vle64.v` with `fence; csrr` measures
//      the DISPATCH -- a handful of cycles -- and reports it as memory latency.
//
//  (2) A scalar STORE does wait, for loads AND stores.
//        acc_dispatcher.sv:128
//          STORE: issue_stall_o = acc_cons_en_i
//                                 & (~acc_no_st_pending | ~acc_no_ld_pending);
//        csr_regfile.sv:2571   acc_cons_q <= EnableAccelerator   (= 1 at reset)
//      So `sd` stalls at ISSUE until Ara has no pending load and no pending
//      store. That makes ONE barrier valid for every case here, which is why
//      a single calibration constant covers the whole matrix.
//
//  (3) `fence` does NOT flush the L1 D-cache.
//        cv64a6_imafdcv_sv39_config_pkg.sv:59
//          localparam CVA6ConfigDcacheFlushOnFence = 1'b0;
//      Without this, an L1-D hit could never be measured: every barrier would
//      cold-start the cache.
//
//  (4) `fence` ALWAYS flushes the pipeline and restarts instruction fetch.
//        controller.sv:118  flush_if_o = flush_id_o = flush_ex_o = 1'b1;
//      So the I$ (4 KiB) sits in the critical path of every single reading. A
//      cold I$ refetches through the very LLC we are measuring. Hence WARM
//      iterations that are timed and thrown away, and an I$-miss counter to
//      PROVE the warm-up worked rather than hoping it did.
//
// The barrier is therefore:      sd zero,(scratch) ; fence ; csrr cycle
// and `bar_cost` measures it empty so it can be subtracted.
//
// NO POINTER CHASE IS NEEDED. A chase exists to create a dependency when you
// cannot stop the machine; here the barrier stops it. Dropping the chase also
// drops its trap: building the chain writes the buffer, which warms the very
// lines the cold test needs cold.
//
// ============================================================================
//  GEOMETRY (verified)
// ============================================================================
//   L1 I$    4 KiB
//   L1 D$    8 KiB, 4-way, 32 B line, write-through   (Ara has NO L1)
//   LLC      1 MiB, 8-way, 128 B line, 1024 lines     (LlcNumBlocks 8 x 16 B)
//   MMU      OFF -- crt0.S never writes satp, bare M-mode, so NO TLB misses
//
//   Same-set stride = NumLines * LineSize = 1024 * 128 = 128 KiB.
//   With 8 ways, the 9th access at that stride evicts. That is test S2.
//
//   A 128 B line means STRIDE 32 IS NOT A MISS GENERATOR: 4 elements land in
//   one line, so you get 1 miss + 3 hits. Stride 32 measures the VLSU's
//   address-generation cost; stride 128 measures misses. Both are here, and
//   they must not be read as the same quantity.
//
// Build: -DFPGA=1 -DNO_BOOT_LLC_FLUSH=1 (FPGA memory map is hardcoded below).

#include <stdint.h>
#include "printf.h"

// ---- FPGA memory map (arch.link.fpga.ld: LLC @ 0x4000_0000, len 0x2000_0000,
//      _stack_top = 0x6000_0000, eoc_address_reg = 0x5FFF_F010) --------------
#define L1_BUF     0x44000000UL   //   4 KiB, fits L1 D$ (8 KiB)
#define SCRATCH    0x44008000UL   //   one line, target of the barrier store
#define HOT_BUF    0x45000000UL   // half the LLC -> cannot self-evict (sized at runtime)
#define DIRTY_BUF  0x46000000UL   // 3/4 of the LLC, to make it fully dirty (runtime)

// ---- LLC geometry is READ FROM THE HARDWARE, never assumed ----------------
// There are bitstreams with a 128 KiB LLC and bitstreams with 1 MiB. Hardcoding
// either one silently destroys the matrix on the other: a "hot" buffer larger
// than the cache turns every hit row into a miss row, and the same-set stride
// (NumLines * LineSize) is 16 KiB on one and 128 KiB on the other. The config
// registers are 32-bit LOW/HIGH pairs (axi_llc_reg_pkg.sv:177-182), unlike the
// DMA's 64-bit map, so these are lw reads.
#define LLC_BASE       0x03001000UL
#define LLC_SET_ASSO   (LLC_BASE + 0x28)
#define LLC_NUM_LINES  (LLC_BASE + 0x30)
#define LLC_NUM_BLOCKS (LLC_BASE + 0x38)

#ifndef NR_LANES
#define NR_LANES 4
#endif
// AxiDataWidth = max(32*NrLanes, 64) bits -> bytes per block
#define BLOCK_BYTES ((NR_LANES * 32 > 64 ? NR_LANES * 32 : 64) / 8)

static uint64_t llc_bytes, llc_line, hot_size, dirty_size, conflict_stride;
static uint64_t hot_win;   // window a hit test walks; must stay well inside the LLC
// COLD_END stops at 0x5000_0000 ON PURPOSE: common/serial.c puts the DDR
// console control words at 0x5000_0004..0x5000_0094 and its 16 MB text
// buffer at 0x5000_2000. Running the cold walk into that would overwrite the
// only channel the results come back on. CONF_BUF sits above it, at
// 0x5400_0000 > 0x5100_2000 (end of the console buffer).
#define COLD_BASE  0x48000000UL   // 128 MiB, never revisited -> always cold
#define COLD_END   0x50000000UL
#define CONF_BUF   0x54000000UL   //   4 MiB, for the same-set stride test

#define REPS 16                   // timed and accumulated
#define WARM 8                    // timed and DISCARDED (warms I$ and Ara)

// ---- CSRs ------------------------------------------------------------------
#define RD_CSR(n) ({ uint64_t v_; __asm__ volatile("csrr %0, " #n : "=r"(v_)); v_; })
#define WR_CSR(n, v) __asm__ volatile("csrw " #n ", %0" :: "r"((uint64_t)(v)))
// mhpmevent3=0x323 mhpmevent4=0x324 mhpmevent5=0x325
// mhpmcounter3=0xB03 mhpmcounter4=0xB04 mhpmcounter5=0xB05
// events: 0x01 I$ miss, 0x02 D$ miss, 0x11 D$ access

static uint64_t sink;

static uint64_t bar_cost;
static uintptr_t cold_cursor = COLD_BASE;
static int overrun = 0;

// The universal barrier. See fact (2): the scalar store stalls at issue until
// Ara has neither a load nor a store pending, so this is valid for vector
// loads, vector stores and scalar accesses alike.
static inline int64_t bar(void) {
  int64_t c;
  __asm__ volatile("sd zero, 0(%1)\n\t"
                   "fence\n\t"
                   "csrr %0, cycle"
                   : "=r"(c) : "r"(SCRATCH) : "memory");
  return c;
}

// Hand out a fresh cold slice. Never returns an address twice, so every
// access into it is guaranteed to miss the LLC -- there is no flush available
// (a mid-run LLC flush hard-stalls the core), so address choice is the ONLY
// way to establish cold state.
static uintptr_t cold_take(uint64_t bytes) {
  uintptr_t a = cold_cursor;
  cold_cursor += (bytes + 4095UL) & ~4095UL;
  if (cold_cursor >= COLD_END) { overrun = 1; cold_cursor = COLD_BASE; }
  return a;
}

// ---- vsetvli: LMUL is an immediate, so one macro per setting ---------------
// VLEN=2048b -> e64 vlmax: m1=32 (256 B), m2=64 (512 B), m4=128 (1 KiB),
// m8=256 (2 KiB = 16 LLC lines, i.e. a real burst).
static size_t vl_now;
#define VSET(MUL)                                                              \
  __asm__ volatile("vsetvli %0, %1, e64, " MUL ", ta, ma"                      \
                   : "=r"(vl_now) : "r"((size_t)4096) : "memory")

// SEW and LMUL are both immediates, so every combination needs its own
// instruction. sew index: 0 = e16, 1 = e32, 2 = e64.
#define VSETI(SEW, MUL, N)                                                     \
  __asm__ volatile("vsetvli %0, %1, " SEW ", " MUL ", ta, ma"                  \
                   : "=r"(vl_now) : "r"((size_t)(N)) : "memory")
#define VSET_AVL(MUL, N) VSETI("e64", MUL, N)

static const char *SEW_NAME[3] = {"e16", "e32", "e64"};
static const int   SEW_BITS[3] = {16, 32, 64};

static void vset(int sew, int lmul_idx, size_t avl) {
  switch (sew * 4 + lmul_idx) {
    case  0: VSETI("e16", "m1", avl); break;
    case  1: VSETI("e16", "m2", avl); break;
    case  2: VSETI("e16", "m4", avl); break;
    case  3: VSETI("e16", "m8", avl); break;
    case  4: VSETI("e32", "m1", avl); break;
    case  5: VSETI("e32", "m2", avl); break;
    case  6: VSETI("e32", "m4", avl); break;
    case  7: VSETI("e32", "m8", avl); break;
    case  8: VSETI("e64", "m1", avl); break;
    case  9: VSETI("e64", "m2", avl); break;
    case 10: VSETI("e64", "m4", avl); break;
    default: VSETI("e64", "m8", avl); break;
  }
}

// Install a range in the LLC using ARA, not CVA6.
//
// Ara has no L1 (vlsu.sv drives the only AXI port), so this fills the LLC and
// leaves CVA6's L1 completely untouched. That is the exact state the scalar
// "L1 miss -> LLC hit" row claims to measure, and there is no way to reach it
// by pre-touching with scalar loads -- those install into L1 as well.
static void ara_touch(uintptr_t base, uint64_t bytes) {
  VSET_AVL("m8", 4096);
  uint64_t span = (uint64_t)vl_now * 8UL;
  for (uint64_t o = 0; o < bytes; o += span)
    __asm__ volatile("vle64.v v8, (%0)" :: "r"(base + o) : "memory");
  __asm__ volatile("fence" ::: "memory");
}

// Read a range without writing it, so its LLC lines are CLEAN.
static void read_fill(uintptr_t base, uint64_t bytes) {
  volatile uint64_t *p = (volatile uint64_t *)base;
  uint64_t s_ = 0;
  for (uint64_t i = 0; i < bytes / 8; i++) s_ += p[i];
  sink += s_;
  __asm__ volatile("fence" ::: "memory");
}

// ---- the measurement bodies ------------------------------------------------
// `a` is in scope inside BODY. WARM iterations are timed and discarded so the
// I$ is hot and Ara's queues are in steady state before anything is kept.
// hit=1 : install the exact address set first, then measure it -> LLC hit.
// hit=0 : the first WARM addresses are consumed as warm-up and never measured,
//         so every measured address is still cold -> LLC miss.
#define GEN_TEST(FN, BODY)                                                     \
  static uint64_t FN(uintptr_t base, uintptr_t step, uintptr_t wrap, int hit) {\
    uint64_t acc = 0;                                                          \
    if (hit) {                                                                 \
      _Pragma("clang loop unroll(disable)")                                    \
      for (int i = 0; i < WARM + REPS; i++) {                                  \
        uintptr_t off = (uintptr_t)i * step;                                   \
        uintptr_t a = base + (wrap ? (off % wrap) : off);                      \
        BODY;                                                                  \
      }                                                                        \
    }                                                                          \
    _Pragma("clang loop unroll(disable)")                                      \
    for (int i = 0; i < WARM + REPS; i++) {                                    \
      uintptr_t off = (uintptr_t)i * step;                                     \
      uintptr_t a = base + (wrap ? (off % wrap) : off);                        \
      int64_t t0 = bar();                                                      \
      BODY;                                                                    \
      int64_t t1 = bar();                                                      \
      if (i >= WARM) acc += (uint64_t)(t1 - t0);                               \
    }                                                                          \
    uint64_t avg = acc / REPS;                                                 \
    return avg > bar_cost ? avg - bar_cost : 0;                                \
  }

// Unit stride, one set per element width: the opcode encodes SEW, so these
// cannot be parameterised at run time.
GEN_TEST(t_vle16, __asm__ volatile("vle16.v v8, (%0)"  :: "r"(a) : "memory"))
GEN_TEST(t_vle32, __asm__ volatile("vle32.v v8, (%0)"  :: "r"(a) : "memory"))
GEN_TEST(t_vle64, __asm__ volatile("vle64.v v8, (%0)"  :: "r"(a) : "memory"))
GEN_TEST(t_vse16, __asm__ volatile("vse16.v v16, (%0)" :: "r"(a) : "memory"))
GEN_TEST(t_vse32, __asm__ volatile("vse32.v v16, (%0)" :: "r"(a) : "memory"))
GEN_TEST(t_vse64, __asm__ volatile("vse64.v v16, (%0)" :: "r"(a) : "memory"))

// Stride is a runtime operand, so the "exactly one LLC line per element" case
// follows the MEASURED line size: 128 B at 4 lanes, 64 B at 2. A fixed 128
// would mean something different on each bitstream.
#define GEN_TEST_S(FN, BODY)                                                   \
  static uint64_t FN(uintptr_t base, uintptr_t step, uintptr_t wrap,           \
                     long stride, int hit) {                                   \
    uint64_t acc = 0;                                                          \
    if (hit) {                                                                 \
      _Pragma("clang loop unroll(disable)")                                    \
      for (int i = 0; i < WARM + REPS; i++) {                                  \
        uintptr_t off = (uintptr_t)i * step;                                   \
        uintptr_t a = base + (wrap ? (off % wrap) : off);                      \
        BODY;                                                                  \
      }                                                                        \
    }                                                                          \
    _Pragma("clang loop unroll(disable)")                                      \
    for (int i = 0; i < WARM + REPS; i++) {                                    \
      uintptr_t off = (uintptr_t)i * step;                                     \
      uintptr_t a = base + (wrap ? (off % wrap) : off);                        \
      int64_t t0 = bar();                                                      \
      BODY;                                                                    \
      int64_t t1 = bar();                                                      \
      if (i >= WARM) acc += (uint64_t)(t1 - t0);                               \
    }                                                                          \
    uint64_t avg = acc / REPS;                                                 \
    return avg > bar_cost ? avg - bar_cost : 0;                                \
  }

GEN_TEST_S(t_vlse16, __asm__ volatile("vlse16.v v8, (%0), %1"  :: "r"(a), "r"(stride) : "memory"))
GEN_TEST_S(t_vlse32, __asm__ volatile("vlse32.v v8, (%0), %1"  :: "r"(a), "r"(stride) : "memory"))
GEN_TEST_S(t_vlse64, __asm__ volatile("vlse64.v v8, (%0), %1"  :: "r"(a), "r"(stride) : "memory"))
GEN_TEST_S(t_vsse16, __asm__ volatile("vsse16.v v16, (%0), %1" :: "r"(a), "r"(stride) : "memory"))
GEN_TEST_S(t_vsse32, __asm__ volatile("vsse32.v v16, (%0), %1" :: "r"(a), "r"(stride) : "memory"))
GEN_TEST_S(t_vsse64, __asm__ volatile("vsse64.v v16, (%0), %1" :: "r"(a), "r"(stride) : "memory"))

typedef uint64_t (*tfn_t )(uintptr_t, uintptr_t, uintptr_t, int);
typedef uint64_t (*tfns_t)(uintptr_t, uintptr_t, uintptr_t, long, int);
static tfn_t  const LE [3] = {t_vle16,  t_vle32,  t_vle64 };
static tfn_t  const SE [3] = {t_vse16,  t_vse32,  t_vse64 };
static tfns_t const LSE[3] = {t_vlse16, t_vlse32, t_vlse64};
static tfns_t const SSE[3] = {t_vsse16, t_vsse32, t_vsse64};

GEN_TEST(t_ld,   { uint64_t v_; __asm__ volatile("ld %0, 0(%1)" : "=r"(v_) : "r"(a) : "memory"); sink += v_; })
GEN_TEST(t_sd,   __asm__ volatile("sd %0, 0(%1)" :: "r"(sink), "r"(a) : "memory"))

// Empty body: this IS the calibration. Must run before bar_cost is used, so it
// subtracts nothing (bar_cost is still 0 at that point).
GEN_TEST(t_null, __asm__ volatile("" :: "r"(a) : "memory"))

// ---- reporting -------------------------------------------------------------
static void row(const char *name, uint64_t bytes, uint64_t hit, uint64_t miss) {
  // cycles per 100 bytes, integer, so no FP is needed anywhere in this app
  uint64_t hpb = bytes ? (hit * 100UL) / bytes : 0;
  uint64_t mpb = bytes ? (miss * 100UL) / bytes : 0;
  printf_("  %-22s %5lu B | hit %6lu  miss %6lu | delta %6lu | c/100B %3lu/%3lu\n",
          name, (unsigned long)bytes,
          (unsigned long)hit, (unsigned long)miss,
          (unsigned long)(miss > hit ? miss - hit : 0),
          (unsigned long)hpb, (unsigned long)mpb);
}

static void prefill(uintptr_t base, uint64_t bytes) {
  volatile uint64_t *p = (volatile uint64_t *)base;
  for (uint64_t i = 0; i < bytes / 8; i++) p[i] = i;
  __asm__ volatile("fence" ::: "memory");
}

int main(void) {
  printf_("\n================ memmatrix ================\n");

  // ---- read the LLC geometry out of the hardware --------------------------
  uint32_t sa = *(volatile uint32_t *)LLC_SET_ASSO;
  uint32_t nl = *(volatile uint32_t *)LLC_NUM_LINES;
  uint32_t nb = *(volatile uint32_t *)LLC_NUM_BLOCKS;
  llc_line        = (uint64_t)nb * BLOCK_BYTES;
  llc_bytes       = (uint64_t)sa * nl * llc_line;
  conflict_stride = (uint64_t)nl * llc_line;
  hot_size        = llc_bytes / 2;      // half -> a hot walk cannot self-evict
  dirty_size      = (llc_bytes * 3) / 4;

  printf_("LLC: %lu ways x %lu lines x %lu B = %lu KiB | same-set stride %lu KiB\n",
          (unsigned long)sa, (unsigned long)nl, (unsigned long)llc_line,
          (unsigned long)(llc_bytes / 1024), (unsigned long)(conflict_stride / 1024));
  printf_("L1D 8K/4w/32B (Ara has none) | MMU off | NR_LANES=%d -> %d B/block\n",
          NR_LANES, BLOCK_BYTES);
  printf_(">>> NR_LANES is COMPILE-TIME. On an LLC1M bitstream this MUST print\n");
  printf_(">>> 1024 KiB above. 512 = l2 binary on 4-lane HW; 2048 = l4 on 2-lane.\n");

  if (sa == 0 || nl == 0 || nb == 0 || llc_bytes > (16UL << 20)) {
    printf_("*** LLC config regs look wrong (%lu/%lu/%lu).\n",
            (unsigned long)sa, (unsigned long)nl, (unsigned long)nb);
    printf_("*** Either this is an LLC-bypass build or 0x03001000 is unmapped.\n");
    printf_("*** Every hit/miss row below would be meaningless. ABORTING.\n");
    return 90;
  }
  if (hot_size > (512UL << 10)) hot_size = 512UL << 10;   // buffer reservation
  if (dirty_size > (768UL << 10)) dirty_size = 768UL << 10;

  // A hit test walks hot_win and each access spans up to vl*llc_line on top.
  // One eighth of the cache leaves room for that span and for the I$ and the
  // barrier's own line, so a hit test can never evict its own working set.
  hot_win = llc_bytes / 8;
  printf_("hit window %lu KiB of %lu KiB LLC (a hit row must not self-evict)\n",
          (unsigned long)(hot_win / 1024), (unsigned long)(llc_bytes / 1024));

  // HPM: 3 <- L1 D$ miss, 4 <- L1 D$ access, 5 <- L1 I$ miss.
  // mcountinhibit resets to 0 (csr_regfile.sv:2570), so they already run.
  WR_CSR(0x323, 0x02);
  WR_CSR(0x324, 0x11);
  WR_CSR(0x325, 0x01);
  WR_CSR(0x320, 0);

  // ---- calibrate the barrier ----------------------------------------------
  bar_cost = 0;
  uint64_t ic0 = RD_CSR(0xB05);
  bar_cost = t_null(SCRATCH, 0, 0, 0);
  uint64_t ic1 = RD_CSR(0xB05);
  printf_("barrier (sd+fence+csrr) = %lu cycles, subtracted from every row\n",
          (unsigned long)bar_cost);
  printf_("I$ misses during calibration: %lu  (must be ~0, see fact 4)\n",
          (unsigned long)(ic1 - ic0));

  prefill(HOT_BUF, hot_size);
  prefill(L1_BUF, 4096);

  // ==== ONE LLC LINE, AT EVERY ELEMENT WIDTH ==============================
  // 16 B is one LLC line at both geometries, and at every SEW it is the SAME
  // 16 bytes -- only the element COUNT differs (8 at e16, 4 at e32, 2 at e64).
  // So this row answers one question directly: does the VLSU charge per BYTE
  // or per ELEMENT?  Equal numbers mean per byte.
  printf_("\n---- single access, 16 B = 1 LLC line, same bytes at every SEW ----\n");
  for (int sw = 0; sw < 3; sw++) {
    vset(sw, 0, (size_t)(128 / SEW_BITS[sw]));      // AVL chosen so vl*SEW = 16 B
    uint64_t hi = LE[sw](HOT_BUF, 4096, hot_win, 1);
    uint64_t mi = LE[sw](cold_take(4096UL * (WARM + REPS)), 4096, 0, 0);
    printf_("  %s vl=%-4lu  LOAD  hit %6lu   miss %6lu   delta %6lu\n",
            SEW_NAME[sw], (unsigned long)vl_now,
            (unsigned long)hi, (unsigned long)mi,
            (unsigned long)(mi > hi ? mi - hi : 0));
    if (mi <= hi)
      printf_("  *** miss <= hit on a single line: residency is wrong. INVALID.\n");
  }

  // ==== VECTOR SWEEP ======================================================
  // e16 and e32 in full: those are what the kernels actually run (FP16 data,
  // FP32 accumulators).  e64 only at LMUL=8, as a tie-back to the earlier
  // e64-only round.
  //
  // At a given LMUL every SEW moves the SAME number of bytes, so the unit
  // rows are directly comparable across widths.  The strided rows are not:
  // strided issues one address per ELEMENT, so e16 asks for 4x the addresses
  // of e64 for the same bytes.
  for (int sw = 0; sw < 3; sw++) {
    for (int m = 0; m < 4; m++) {
      if (sw == 2 && m != 3) continue;             // e64: LMUL=8 anchor only

      vset(sw, m, 4096);
      uint64_t B     = (uint64_t)vl_now * (SEW_BITS[sw] / 8);
      uint64_t stepC = (B + 4095UL) & ~4095UL;

      printf_("\n---- %s LMUL=%d : vl=%lu, %lu B/instr, %lu LLC lines ----\n",
              SEW_NAME[sw], 1 << m, (unsigned long)vl_now, (unsigned long)B,
              (unsigned long)((B + llc_line - 1) / llc_line));

      uint64_t hi, mi;
      hi = LE[sw](HOT_BUF, 4096, hot_win, 1);
      mi = LE[sw](cold_take(stepC * (WARM + REPS)), stepC, 0, 0);
      row("unit LOAD", B, hi, mi);

      hi = SE[sw](HOT_BUF, 4096, hot_win, 1);
      mi = SE[sw](cold_take(stepC * (WARM + REPS)), stepC, 0, 0);
      row("unit STORE", B, hi, mi);

      // stride 32: sub-line at both geometries, so this is addressing cost
      uint64_t S32   = (uint64_t)vl_now * 32UL;
      uint64_t step32 = (S32 + 4095UL) & ~4095UL;
      hi = LSE[sw](HOT_BUF, 4096, hot_win, 32, 1);
      mi = LSE[sw](cold_take(step32 * (WARM + REPS)), step32, 0, 32, 0);
      row("strided-32 LOAD", S32, hi, mi);

      hi = SSE[sw](HOT_BUF, 4096, hot_win, 32, 1);
      mi = SSE[sw](cold_take(step32 * (WARM + REPS)), step32, 0, 32, 0);
      row("strided-32 STORE", S32, hi, mi);

      // stride == LLC line: one line per element, zero spatial reuse
      uint64_t SL    = (uint64_t)vl_now * llc_line;
      uint64_t stepL = (SL + 4095UL) & ~4095UL;
      hi = LSE[sw](HOT_BUF, 4096, hot_win, (long)llc_line, 1);
      mi = LSE[sw](cold_take(stepL * (WARM + REPS)), stepL, 0, (long)llc_line, 0);
      row("strided-LINE LOAD", SL, hi, mi);

      hi = SSE[sw](HOT_BUF, 4096, hot_win, (long)llc_line, 1);
      mi = SSE[sw](cold_take(stepL * (WARM + REPS)), stepL, 0, (long)llc_line, 0);
      row("strided-LINE STORE", SL, hi, mi);
    }
  }

  printf_("\n---- scalar (8 B) ----\n");
  {
    uint64_t d0 = RD_CSR(0xB03), a0 = RD_CSR(0xB04);
    uint64_t l1 = t_ld(L1_BUF, 8, 4096, 1);       // 4 KiB, fits L1 D$
    uint64_t d1 = RD_CSR(0xB03), a1 = RD_CSR(0xB04);
    printf_("  L1-D hit             : %6lu cycles   (D$ miss %lu of %lu access)\n",
            (unsigned long)l1, (unsigned long)(d1 - d0), (unsigned long)(a1 - a0));

    // Pre-touch with ARA so the lines are in the LLC and NOT in CVA6's L1,
    // then measure with hit=0 so the scalar pass does not pre-touch and
    // pollute L1 itself. Step 4096 puts every address in the same L1 set
    // (8 KiB / 4 ways / 32 B = 64 sets, index bits [10:5]), which is why the
    // D$ miss count below should equal the access count.
    ara_touch(HOT_BUF, (uint64_t)(WARM + REPS) * 4096UL);
    d0 = RD_CSR(0xB03);
    uint64_t llc = t_ld(HOT_BUF, 4096, 0, 0);
    d1 = RD_CSR(0xB03);
    printf_("  L1 miss -> LLC hit   : %6lu cycles   (D$ miss %lu, expect ~%d)\n",
            (unsigned long)llc, (unsigned long)(d1 - d0), WARM + REPS);

    d0 = RD_CSR(0xB03);
    uint64_t ddr = t_ld(cold_take(4096UL * (WARM + REPS)), 4096, 0, 0);
    d1 = RD_CSR(0xB03);
    printf_("  L1 miss -> LLC miss  : %6lu cycles   (D$ miss %lu, expect ~%d)\n",
            (unsigned long)ddr, (unsigned long)(d1 - d0), WARM + REPS);
    if (ddr <= llc)
      printf_("  *** miss <= hit: the hit row was NOT resident. INVALID.\n");

    uint64_t st_h = t_sd(L1_BUF, 8, 4096, 1);
    uint64_t st_m = t_sd(cold_take(4096UL * (WARM + REPS)), 4096, 0, 0);
    printf_("  scalar STORE         : hit %6lu   cold %6lu\n",
            (unsigned long)st_h, (unsigned long)st_m);
  }

  // ==== EXTRA CASES ========================================================
  vset(0, 3, 4096);        // e16 LMUL=8 -- the kernels' operating point
  uint64_t B8 = (uint64_t)vl_now * 2UL;   // e16
  uint64_t step8 = (B8 + 4095UL) & ~4095UL;

  printf_("\n---- extra ----\n");

  // E1  writeback cost. Fill the LLC by READING (clean victims), measure a
  //     cold miss; then fill it by WRITING (dirty victims) and measure again.
  //     v1 took its "clean" number after a sweep that had already dirtied the
  //     cache, so it was dirty-vs-dirty.
  {
    read_fill(DIRTY_BUF, dirty_size);
    uint64_t clean = t_vle16(cold_take(step8 * (WARM + REPS)), step8, 0, 0);
    prefill(DIRTY_BUF, dirty_size);
    uint64_t dirty = t_vle16(cold_take(step8 * (WARM + REPS)), step8, 0, 0);
    printf_("  E1 miss, clean LLC   : %6lu\n", (unsigned long)clean);
    printf_("  E1 miss, dirty LLC   : %6lu   <- writeback cost = %lu\n",
            (unsigned long)dirty, (unsigned long)(dirty > clean ? dirty - clean : 0));
  }

  // E2  same-set thrash: stride = NumLines*LineSize maps every access to ONE
  //     set, so after `set_asso` of them the next evicts while the rest of the
  //     cache sits idle. Both rows are pre-touched, so the only difference is
  //     whether the working set fits in one set or is spread over many.
  {
    uint64_t spread = t_vle16(HOT_BUF, 4096, hot_win, 1);
    uint64_t conf   = t_vle16(CONF_BUF, conflict_stride, 4UL * 1024UL * 1024UL, 1);
    printf_("  E2 spread (hit)      : %6lu\n", (unsigned long)spread);
    printf_("  E2 same-set stride   : %6lu   <- conflict cost = %lu\n",
            (unsigned long)conf, (unsigned long)(conf > spread ? conf - spread : 0));
    if (conf < spread)
      printf_("  *** same-set faster than spread: check residency. SUSPECT.\n");
  }

  // E3  is a store write-allocate? If a cold STORE costs about as much as a
  //     cold LOAD, the line is fetched from DRAM before being written -- a
  //     read you never asked for, which matters for double buffering.
  {
    uint64_t sc = t_vse16(cold_take(step8 * (WARM + REPS)), step8, 0, 0);
    uint64_t lc = t_vle16(cold_take(step8 * (WARM + REPS)), step8, 0, 0);
    printf_("  E3 cold STORE %6lu vs cold LOAD %6lu  -> %s\n",
            (unsigned long)sc, (unsigned long)lc,
            (sc * 4 > lc * 3) ? "looks WRITE-ALLOCATE" : "looks write-around");
  }

  // E4  throughput: no per-access barrier, so refills overlap. Against E1's
  //     clean-miss latency this gives the achieved memory-level parallelism.
  {
    const int N = 64;
    uintptr_t c = cold_take(step8 * (N + 4));
    int64_t t0 = bar();
    for (int i = 0; i < N; i++)
      __asm__ volatile("vle16.v v8, (%0)" :: "r"(c + (uintptr_t)i * step8) : "memory");
    int64_t t1 = bar();
    uint64_t per = (uint64_t)(t1 - t0) / N;
    printf_("  E4 streaming miss    : %6lu cycles/instr (overlapped)\n",
            (unsigned long)per);
    printf_("     compare with E1 clean-miss latency: ratio = achieved MLP\n");
  }

  uint64_t itot = RD_CSR(0xB05) - ic0;
  printf_("\nI$ misses, whole run: %lu\n", (unsigned long)itot);
  if (overrun)
    printf_("*** COLD REGION WRAPPED -- later 'miss' rows may be hits. INVALID.\n");
  printf_("cold region used: %lu KiB of %lu KiB\n",
          (unsigned long)((cold_cursor - COLD_BASE) / 1024),
          (unsigned long)((COLD_END - COLD_BASE) / 1024));
  printf_("================ done ================\n");
  return overrun;
}
