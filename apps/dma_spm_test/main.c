// dma_spm_test: bare-metal hardware validation for the axi_llc SPM partition
// and the iDMA 2-D frontend on the Cheshire+CVA6+Ara ZCU102 build. Neither
// block has ever been exercised on real hardware before this test -- see
// fpga_port_archive.md section 4 for the register map this file implements
// and the exact rationale for each address.
//
// No floating point anywhere in this file (not even via printf %f/%lf):
// this branch's CVA6 has no scalar D hardware (RVD=0), and any double in a
// printf call gets promoted there by C variadic rules regardless of the
// app's own code -- that was the original conv_layer16only bring-up bug.
// Integer/hex-only output sidesteps it entirely, so this builds via the
// plain (non-fp16-only) .fpga target with no special toolchain override.

#include <stdint.h>

#ifdef SPIKE
#include "util.h"
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define REG32(addr) (*(volatile uint32_t *)(uintptr_t)(addr))

// ---- axi_llc config regs, AmLlc = 0x0300_1000 (fpga_port_archive.md §4.1) --
#define LLC_BASE 0x03001000UL
#define LLC_CFG_SPM_LOW (LLC_BASE + 0x00)
#define LLC_CFG_SPM_HIGH (LLC_BASE + 0x04)
#define LLC_CFG_FLUSH_LOW (LLC_BASE + 0x08)
#define LLC_CFG_FLUSH_HIGH (LLC_BASE + 0x0c)
#define LLC_COMMIT_CFG (LLC_BASE + 0x10)
#define LLC_FLUSHED_LOW (LLC_BASE + 0x18)
#define LLC_SET_ASSO_LOW (LLC_BASE + 0x28)
#define LLC_NUM_LINES_LOW (LLC_BASE + 0x30)
#define LLC_NUM_BLOCKS_LOW (LLC_BASE + 0x38)

// SPM address regions: cached alias and uncached alias of the same memory.
#define SPM_CACHED 0x10000000UL
#define SPM_UNCACHED 0x14000000UL
// 8 ways x 1024 lines x 8 blocks x 16B = 1 MiB total, 128 KiB/way.
// cfg_spm = 0x3F -> 6 ways SPM = 768 KiB, 2 ways left as cache.
#define SPM_WAYS_MASK 0x3FUL
#define SPM_SIZE_BYTES (768UL * 1024UL)

// ---- iDMA 2-D frontend, base 0x0100_0000 (fpga_port_archive.md §4.1) ------
#define IDMA_BASE 0x01000000UL
#define IDMA_SRC_ADDR (IDMA_BASE + 0x00)
#define IDMA_DST_ADDR (IDMA_BASE + 0x04)
#define IDMA_NUM_BYTES (IDMA_BASE + 0x08)
#define IDMA_CONF (IDMA_BASE + 0x0c)
#define IDMA_STRIDE_SRC (IDMA_BASE + 0x10)
#define IDMA_STRIDE_DST (IDMA_BASE + 0x14)
#define IDMA_NUM_REPETITIONS (IDMA_BASE + 0x18)
#define IDMA_STATUS (IDMA_BASE + 0x1c)
#define IDMA_NEXT_ID (IDMA_BASE + 0x20)
#define IDMA_DONE (IDMA_BASE + 0x24)

// DDR scratch buffers -- well clear of the loaded image (0x4000_0000+) and
// the stack top (_stack_top = 0x6000_0000, arch.link.fpga.ld).
#define DDR_SRC 0x48000000UL
#define DDR_DST 0x48100000UL

// All polling loops are bounded: an unbounded wait on hardware nobody has
// exercised before turns a real hang into "the sentinel never leaves
// 0xDEADDEAD" with no diagnosis. ~5M iterations of a tight lw+branch is a
// fraction of a second even at 50 MHz, but generous next to any real
// completion latency for these transfer sizes.
#define POLL_LIMIT 5000000UL

static int fail_count = 0;

static void check_eq32(const char *label, uint32_t got, uint32_t want) {
  if (got == want) {
    printf("  PASS %s: 0x%x\n", label, got);
  } else {
    printf("  FAIL %s: got 0x%x want 0x%x\n", label, got, want);
    fail_count++;
  }
}

// Flush every LLC way (mirrors crt0.S's _eoc flush exactly) so any dirty
// CVA6 write is visible to the iDMA/PS before it, and reconfigure cfg_spm
// as part of the same commit if spm_mask != 0xFFFFFFFF (sentinel "no
// reconfig, just flush").
static int llc_flush_and_configure(uint32_t spm_mask) {
  if (spm_mask != 0xFFFFFFFFUL) {
    REG32(LLC_CFG_SPM_LOW) = spm_mask;
    REG32(LLC_CFG_SPM_HIGH) = 0;
  }
  REG32(LLC_CFG_FLUSH_LOW) = 0xFF;
  REG32(LLC_CFG_FLUSH_HIGH) = 0;
  REG32(LLC_COMMIT_CFG) = 1;
  for (unsigned long i = 0; i < POLL_LIMIT; i++) {
    if (REG32(LLC_CFG_FLUSH_LOW) == 0)
      return 1;
  }
  return 0; // timed out
}

// Launch a 1-D iDMA transfer and wait for completion. Returns 1 on success,
// 0 on timeout. Prints every register touched so a wrong protocol guess is
// diagnosable from the console even if this fails.
static int idma_transfer_1d(uint32_t src, uint32_t dst, uint32_t nbytes) {
  REG32(IDMA_SRC_ADDR) = src;
  REG32(IDMA_DST_ADDR) = dst;
  REG32(IDMA_NUM_BYTES) = nbytes;
  REG32(IDMA_CONF) = 0;
  REG32(IDMA_STRIDE_SRC) = 0;
  REG32(IDMA_STRIDE_DST) = 0;
  REG32(IDMA_NUM_REPETITIONS) = 1;
  uint32_t launched_id = REG32(IDMA_NEXT_ID); // read triggers launch
  printf("  idma: src=0x%x dst=0x%x nbytes=%u launched_id=%u status=0x%x\n",
         src, dst, nbytes, launched_id, REG32(IDMA_STATUS));
  uint32_t done = 0;
  unsigned long i;
  for (i = 0; i < POLL_LIMIT; i++) {
    done = REG32(IDMA_DONE);
    if (done != 0 && done >= launched_id)
      break;
  }
  printf("  idma: done_reg=0x%x status=0x%x after %lu polls\n", done,
         REG32(IDMA_STATUS), i);
  return (i < POLL_LIMIT);
}

static void t0_geometry(void) {
  printf(">>> T0 LLC geometry sanity check\n");
  uint32_t ways = REG32(LLC_SET_ASSO_LOW);
  uint32_t lines = REG32(LLC_NUM_LINES_LOW);
  uint32_t blocks = REG32(LLC_NUM_BLOCKS_LOW);
  check_eq32("SET_ASSO_LOW (ways)", ways, 8);
  check_eq32("NUM_LINES_LOW", lines, 1024);
  check_eq32("NUM_BLOCKS_LOW", blocks, 8);
}

static void t1_spm_basic(void) {
  printf(">>> T1 SPM basic access (cfg_spm=0x%x, %lu KiB)\n",
         (unsigned)SPM_WAYS_MASK, SPM_SIZE_BYTES / 1024);
  if (!llc_flush_and_configure(SPM_WAYS_MASK)) {
    printf("  FAIL CFG_FLUSH never cleared (timeout) -- SPM not configured\n");
    fail_count++;
    return;
  }
  printf("  SPM partition committed, CFG_FLUSH cleared\n");

  // Write a known pattern through the cached alias.
  uint32_t n_words = 64;
  for (uint32_t i = 0; i < n_words; i++)
    REG32(SPM_CACHED + 4 * i) = 0xA5A50000UL + i;

  // Boundary word: last word of the 768 KiB SPM region.
  uint32_t last_off = SPM_SIZE_BYTES - 4;
  REG32(SPM_CACHED + last_off) = 0x5A5A5A5AUL;

  int ok = 1;
  for (uint32_t i = 0; i < n_words; i++) {
    uint32_t got = REG32(SPM_CACHED + 4 * i);
    uint32_t want = 0xA5A50000UL + i;
    if (got != want) {
      printf("  FAIL cached[%u]: got 0x%x want 0x%x\n", i, got, want);
      fail_count++;
      ok = 0;
    }
  }
  if (ok)
    printf("  PASS cached alias: %u/%u words round-tripped\n", n_words,
           n_words);

  check_eq32("cached boundary word (768KiB-4)", REG32(SPM_CACHED + last_off),
             0x5A5A5A5AUL);

  // Same memory through the uncached alias -- must see what CVA6 just wrote.
  ok = 1;
  for (uint32_t i = 0; i < n_words; i++) {
    uint32_t got = REG32(SPM_UNCACHED + 4 * i);
    uint32_t want = 0xA5A50000UL + i;
    if (got != want) {
      printf("  FAIL uncached[%u]: got 0x%x want 0x%x\n", i, got, want);
      fail_count++;
      ok = 0;
    }
  }
  if (ok)
    printf("  PASS uncached alias matches cached alias: %u/%u words\n",
           n_words, n_words);
  check_eq32("uncached boundary word (768KiB-4)",
             REG32(SPM_UNCACHED + last_off), 0x5A5A5A5AUL);

  // Write through the uncached alias, confirm visible via the cached one.
  REG32(SPM_UNCACHED + 8) = 0xC001C0DEUL;
  check_eq32("write-uncached/read-cached coherence", REG32(SPM_CACHED + 8),
             0xC001C0DEUL);
}

static void t2_t3_dma_roundtrip(void) {
  printf(">>> T2/T3 DMA DDR->SPM->DDR round trip\n");
  uint32_t n_words = 256; // 1 KiB
  uint32_t nbytes = n_words * 4;

  for (uint32_t i = 0; i < n_words; i++)
    REG32(DDR_SRC + 4 * i) = 0xDEAD0000UL + i;
  // Make the DDR writes visible outside CVA6's own view before the DMA
  // engine (a separate AXI master) reads them.
  if (!llc_flush_and_configure(0xFFFFFFFFUL)) {
    printf("  FAIL pre-DMA flush timed out\n");
    fail_count++;
    return;
  }

  printf("  T2: DMA DDR (0x%lx) -> SPM cached (0x%lx), %u bytes\n",
         (unsigned long)DDR_SRC, (unsigned long)SPM_CACHED, nbytes);
  if (!idma_transfer_1d((uint32_t)DDR_SRC, (uint32_t)SPM_CACHED, nbytes)) {
    printf("  FAIL T2 DMA timed out (DONE never reached launched_id)\n");
    fail_count++;
    return;
  }
  int ok = 1;
  for (uint32_t i = 0; i < n_words; i++) {
    uint32_t got = REG32(SPM_CACHED + 4 * i);
    uint32_t want = 0xDEAD0000UL + i;
    if (got != want) {
      if (ok)
        printf("  FAIL T2 first mismatch [%u]: got 0x%x want 0x%x\n", i, got,
               want);
      fail_count++;
      ok = 0;
    }
  }
  if (ok)
    printf("  PASS T2: %u/%u words landed correctly in SPM\n", n_words,
           n_words);

  printf("  T3: DMA SPM cached (0x%lx) -> DDR (0x%lx), %u bytes\n",
         (unsigned long)SPM_CACHED, (unsigned long)DDR_DST, nbytes);
  if (!idma_transfer_1d((uint32_t)SPM_CACHED, (uint32_t)DDR_DST, nbytes)) {
    printf("  FAIL T3 DMA timed out (DONE never reached launched_id)\n");
    fail_count++;
    return;
  }
  // The DMA writes DDR directly; flush again so CVA6's own read (which may
  // hit stale cache state for this address) sees the fresh data.
  llc_flush_and_configure(0xFFFFFFFFUL);
  ok = 1;
  for (uint32_t i = 0; i < n_words; i++) {
    uint32_t got = REG32(DDR_DST + 4 * i);
    uint32_t want = 0xDEAD0000UL + i;
    if (got != want) {
      if (ok)
        printf("  FAIL T3 first mismatch [%u]: got 0x%x want 0x%x\n", i, got,
               want);
      fail_count++;
      ok = 0;
    }
  }
  if (ok)
    printf("  PASS T3: %u/%u words round-tripped SPM->DDR, match original\n",
           n_words, n_words);
}

int main(void) {
  printf("dma_spm_test starting\n");

  t0_geometry();
  t1_spm_basic();
  t2_t3_dma_roundtrip();

  if (fail_count == 0)
    printf("ALL TESTS PASSED\n");
  else
    printf("%d CHECK(S) FAILED\n", fail_count);

  return fail_count;
}
