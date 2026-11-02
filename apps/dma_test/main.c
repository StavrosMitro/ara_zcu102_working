// dma_test - hardware validation of the iDMA and the LLC scratchpad (SPM).
//
// Runs as a normal FP16 app so it inherits the PROVEN observability path:
// crt0 -> printf_() -> DDR console buffer -> crt0 `_eoc` LLC flush -> devmem.
// Build with -DNO_BOOT_LLC_FLUSH=1 (see notes 7c) or it hangs before main().
//
// Deliberately does NOT flush the LLC mid-run: that hard-stalls the core
// (fpga_port_archive.md 12.2). All output appears at once when the app exits.
//
// If the app hangs, rebuild with -DSKIP_SPM=1 to drop the LLC/SPM phases and
// isolate whether the DMA half works on its own.

#include <stdint.h>
#include "printf.h"

#define LLC_BASE    0x03001000UL
#define DMA_BASE    0x01000000UL
#define DDR_SRC     0x48000000UL   // clear of the program image at 0x40000000
#define DDR_DST     0x48010000UL
#define SPM_CACHED  0x10000000UL   // AmSpm
#define SPM_UNC     0x14000000UL   // AmSpmBaseUncached (needs the remap fix)

// iDMA reg64 frontend: 64-bit registers, accessed as 32-bit lo/hi pairs.
#define DMA_SRC 0x00
#define DMA_DST 0x08
#define DMA_LEN 0x10
#define DMA_CNF 0x18
#define DMA_NID 0x28              // *** READING THIS LAUNCHES A TRANSFER ***
#define DMA_DON 0x30
#define DMA_SSR 0x38
#define DMA_SDS 0x40
#define DMA_REP 0x48
// LLC config
#define LLC_SPM   0x00
#define LLC_FLUSH 0x08
#define LLC_COMMIT 0x10
#define LLC_FLUSHED 0x18
#define LLC_ASSO  0x28

#define NWORDS 256                // 1 KiB transfers
#define PAT(i) (0x0DE00000u | (uint32_t)(i))

static inline uint32_t rd32(uintptr_t a) { return *(volatile uint32_t *)a; }
static inline void wr32(uintptr_t a, uint32_t v) { *(volatile uint32_t *)a = v; }
static inline uint64_t rdcyc(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }

static int fails = 0;
static void check(const char *name, uint32_t got, uint32_t exp) {
  int ok = (got == exp);
  if (!ok) fails++;
  printf_("  [%s] %-22s got=0x%08lx exp=0x%08lx\n",
          ok ? "PASS" : "FAIL", name, (unsigned long)got, (unsigned long)exp);
}

// Program and launch one 1-D transfer. Returns cycles, or 0 on timeout.
static uint64_t dma_run(uint64_t src, uint64_t dst, uint32_t len) {
  wr32(DMA_BASE + DMA_SRC, (uint32_t)src);  wr32(DMA_BASE + DMA_SRC + 4, (uint32_t)(src >> 32));
  wr32(DMA_BASE + DMA_DST, (uint32_t)dst);  wr32(DMA_BASE + DMA_DST + 4, (uint32_t)(dst >> 32));
  wr32(DMA_BASE + DMA_LEN, len);            wr32(DMA_BASE + DMA_LEN + 4, 0);
  wr32(DMA_BASE + DMA_CNF, 0);              wr32(DMA_BASE + DMA_CNF + 4, 0);  // 1-D
  wr32(DMA_BASE + DMA_REP, 1);              wr32(DMA_BASE + DMA_REP + 4, 0);
  __asm__ volatile("fence" ::: "memory");
  uint64_t t0 = rdcyc();
  uint32_t id = rd32(DMA_BASE + DMA_NID);                 // launch
  for (int i = 0; i < 4000000; i++)
    if (rd32(DMA_BASE + DMA_DON) >= id) return rdcyc() - t0;
  return 0;                                               // timeout
}

int main(void) {
  printf_("\n===== dma_test: iDMA + SPM validation =====\n");

  // ---- T1: config registers hold their values -----------------------------
  // Offsets 0x08 and 0x38 are the ones the 128-bit truncation bug discarded.
  printf_("\n[T1] DMA config register read/write\n");
  const uint32_t offs[5] = {DMA_SRC, DMA_DST, DMA_LEN, DMA_SSR, DMA_SDS};
  const char *nms[5] = {"SRC_ADDR(0x00)", "DST_ADDR(0x08)", "NUM_BYTES(0x10)",
                        "STRIDE_SRC(0x38)", "STRIDE_DST(0x40)"};
  for (int i = 0; i < 5; i++) wr32(DMA_BASE + offs[i], 0xC0DE0000u | offs[i]);
  __asm__ volatile("fence" ::: "memory");
  for (int i = 0; i < 5; i++) check(nms[i], rd32(DMA_BASE + offs[i]), 0xC0DE0000u | offs[i]);

  // ---- T2: DDR -> DDR transfer -------------------------------------------
  printf_("\n[T2] DMA DDR->DDR, %d bytes\n", NWORDS * 4);
  volatile uint32_t *s = (volatile uint32_t *)DDR_SRC;
  volatile uint32_t *d = (volatile uint32_t *)DDR_DST;
  for (int i = 0; i < NWORDS; i++) { s[i] = PAT(i); d[i] = 0xBADBAD00u; }
  __asm__ volatile("fence" ::: "memory");
  uint64_t cyc = dma_run(DDR_SRC, DDR_DST, NWORDS * 4);
  if (!cyc) { printf_("  [FAIL] DMA timeout (DONE never reached id)\n"); fails++; }
  else {
    printf_("  transfer took %lu cycles\n", (unsigned long)cyc);
    int bad = -1;
    for (int i = 0; i < NWORDS; i++) if (d[i] != PAT(i)) { bad = i; break; }
    if (bad < 0) printf_("  [PASS] all %d words match\n", NWORDS);
    else { printf_("  [FAIL] word %d got=0x%08lx exp=0x%08lx\n",
                   bad, (unsigned long)d[bad], (unsigned long)PAT(bad)); fails++; }
  }

#ifndef SKIP_SPM
  // ---- T3: LLC geometry + SPM ways ---------------------------------------
  printf_("\n[T3] LLC config\n");
  printf_("  SET_ASSO=%lu  FLUSHED=0x%02lx\n",
          (unsigned long)rd32(LLC_BASE + LLC_ASSO),
          (unsigned long)rd32(LLC_BASE + LLC_FLUSHED));

  // ---- T4: SPM direct access + cached/uncached alias ----------------------
  // The uncached alias only works with the cheshire_soc.sv remap fix.
  printf_("\n[T4] SPM access and alias\n");
  volatile uint32_t *pc = (volatile uint32_t *)SPM_CACHED;
  volatile uint32_t *pu = (volatile uint32_t *)SPM_UNC;
  pc[0] = 0x5EED0001u; pc[1] = 0x5EED0002u;
  __asm__ volatile("fence" ::: "memory");
  check("SPM cached rw", pc[0], 0x5EED0001u);
  check("SPM alias unc==cached", pu[0], 0x5EED0001u);
  check("SPM alias unc[1]", pu[1], 0x5EED0002u);

  // ---- T5: DDR -> SPM (the double-buffering primitive) --------------------
  printf_("\n[T5] DMA DDR->SPM, %d bytes\n", NWORDS * 4);
  for (int i = 0; i < NWORDS; i++) { s[i] = PAT(i); pc[i] = 0xBADBAD00u; }
  __asm__ volatile("fence" ::: "memory");
  cyc = dma_run(DDR_SRC, SPM_CACHED, NWORDS * 4);
  if (!cyc) { printf_("  [FAIL] DMA timeout\n"); fails++; }
  else {
    printf_("  transfer took %lu cycles\n", (unsigned long)cyc);
    int bad = -1;
    for (int i = 0; i < NWORDS; i++) if (pc[i] != PAT(i)) { bad = i; break; }
    if (bad < 0) printf_("  [PASS] all %d words landed in SPM\n", NWORDS);
    else { printf_("  [FAIL] word %d got=0x%08lx exp=0x%08lx\n",
                   bad, (unsigned long)pc[bad], (unsigned long)PAT(bad)); fails++; }
  }
#else
  printf_("\n[T3-T5] SKIPPED (built with -DSKIP_SPM=1)\n");
#endif

  printf_("\n===== dma_test done: %d failure(s) =====\n", fails);
  return fails;
}
