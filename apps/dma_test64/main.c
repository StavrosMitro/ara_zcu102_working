// dma_test64 - iDMA validation over the 64-bit MMIO interface.
//
// WHY THIS EXISTS
// ---------------
// apps/dma_test/main.c programs the descriptor with 32-bit stores and "fixes
// up" the upper half at offset+4. Both halves of that assumption are wrong,
// and the RTL says so:
//
//  1) The register file decodes ONLY aligned 8-byte offsets.
//     idma_reg64_2d_frontend_reg_top.sv:401
//         addr_hit[0] = (reg_addr == IDMA_..._SRC_ADDR_OFFSET);
//     is an exact compare, so every `offset+4` store is an addrmiss and is
//     dropped. CVA6 never sees it: wt_axi_adapter.sv pops the B FIFO and
//     returns a plain store ACK without inspecting BRESP, so trap == 0.
//
//  2) A 32-bit store to the LOW half is NOT rejected and does NOT merge.
//         reg_pkg.sv:126   parameter logic [3:0] PERMIT [10]   <- 4 bits
//         reg_top.sv:38    logic [DBW-1:0] reg_be              <- 8 bits (DW=64)
//         reg_top.sv:418   wr_err = ... |(PERMIT[0] & ~reg_be)
//     PERMIT zero-extends to 8'h0F, so a byte-enable of 8'h0F passes the
//     guard that exists precisely to catch partial writes; reg_top.sv:431
//     then writes reg_wdata[63:0] wholesale, with no byte-enable merge.
//     CVA6 replicates a 32-bit store into BOTH halves of the 64-bit AXI beat
//     (wt_dcache_wbuffer.sv repData64), so SRC lands as {V,V}.
//
//     With the old test's values that yields
//         SRC = 0x4800000048000000   unmapped
//         LEN = 0x0000040000000400   ~4.4 TB
//         REP = 0x0000000100000001   ~4.3e9 repetitions
//     on top of the strides its own T1 register probe left behind and never
//     cleared. That is a runaway transfer, not a memcpy.
//
// Every access here is therefore an aligned `volatile uint64_t` load/store,
// i.e. plain RV64 ld/sd. XLEN is 64 (cv64a6_imafdcv_sv39_config_pkg.sv:13);
// what this build dropped is RVD (FP64, line 22), which has nothing to do
// with integer 64-bit memory ops -- crt0.S publishes the exit code with `sd`
// on every single run.
//
// PHASE LADDER
// ------------
// The console only reaches the host at _eoc, so a hang prints NOTHING. The
// diagnostic is therefore *which phase hangs*. Rebuild to move the ladder --
// no new bitstream is needed for any of this:
//     -DDMA_PHASE=1   registers only, never reads NEXT_ID  -> cannot hang
//     -DDMA_PHASE=2   + 16-byte DDR->DDR transfer
//     -DDMA_PHASE=3   + 1 KiB DDR->DDR transfer            (default)
//
// SPM is deliberately out of scope: it needs CFG_SPM programmed, and a
// mid-run LLC reconfigure has hard-stalled the core before
// (fpga_port_archive.md 12.2). Prove the DMA first.
//
// Build with -DNO_BOOT_LLC_FLUSH=1 like the other FP16 apps.

#include <stdint.h>
#include "printf.h"

#ifndef DMA_PHASE
#define DMA_PHASE 3
#endif

#define DMA_BASE 0x01000000UL

// idma_reg64_2d_frontend, from sw/include/regs/idma.h. 8-byte spacing.
#define R_SRC 0x00
#define R_DST 0x08
#define R_LEN 0x10
#define R_CNF 0x18
#define R_STA 0x20
#define R_NID 0x28   // *** READING THIS LAUNCHES A TRANSFER ***
#define R_DON 0x30
#define R_SSR 0x38
#define R_SDS 0x40
#define R_REP 0x48

#define DDR_SRC 0x48000000UL   // clear of the image at 0x40000000
#define DDR_DST 0x48010000UL

// Bounded everywhere: an unbounded wait turns a diagnosable stall into a
// silent 0xDEADDEAD. 2M MMIO loads is already seconds at 50 MHz.
#define POLL_LIMIT 2000000UL

static int fails = 0;

static inline uint64_t rd64(uintptr_t a) { return *(volatile uint64_t *)a; }
static inline void wr64(uintptr_t a, uint64_t v) { *(volatile uint64_t *)a = v; }
static inline void wr32(uintptr_t a, uint32_t v) { *(volatile uint32_t *)a = v; }
static inline void fence(void) { __asm__ volatile("fence" ::: "memory"); }

// Print as two 32-bit halves: the replication bug is visible at a glance, and
// it does not depend on the minimal printf supporting 64-bit conversions.
static void p64(const char *label, uint64_t v) {
  printf_("  %-16s = %08lx_%08lx\n", label,
          (unsigned long)(uint32_t)(v >> 32), (unsigned long)(uint32_t)v);
}

static void check64(const char *label, uint64_t got, uint64_t exp) {
  if (got == exp) {
    printf_("  [PASS] %-16s %08lx_%08lx\n", label,
            (unsigned long)(uint32_t)(got >> 32), (unsigned long)(uint32_t)got);
  } else {
    fails++;
    printf_("  [FAIL] %-16s got %08lx_%08lx exp %08lx_%08lx\n", label,
            (unsigned long)(uint32_t)(got >> 32), (unsigned long)(uint32_t)got,
            (unsigned long)(uint32_t)(exp >> 32), (unsigned long)(uint32_t)exp);
  }
}

#if DMA_PHASE >= 2
// Program a clean 1-D descriptor. Every field is written, including the ones
// a previous phase may have dirtied -- the old test's stale strides are half
// of why its transfers ran away.
static void dma_program_1d(uint64_t src, uint64_t dst, uint64_t len) {
  wr64(DMA_BASE + R_SRC, src);
  wr64(DMA_BASE + R_DST, dst);
  wr64(DMA_BASE + R_LEN, len);
  wr64(DMA_BASE + R_CNF, 0);      // no decouple / deburst / serialize
  wr64(DMA_BASE + R_SSR, 0);      // strides MUST be cleared explicitly
  wr64(DMA_BASE + R_SDS, 0);
  // num_reps = 1, NOT 0. Cheshire always instantiates the 2-D frontend, and
  // idma_nd_midend.sv:97  burst_req_valid_o = nd_req_valid_i & !zero
  // means reps==0 issues no backend burst at all -- it only manufactures an
  // ND_MIDEND error response. sw/include/dif/dma.h:83 writes 0 and is broken
  // for exactly this reason.
  wr64(DMA_BASE + R_REP, 1);
  fence();
}

static void dma_dump(const char *when) {
  printf_("  descriptor read-back (%s):\n", when);
  p64("SRC", rd64(DMA_BASE + R_SRC));
  p64("DST", rd64(DMA_BASE + R_DST));
  p64("NUM_BYTES", rd64(DMA_BASE + R_LEN));
  p64("STRIDE_SRC", rd64(DMA_BASE + R_SSR));
  p64("STRIDE_DST", rd64(DMA_BASE + R_SDS));
  p64("NUM_REPS", rd64(DMA_BASE + R_REP));
}

// Launch and wait. Returns cycles, 0 on timeout, and sets *idp to the id the
// frontend handed back.
//
// A zero id means the launch was REJECTED: idma_reg64_2d_frontend.sv:78 only
// produces an id when num_bytes != 0, and drives next_id.d = '0 otherwise.
// The old test compared `done >= id` without checking this, so a rejected
// launch read as an instant pass.
static uint64_t dma_launch_wait(uint64_t *idp) {
  uint64_t t0, id, done;
  __asm__ volatile("rdcycle %0" : "=r"(t0));
  id = rd64(DMA_BASE + R_NID);          // <-- the launch
  *idp = id;
  if (id == 0) return 0;
  for (unsigned long i = 0; i < POLL_LIMIT; i++) {
    done = rd64(DMA_BASE + R_DON);
    if (done >= id) {
      uint64_t t1;
      __asm__ volatile("rdcycle %0" : "=r"(t1));
      return t1 - t0;
    }
  }
  return 0;
}

static void run_transfer(const char *name, uint64_t len) {
  const unsigned nw = (unsigned)(len / 4);
  volatile uint32_t *s = (volatile uint32_t *)DDR_SRC;
  volatile uint32_t *d = (volatile uint32_t *)DDR_DST;

  printf_("\n[%s] DDR->DDR, %lu bytes\n", name, (unsigned long)len);
  for (unsigned i = 0; i < nw; i++) { s[i] = 0x0DE00000u | i; d[i] = 0xBADBAD00u; }
  fence();

  dma_program_1d(DDR_SRC, DDR_DST, len);
  dma_dump("before launch");

  uint64_t id = 0;
  uint64_t cyc = dma_launch_wait(&id);
  printf_("  next_id = %lu\n", (unsigned long)id);
  if (id == 0) {
    printf_("  [FAIL] launch REJECTED (num_bytes seen as 0)\n"); fails++; return;
  }
  if (!cyc) { printf_("  [FAIL] DONE never reached id (bounded poll expired)\n"); fails++; return; }
  printf_("  transfer took %lu cycles\n", (unsigned long)cyc);

  int bad = -1;
  for (unsigned i = 0; i < nw; i++)
    if (d[i] != (0x0DE00000u | i)) { bad = (int)i; break; }
  if (bad < 0) printf_("  [PASS] all %u words match\n", nw);
  else {
    fails++;
    printf_("  [FAIL] word %d got=0x%08lx exp=0x%08lx\n", bad,
            (unsigned long)d[bad], (unsigned long)(0x0DE00000u | (unsigned)bad));
  }
}
#endif

int main(void) {
  printf_("\n===== dma_test64 (DMA_PHASE=%d) =====\n", DMA_PHASE);

  // ---- P1a: 64-bit register write/read-back ------------------------------
  // Halves are deliberately DIFFERENT. A replicated 32-bit store would make
  // them equal; a truncating converter would zero the top one. Either bug is
  // visible in the printed value, not just in a pass/fail bit.
  printf_("\n[P1a] 64-bit descriptor registers\n");
  const uintptr_t offs[6] = {R_SRC, R_DST, R_LEN, R_SSR, R_SDS, R_REP};
  const char *nms[6] = {"SRC(0x00)", "DST(0x08)", "LEN(0x10)",
                        "STRIDE_SRC(0x38)", "STRIDE_DST(0x40)", "NUM_REPS(0x48)"};
  for (int i = 0; i < 6; i++)
    wr64(DMA_BASE + offs[i], 0xA5A50000ULL << 32 | (0xC0DE0000ULL | offs[i]));
  fence();
  for (int i = 0; i < 6; i++)
    check64(nms[i], rd64(DMA_BASE + offs[i]),
            0xA5A50000ULL << 32 | (0xC0DE0000ULL | offs[i]));

  // ---- P1b: does a 32-bit store really corrupt the upper half? -----------
  // This is the one claim about this SoC that static analysis could not
  // settle: whether CVA6's replicated beat survives the config port's
  // 64->128->64 round trip on a 4-lane build. Write a known 64-bit value,
  // overwrite only the low half with a 32-bit store, read the full 64 back.
  //   upper == 0xDEAD0000  -> byte enables were honoured (no bug here)
  //   upper == 0x11112222  -> CVA6's replication reached the register  <-- bug
  //   upper == 0x00000000  -> the converter zeroed it
  // Harmless either way: nothing is launched, and P2 reprograms everything.
  printf_("\n[P1b] 32-bit store into a 64-bit register\n");
  wr64(DMA_BASE + R_SRC, 0xDEAD0000BEEF0000ULL);
  fence();
  wr32(DMA_BASE + R_SRC, 0x11112222u);
  fence();
  p64("SRC after sw", rd64(DMA_BASE + R_SRC));
  printf_("  expect upper=DEAD0000 if byte enables work,"
          " 11112222 if replicated, 00000000 if zeroed\n");

  // ---- P1c: STATUS is readable and the backend is idle -------------------
  // A pure read, no launch side effect (only NEXT_ID has one).
  printf_("\n[P1c] STATUS\n");
  p64("STATUS", rd64(DMA_BASE + R_STA));
  p64("DONE", rd64(DMA_BASE + R_DON));

#if DMA_PHASE >= 2
  run_transfer("P2", 16);          // small enough that a bad descriptor cannot run away
#endif
#if DMA_PHASE >= 3
  run_transfer("P3", 1024);
#endif
#if DMA_PHASE < 2
  printf_("\n[P2+] skipped: NEXT_ID is never read at DMA_PHASE=1\n");
#endif

  printf_("\n===== dma_test64 done: %d failure(s) =====\n", fails);
  return fails;
}
