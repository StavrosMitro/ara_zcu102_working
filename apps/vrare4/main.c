// vrare4 -- is the FPGA "vcpop = 12" caused by Defect A (vid residue)?
//
// The RTL-simulation diagnosis (ARA_STREAM_HARDWARE_DIAGNOSIS_20260928.md)
// found that vid.v leaves an unacknowledged ALU payload in the lanes, and the
// NEXT mask operation consumes it instead of its own operand.  vcpop.m reads
// exactly that ALU-operand path (masku.sv:860).  In simulation, vmset->vcpop
// with no vid before it returns 64.  On the FPGA (vrare2) every vcpop ran
// AFTER a vid and returned 12.
//
// This program settles it within ONE run, same bitstream, same boot:
//   PHASE 1  the tests, CLEAN -- no vid has executed since reset
//   PHASE 2  one bare vid.v, then the very same tests again
// Clean correct + contaminated wrong  => 12 is Defect A contamination.
// Clean already 12                    => FPGA-specific, not contamination.
//
// Every destination GPR is preloaded with 0x0BADF00D.  If it comes back
// unchanged, the instruction never wrote its register.
//
// Masks are built WITHOUT vid (vmset/vmclr, or data loaded from RAM) and
// without vlm.v (Defect D).  vl = 64 at e32,m1 is exactly 4 slices of 16, so
// Defect C (tail bits in a partial slice) cannot interfere.
//
// LLC is flushed around every step so results survive a hang in phase 2.

#include <stdint.h>
#include "printf.h"

#define LLC_BASE   0x03001000UL
#define A_STEP     0x50000100UL
#define A_DONE     0x50000104UL
#define A_RES      0x50000110UL          // result of step n at A_RES + 4*n
#define A_TRAP     0x50000050UL
#define TRAP_CLEAN 0xC1EAC1EAUL
#define SENT       0x0BADF00DL

static uint32_t par[64] __attribute__((aligned(64)));

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
static void rec(int n, long r, long want) {
  mark(A_RES + 4 * n, (uint32_t)r);
  mark(A_DONE, n);
  printf_("  %2d  got %-10ld (0x%08lx)  want %-3ld  %s%s\n", n, r,
          (unsigned long)(uint32_t)r, want, r == want ? "ok" : "BAD",
          r == SENT ? "   <- register NEVER WRITTEN" : "");
}

#define VCPOP_ONES(r)  __asm__ volatile("vmset.m v0\n\t vcpop.m %0, v0" : "+r"(r) :: "memory")
#define VCPOP_ZERO(r)  __asm__ volatile("vmclr.m v0\n\t vcpop.m %0, v0" : "+r"(r) :: "memory")
#define VCPOP_EVEN(r)  __asm__ volatile("vle32.v v8, (%1)\n\t vmseq.vi v0, v8, 0\n\t" \
                                        "vcpop.m %0, v0" : "+r"(r) : "r"(par) : "memory")

int main(void) {
  size_t vl;
  long r;
  mark(A_STEP, 0); mark(A_DONE, 0); mark(A_TRAP, TRAP_CLEAN);
  for (int i = 0; i < 64; i++) par[i] = (uint32_t)(i & 1);   // 0,1,0,1,... from RAM
  __asm__ volatile("fence" ::: "memory"); llc_flush();

  printf_("\n===== vrare4: is vcpop=12 caused by vid residue? =====\n");
  __asm__ volatile("vsetvli %0, zero, e32, m1, ta, ma" : "=r"(vl) :: "memory");
  printf_("vl=%lu (want 64: exactly 4 slices, no partial slice)\n", (unsigned long)vl);

  printf_("\nPHASE 1 -- CLEAN: no vid has run since reset\n");
  mark(A_STEP, 1); r = SENT; VCPOP_ONES(r); rec(1, r, (long)vl);
  mark(A_STEP, 2); r = SENT; VCPOP_ZERO(r); rec(2, r, 0);
  mark(A_STEP, 3); r = SENT; VCPOP_EVEN(r); rec(3, r, (long)vl / 2);
  mark(A_STEP, 4); r = SENT; VCPOP_ONES(r); rec(4, r, (long)vl);

  printf_("\nPHASE 2 -- one bare vid.v, result unused, then the SAME tests\n");
  mark(A_STEP, 10);
  __asm__ volatile("vid.v v8" ::: "memory");
  mark(A_DONE, 10);
  printf_("  10  vid.v executed\n");
  mark(A_STEP, 11); r = SENT; VCPOP_ONES(r); rec(11, r, (long)vl);
  mark(A_STEP, 12); r = SENT; VCPOP_ZERO(r); rec(12, r, 0);
  mark(A_STEP, 13); r = SENT; VCPOP_EVEN(r); rec(13, r, (long)vl / 2);
  mark(A_STEP, 14); r = SENT; VCPOP_ONES(r); rec(14, r, (long)vl);

  printf_("\nverdict:\n");
  printf_("  1-4 ok and 11-14 BAD  -> vcpop=12 IS Defect A contamination\n");
  printf_("  1-4 already BAD       -> FPGA-specific, not contamination\n");

  mark(A_STEP, 99);
  printf_("\n===== vrare4 done, trap word 0x%08lx =====\n",
          (unsigned long)*(volatile uint32_t *)A_TRAP);
  mark(A_DONE, 99);
  return 0;
}
