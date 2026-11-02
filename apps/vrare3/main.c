// vrare3 -- does vcpop.m WRITE its destination register at all?
//
// vrare2 showed vcpop.m returning 12 for an all-ones mask, an all-zeros mask
// and an even mask alike.  Two mechanisms fit "always the same answer", and
// they call for completely different fixes:
//
//   (a) the counter carries residue from earlier instructions.  Upstream fixes
//       this (8b98182, "Fix vfirst/vpop reset conditions") by clearing
//       popcount_d / vfirst_count_d at instruction START; ours only clears at
//       commit.  If this is it, porting that fix helps.
//   (b) the instruction never delivers its scalar result, so the destination
//       GPR keeps whatever it already held and 12 is just leftover.  masku's
//       scalar interface has result_scalar_o + result_scalar_valid_o and NO
//       ready/ack input -- even on upstream main today -- so a missed
//       one-cycle valid would do exactly this.  If this is it, the upstream
//       reset fix changes nothing and porting it would be wasted.
//
// (a) predicts the answer CHANGES between back-to-back identical vcpops.
// (b) predicts the destination register is left EXACTLY as we pre-loaded it.
// Pre-loading a sentinel separates them in one instruction.

#include <stdint.h>
#include "printf.h"

#define LLC_BASE 0x03001000UL
#define A_STEP 0x50000100UL
#define A_DONE 0x50000104UL

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

int main(void) {
  size_t vl;
  long r1, r2, r3, r4;
  mark(A_STEP, 0); mark(A_DONE, 0);
  printf_("\n===== vrare3: does vcpop write its destination? =====\n");
  __asm__ volatile("vsetvli %0, zero, e32, m1, ta, ma" : "=r"(vl) :: "memory");
  printf_("VLMAX=%lu  -> an all-ones mask must count %lu\n",
          (unsigned long)vl, (unsigned long)vl);

  // ---- 1. pre-load the destination with a sentinel the hardware cannot invent
  mark(A_STEP, 1);
  r1 = 0x0BADF00D;
  __asm__ volatile("vmset.m v0\n\t vcpop.m %0, v0" : "+r"(r1) :: "memory");
  mark(A_DONE, 1);
  printf_("\n 1  vcpop into a0 preloaded 0x0BADF00D : got 0x%08lx\n", (unsigned long)r1);
  printf_("      -> unchanged means the instruction NEVER WRITES the register\n");

  // ---- 2. a different sentinel, to rule out coincidence
  mark(A_STEP, 2);
  r2 = 0x1234;
  __asm__ volatile("vmset.m v0\n\t vcpop.m %0, v0" : "+r"(r2) :: "memory");
  mark(A_DONE, 2);
  printf_(" 2  same, preloaded 0x00001234          : got 0x%08lx\n", (unsigned long)r2);
  printf_("      -> if 1 and 2 both keep their own sentinel, it is mechanism (b)\n");
  printf_("      -> if both become the SAME number, it writes, but wrongly\n");

  // ---- 3/4. two identical vcpops back to back: does the answer drift?
  mark(A_STEP, 3);
  r3 = 0;
  __asm__ volatile("vmset.m v0\n\t vcpop.m %0, v0" : "+r"(r3) :: "memory");
  mark(A_DONE, 3);
  mark(A_STEP, 4);
  r4 = 0;
  __asm__ volatile("vmset.m v0\n\t vcpop.m %0, v0" : "+r"(r4) :: "memory");
  mark(A_DONE, 4);
  printf_("\n 3  vcpop all-ones, dest preloaded 0    : got %ld\n", r3);
  printf_(" 4  the very same again                 : got %ld\n", r4);
  printf_("      -> 3 != 4 means the counter ACCUMULATES: mechanism (a),\n");
  printf_("         and the upstream reset fix is the right one\n");

  // ---- 5. vfirst shares the same scalar path
  mark(A_STEP, 5);
  long f = 0x5EED;
  __asm__ volatile("vid.v v8\n\t vmseq.vi v0, v8, 3\n\t vfirst.m %0, v0"
                   : "+r"(f) :: "memory");
  mark(A_DONE, 5);
  printf_("\n 5  vfirst, dest preloaded 0x5EED       : got 0x%08lx  want 3\n",
          (unsigned long)f);

  mark(A_STEP, 99); mark(A_DONE, 99);
  printf_("\n===== vrare3 done =====\n");
  return 0;
}
