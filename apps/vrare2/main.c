// vrare2 -- separate the MASK GENERATOR from the MASK CONSUMER, and prove
// whether the core dies silently or takes a trap.
//
// vrare showed vcpop.m / vcompress.vm / viota.m all disagreeing with
// expectation and vmsif.m hanging outright.  But every one of those readings
// depended on a mask built with vid+vand+vmseq, so "the mask is wrong" and
// "the consumers are wrong" were not separable.  Here they are:
//
//   PHASE A  builds masks and dumps them with vsm.v -- a plain store of the
//            mask register.  No comparison, no consumer, nothing that could
//            hide a fault.  vmset.m / vmclr.m need no compare at all.
//   PHASE B  feeds ONLY masks whose raw bits phase A just verified into the
//            consumers.  A failure here cannot be blamed on the generator.
//   PHASE C  mask-to-mask ops, read back with vsm.v -- never through vcpop.m,
//            which vrare already showed is untrustworthy.  vmsif.m goes LAST
//            because it hangs.
//
// Trap detection: 0x50000050 is set to a private sentinel at startup and
// flushed.  crt0's trap_vector writes 0x7A7A7A7A there on any exception.  So
// after the run that word says exactly one of three things:
//     0xC1EAC1EA -> no trap, the core died silently  (a real deadlock)
//     0x7A7A7A7A -> an exception fired
//     anything else -> the word never reached DDR

#include <stdint.h>
#include <stddef.h>
#include "printf.h"

#define LLC_BASE  0x03001000UL
#define A_STEP 0x50000100UL
#define A_DONE 0x50000104UL
#define A_FAIL 0x50000108UL
#define A_SEEN 0x5000010CUL
#define A_TRAP 0x50000050UL
#define TRAP_CLEAN 0xC1EAC1EAUL

static uint32_t fails, seen, idx;

static inline void llc_flush(void) {
  uint32_t ways = *(volatile uint32_t *)(LLC_BASE + 0x28);
  uint32_t mask = (ways >= 32) ? 0xFFFFFFFFu : ((1u << ways) - 1u);
  *(volatile uint32_t *)(LLC_BASE + 0x08) = mask;
  *(volatile uint32_t *)(LLC_BASE + 0x0c) = 0;
  *(volatile uint32_t *)(LLC_BASE + 0x10) = 1;
  while (*(volatile uint32_t *)(LLC_BASE + 0x08)) { }
}
static inline void mark(uintptr_t a, uint32_t v) {
  *(volatile uint32_t *)a = v;
  __asm__ volatile("fence" ::: "memory");
  llc_flush();
}
#define STEP(n) mark(A_STEP, (n))
#define DONE(n) mark(A_DONE, (n))

static uint8_t  mbuf[256] __attribute__((aligned(64)));
static uint32_t buf[1024] __attribute__((aligned(64)));

static void say(const char *phase, const char *what, int ok, unsigned long got,
                unsigned long want) {
  seen++;
  if (!ok) fails |= (1u << (idx < 31 ? idx : 31));
  mark(A_FAIL, fails); mark(A_SEEN, seen);
  printf_("  %-3s %-16s %-4s got 0x%08lx  want 0x%08lx\n",
          phase, what, ok ? "ok" : "BAD", got, want);
}
// first 8 mask bytes as one word pair, printed as a 64-bit hex
static unsigned long mask64(void) {
  unsigned long v = 0;
  for (int i = 7; i >= 0; i--) v = (v << 8) | mbuf[i];
  return v;
}

int main(void) {
  printf_("\n===== vrare2: mask generator vs mask consumer =====\n");
  mark(A_STEP, 0); mark(A_DONE, 0); mark(A_FAIL, 0); mark(A_SEEN, 0);
  mark(A_TRAP, TRAP_CLEAN);           // private sentinel: "no trap yet"
  printf_("trap word 0x%08lx seeded with 0x%08lx (0x7A7A7A7A = exception)\n",
          (unsigned long)A_TRAP, (unsigned long)TRAP_CLEAN);

  size_t vl;
  long a0;
  __asm__ volatile("vsetvli %0, zero, e32, m1, ta, ma" : "=r"(vl) :: "memory");
  printf_("\n-- e32,m1  VLMAX=%lu  -> a mask is %lu bytes --\n",
          (unsigned long)vl, (unsigned long)(vl / 8));

  // ================= PHASE A : GENERATOR, raw dump only =================
  printf_("\nPHASE A -- build a mask, store it with vsm.v, look at the bits\n");

  idx = 1; STEP(1);
  __asm__ volatile("vmclr.m v0\n\t vsm.v v0, (%0)" :: "r"(mbuf) : "memory");
  DONE(1);
  say("A1", "vmclr.m", mask64() == 0UL, mask64(), 0UL);

  idx = 2; STEP(2);
  __asm__ volatile("vmset.m v0\n\t vsm.v v0, (%0)" :: "r"(mbuf) : "memory");
  DONE(2);
  say("A2", "vmset.m", mask64() == 0xFFFFFFFFFFFFFFFFUL, mask64(),
      0xFFFFFFFFFFFFFFFFUL);

  idx = 3; STEP(3);
  __asm__ volatile("vid.v v8\n\t vmseq.vi v0, v8, 3\n\t vsm.v v0, (%0)"
                   :: "r"(mbuf) : "memory");
  DONE(3);
  say("A3", "vmseq.vi ==3", mask64() == 0x8UL, mask64(), 0x8UL);

  idx = 4; STEP(4);
  __asm__ volatile("vid.v v8\n\t vand.vi v9, v8, 1\n\t vmseq.vi v0, v9, 0\n\t"
                   "vsm.v v0, (%0)" :: "r"(mbuf) : "memory");
  DONE(4);
  say("A4", "even mask", mask64() == 0x5555555555555555UL, mask64(),
      0x5555555555555555UL);

  // ================= PHASE B : CONSUMERS on verified masks ==============
  printf_("\nPHASE B -- feed the consumers masks whose bits phase A just showed\n");

  idx = 5; STEP(5);
  __asm__ volatile("vmset.m v0\n\t vcpop.m %0, v0" : "=r"(a0) :: "memory");
  DONE(5);
  say("B1", "vcpop all-ones", a0 == (long)vl, (unsigned long)a0, (unsigned long)vl);

  idx = 6; STEP(6);
  __asm__ volatile("vmclr.m v0\n\t vcpop.m %0, v0" : "=r"(a0) :: "memory");
  DONE(6);
  say("B2", "vcpop all-zero", a0 == 0, (unsigned long)a0, 0UL);

  idx = 7; STEP(7);
  __asm__ volatile("vid.v v8\n\t vand.vi v9, v8, 1\n\t vmseq.vi v0, v9, 0\n\t"
                   "vcpop.m %0, v0" : "=r"(a0) :: "memory");
  DONE(7);
  say("B3", "vcpop even", a0 == (long)(vl / 2), (unsigned long)a0,
      (unsigned long)(vl / 2));

  idx = 8; STEP(8);
  __asm__ volatile("vid.v v8\n\t vand.vi v9, v8, 1\n\t vmseq.vi v0, v9, 0\n\t"
                   "vid.v v12\n\t vcompress.vm v16, v12, v0\n\t vse32.v v16, (%0)"
                   :: "r"(buf) : "memory");
  DONE(8);
  say("B4", "vcompress[1]", buf[1] == 2, buf[1], 2UL);

  idx = 9; STEP(9);
  __asm__ volatile("vid.v v8\n\t vand.vi v9, v8, 1\n\t vmseq.vi v0, v9, 0\n\t"
                   "viota.m v16, v0\n\t vse32.v v16, (%0)" :: "r"(buf) : "memory");
  DONE(9);
  say("B5", "viota[4]", buf[4] == 2, buf[4], 2UL);

  // ============ PHASE C : mask->mask, read back with vsm.v ==============
  printf_("\nPHASE C -- mask-to-mask ops, read back RAW (never through vcpop)\n");

  idx = 10; STEP(10);
  __asm__ volatile("vid.v v8\n\t vmseq.vi v1, v8, 3\n\t vmsbf.m v2, v1\n\t"
                   "vsm.v v2, (%0)" :: "r"(mbuf) : "memory");
  DONE(10);
  say("C1", "vmsbf.m", mask64() == 0x7UL, mask64(), 0x7UL);

  idx = 11; STEP(11);
  __asm__ volatile("vid.v v8\n\t vmseq.vi v1, v8, 3\n\t vmsof.m v2, v1\n\t"
                   "vsm.v v2, (%0)" :: "r"(mbuf) : "memory");
  DONE(11);
  say("C2", "vmsof.m", mask64() == 0x8UL, mask64(), 0x8UL);

  // everything above is now safely in DDR before we touch the one that hangs
  printf_("\n-- trap word before the suspect instruction: 0x%08lx --\n",
          (unsigned long)*(volatile uint32_t *)A_TRAP);
  llc_flush();

  idx = 12; STEP(12);                      // <<<< this one hung vrare
  __asm__ volatile("vid.v v8\n\t vmseq.vi v1, v8, 3\n\t vmsif.m v2, v1\n\t"
                   "vsm.v v2, (%0)" :: "r"(mbuf) : "memory");
  DONE(12);
  say("C3", "vmsif.m", mask64() == 0xFUL, mask64(), 0xFUL);

  STEP(99);
  printf_("\n===== vrare2 done: %lu ran, fail mask 0x%08lx, trap word 0x%08lx =====\n",
          (unsigned long)seen, (unsigned long)fails,
          (unsigned long)*(volatile uint32_t *)A_TRAP);
  mark(A_DONE, 99);
  return (int)fails;
}
