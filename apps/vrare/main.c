// vrare -- exercise every vector instruction in the mask/permute/index family
// that fconv_cmp uses and no other app on this rig ever has.
//
// WHY THE FLUSH: every DDR address the ARM reads back (console buffer, length
// counter, sentinel, these markers) is inside LlcOutRegionStart..End =
// 0x4000_0000..0x1_0000_0000, which cheshire_pkg lists in CachedRegionAddrBase.
// CVA6 stores go write-through to the LLC, NOT to DDR.  They only reach DDR
// when something flushes -- normally crt0's _eoc, at the very end.  So a hung
// run shows the ARM nothing at all, no matter how far it actually got.  This
// was measured on 2026-09-11: a run's console read 309 bytes at the poll
// timeout and 580 bytes seconds later, after it finished and flushed.
//
// Therefore: STEP is written and the LLC is flushed before AND after every
// single instruction under test.  If the board hangs, `devmem 0x50000100`
// names the exact instruction that did it.

#include <stdint.h>
#include <stddef.h>
#include "printf.h"

#define LLC_BASE   0x03001000UL
#define LLC_FLUSH  (LLC_BASE + 0x08)
#define LLC_COMMIT (LLC_BASE + 0x10)
#define LLC_ASSO   (LLC_BASE + 0x28)

#define A_STEP 0x50000100UL   // test we are ABOUT to execute
#define A_DONE 0x50000104UL   // last test that came back
#define A_FAIL 0x50000108UL   // bitmask of wrong answers
#define A_SEEN 0x5000010CUL   // how many tests produced a result

static uint32_t fails, seen;

static inline void llc_flush(void) {
  uint32_t ways = *(volatile uint32_t *)LLC_ASSO;
  uint32_t mask = (ways >= 32) ? 0xFFFFFFFFu : ((1u << ways) - 1u);
  *(volatile uint32_t *)LLC_FLUSH        = mask;
  *(volatile uint32_t *)(LLC_BASE + 0xc) = 0;
  *(volatile uint32_t *)LLC_COMMIT       = 1;
  while (*(volatile uint32_t *)LLC_FLUSH) { }
}

// The scalar store is itself the barrier: it stalls at issue on
// ~acc_no_st_pending | ~acc_no_ld_pending, so it cannot retire while Ara
// still has a vector memory op in flight.  fence alone would not do that.
static inline void mark(uintptr_t a, uint32_t v) {
  *(volatile uint32_t *)a = v;
  __asm__ volatile("fence" ::: "memory");
  llc_flush();
}

#define STEP(n)  mark(A_STEP, (n))
#define DONE(n)  do { mark(A_DONE, (n)); } while (0)

static void verdict(int id, const char *name, int ok, long got, long want) {
  seen++;
  if (!ok) fails |= (1u << id);
  mark(A_FAIL, fails);
  mark(A_SEEN, seen);
  printf_("  %2d %-14s %-4s  got %ld  want %ld\n",
          id, name, ok ? "ok" : "BAD", got, want);
}

static uint32_t buf[1024] __attribute__((aligned(64)));
static uint16_t hbuf[2048] __attribute__((aligned(64)));

int main(void) {
  printf_("\n===== vrare: masku / permute instruction sweep =====\n");
  printf_("marker map: STEP=0x%08lx DONE=0x%08lx FAIL=0x%08lx SEEN=0x%08lx\n",
          (unsigned long)A_STEP, (unsigned long)A_DONE,
          (unsigned long)A_FAIL, (unsigned long)A_SEEN);
  mark(A_STEP, 0); mark(A_DONE, 0); mark(A_FAIL, 0); mark(A_SEEN, 0);

  size_t vl;
  long a0;
  int ok;

  // ---------------------------------------------------------------- e32, m1
  __asm__ volatile("vsetvli %0, zero, e32, m1, ta, ma" : "=r"(vl) :: "memory");
  printf_("\n-- e32,m1  VLMAX=%lu --\n", (unsigned long)vl);

  STEP(1);                                              // vid.v
  __asm__ volatile("vid.v v8\n\t vse32.v v8, (%0)" :: "r"(buf) : "memory");
  DONE(1);
  ok = 1; for (size_t i = 0; i < vl; i++) if (buf[i] != i) { ok = 0; break; }
  verdict(1, "vid.v", ok, buf[vl > 3 ? 3 : 0], vl > 3 ? 3 : 0);

  STEP(2);                                              // vremu.vx
  { long m = 7;
    __asm__ volatile("vid.v v8\n\t vremu.vx v9, v8, %1\n\t vse32.v v9, (%0)"
                     :: "r"(buf), "r"(m) : "memory"); }
  DONE(2);
  ok = 1; for (size_t i = 0; i < vl; i++) if (buf[i] != i % 7) { ok = 0; break; }
  verdict(2, "vremu.vx", ok, buf[vl > 9 ? 9 : 0], vl > 9 ? 9 % 7 : 0);

  // v0 = mask with every even lane set
  __asm__ volatile("vid.v v8\n\t vand.vi v9, v8, 1\n\t vmseq.vi v0, v9, 0"
                   ::: "memory");

  STEP(3);                                              // vcpop.m -> SCALAR
  __asm__ volatile("vcpop.m %0, v0" : "=r"(a0) :: "memory");
  DONE(3);
  verdict(3, "vcpop.m", a0 == (long)((vl + 1) / 2), a0, (long)((vl + 1) / 2));

  STEP(4);                                              // vfirst.m -> SCALAR
  __asm__ volatile("vfirst.m %0, v0" : "=r"(a0) :: "memory");
  DONE(4);
  verdict(4, "vfirst.m", a0 == 0, a0, 0);

  STEP(5);                                              // vcompress.vm
  __asm__ volatile("vid.v v8\n\t vcompress.vm v12, v8, v0\n\t vse32.v v12, (%0)"
                   :: "r"(buf) : "memory");
  DONE(5);
  ok = 1; for (size_t i = 0; i < (vl + 1) / 2; i++)
            if (buf[i] != 2 * i) { ok = 0; break; }
  verdict(5, "vcompress.vm", ok, buf[1], 2);

  STEP(6);                                              // viota.m
  __asm__ volatile("viota.m v12, v0\n\t vse32.v v12, (%0)" :: "r"(buf) : "memory");
  DONE(6);
  ok = 1; { uint32_t acc = 0;
            for (size_t i = 0; i < vl; i++) {
              if (buf[i] != acc) { ok = 0; break; }
              if ((i & 1) == 0) acc++;
            } }
  verdict(6, "viota.m", ok, buf[4], 2);

  // v1 = mask with exactly one bit set, at index 3
  __asm__ volatile("vid.v v8\n\t vmseq.vi v1, v8, 3" ::: "memory");

  STEP(7);                                              // vmsbf.m
  __asm__ volatile("vmsbf.m v2, v1\n\t vcpop.m %0, v2" : "=r"(a0) :: "memory");
  DONE(7);
  verdict(7, "vmsbf.m", a0 == 3, a0, 3);

  STEP(8);                                              // vmsif.m
  __asm__ volatile("vmsif.m v2, v1\n\t vcpop.m %0, v2" : "=r"(a0) :: "memory");
  DONE(8);
  verdict(8, "vmsif.m", a0 == 4, a0, 4);

  STEP(9);                                              // vmsof.m
  __asm__ volatile("vmsof.m v2, v1\n\t vcpop.m %0, v2" : "=r"(a0) :: "memory");
  DONE(9);
  verdict(9, "vmsof.m", a0 == 1, a0, 1);

  STEP(10);                                             // vmv.s.x
  { long v = 0x5A5A;
    __asm__ volatile("vmv.s.x v16, %1\n\t vse32.v v16, (%0)"
                     :: "r"(buf), "r"(v) : "memory"); }
  DONE(10);
  verdict(10, "vmv.s.x", buf[0] == 0x5A5A, buf[0], 0x5A5A);

  STEP(11);                                             // vmv.x.s -> SCALAR
  __asm__ volatile("vmv.x.s %0, v16" : "=r"(a0) :: "memory");
  DONE(11);
  verdict(11, "vmv.x.s", a0 == 0x5A5A, a0, 0x5A5A);

  STEP(12);                                             // vrgather.vv
  { long last = (long)vl - 1;
    __asm__ volatile("vid.v v8\n\t vrsub.vx v9, v8, %1\n\t"
                     "vrgather.vv v12, v8, v9\n\t vse32.v v12, (%0)"
                     :: "r"(buf), "r"(last) : "memory"); }
  DONE(12);
  ok = 1; for (size_t i = 0; i < vl; i++)
            if (buf[i] != vl - 1 - i) { ok = 0; break; }
  verdict(12, "vrgather.vv", ok, buf[0], (long)vl - 1);

  STEP(13);                                             // vslideup.vx
  __asm__ volatile("vid.v v8\n\t vmv.v.i v12, 0\n\t vslideup.vx v12, v8, %1\n\t"
                   "vse32.v v12, (%0)" :: "r"(buf), "r"(2L) : "memory");
  DONE(13);
  verdict(13, "vslideup.vx", buf[5] == 3, buf[5], 3);

  STEP(14);                                             // vslidedown.vx
  __asm__ volatile("vid.v v8\n\t vslidedown.vx v12, v8, %1\n\t vse32.v v12, (%0)"
                   :: "r"(buf), "r"(2L) : "memory");
  DONE(14);
  verdict(14, "vslidedown.vx", buf[5] == 7, buf[5], 7);

  // ------------------------------------------- e16, m4: what fconv_cmp uses
  size_t vlh;
  __asm__ volatile("vsetvli %0, zero, e16, m4, ta, ma" : "=r"(vlh) :: "memory");
  printf_("\n-- e16,m4  VLMAX=%lu  (the config fconv_cmp's stream kernel runs at) --\n",
          (unsigned long)vlh);

  STEP(21);
  __asm__ volatile("vid.v v8\n\t vse16.v v8, (%0)" :: "r"(hbuf) : "memory");
  DONE(21);
  ok = 1; for (size_t i = 0; i < vlh; i++)
            if (hbuf[i] != (uint16_t)i) { ok = 0; break; }
  verdict(21, "vid.v m4", ok, hbuf[3], 3);

  STEP(22);
  { long m = 29;   // W=29 -- the prime width fconv_cmp actually rotates by
    __asm__ volatile("vid.v v8\n\t vremu.vx v12, v8, %1\n\t vse16.v v12, (%0)"
                     :: "r"(hbuf), "r"(m) : "memory"); }
  DONE(22);
  ok = 1; for (size_t i = 0; i < vlh; i++)
            if (hbuf[i] != (uint16_t)(i % 29)) { ok = 0; break; }
  verdict(22, "vremu.vx m4", ok, hbuf[30], 1);

  __asm__ volatile("vid.v v8\n\t vand.vi v12, v8, 1\n\t vmseq.vi v0, v12, 0"
                   ::: "memory");

  STEP(23);
  __asm__ volatile("vcpop.m %0, v0" : "=r"(a0) :: "memory");
  DONE(23);
  verdict(23, "vcpop.m m4", a0 == (long)((vlh + 1) / 2), a0, (long)((vlh + 1) / 2));

  STEP(25);
  __asm__ volatile("vid.v v8\n\t vcompress.vm v16, v8, v0\n\t vse16.v v16, (%0)"
                   :: "r"(hbuf) : "memory");
  DONE(25);
  ok = 1; for (size_t i = 0; i < (vlh + 1) / 2; i++)
            if (hbuf[i] != (uint16_t)(2 * i)) { ok = 0; break; }
  verdict(25, "vcompress m4", ok, hbuf[1], 2);

  STEP(99);
  printf_("\n===== vrare done: %lu tests ran, fail mask 0x%08lx =====\n",
          (unsigned long)seen, (unsigned long)fails);
  mark(A_DONE, 99);
  return (int)fails;
}
