// Copyright 2022 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Matteo Perotti <mperotti@iis.ee.ethz.ch>
//
// Utility functions for Ara software environment

#include "util.h"

int *__dummy__errno__ptr__;

unsigned long int timer;

// Floating-point similarity check with threshold
int similarity_check(double a, double b, double threshold) {
  double diff = a - b;
  if (FABS(diff) > threshold)
    return 0;
  else
    return 1;
}
int similarity_check_32b(float a, float b, float threshold) {
  float diff = a - b;
  if (FABS(diff) > threshold)
    return 0;
  else
    return 1;
}

// Dummy declaration for libm exp
int *__errno(void) { return __dummy__errno__ptr__; }

#ifdef FPGA
// K5-VGG: newlib ships every syscall stub (_exit, _fstat, _isatty, _write,
// _read, _close, _lseek, ...) as a generic RISC-V Linux/semihosting `ecall`,
// expecting an OS or debug agent to service it. Real hardware has neither:
// each one traps into crt0.S's trap_vector with mcause=11 (Environment call
// from M-mode) the first time it's actually reached, with no printed error
// and no useful exit code (just "11", the mcause, not anything meaningful).
//
// This is not specific to apps that call exit()/fopen() directly. Found via
// vggnet, which does neither on this path -- the trap fired from inside
// _fstat, reached through _isatty_r -> __smakebuf_r -> __swsetup_r ->
// _vfprintf_r, i.e. newlib's OWN internal stdio-buffer setup, triggered
// the first time ANYTHING touches a real FILE* stream (stb_image.h's
// assert()-adjacent code path, in that case) -- confirmed by reading the
// trap handler's saved mepc (0x40006eb4, inside _fstat) directly out of
// DDR while the core sat halted in _fail's loop, not guessed. So this is a
// real gap in the platform, not a one-off in application code, and gets
// fixed at that level: a minimal, standard bare-metal syscalls.c, the same
// shape used by essentially every newlib port that has no real OS under it.
//
// Not needed for SPIKE: the ISA simulator's own HTIF intercepts the ecall
// and services each syscall correctly on its own (confirmed by vggnet's own
// SPIKE-only handle_trap()/tohost path in main.c, a separate mechanism that
// already assumes this). These are strong symbols, so they override
// libnosys's weak ones at link time for FPGA only; SPIKE is unaffected.
#include <sys/stat.h>
#include <sys/unistd.h>

extern void _fail(int status) __attribute__((noreturn));
extern void _putchar(char character);

void _exit(int status) __attribute__((noreturn));
void _exit(int status) { _fail(status); }

// No real terminal on bare-metal FPGA.
int _isatty(int file) { (void)file; return 0; }

// Every fd is a character device -- enough for newlib's stdio buffer-mode
// probe (__smakebuf_r checks S_ISCHR via _fstat) to pick unbuffered/line
// mode and move on without needing a real filesystem underneath it.
int _fstat(int file, struct stat *st) {
  (void)file;
  st->st_mode = S_IFCHR;
  return 0;
}

// Route real newlib stdio (printf/fprintf/puts/...) through the same
// DDR-console _putchar() the custom printf_ already uses -- see serial.c.
// So this doesn't just stop the crash, it makes real printf()/fprintf()
// actually work, the same as printf_ already does.
_READ_WRITE_RETURN_TYPE _write(int file, const void *buf, size_t nbyte) {
  (void)file;
  const char *p = (const char *)buf;
  for (size_t i = 0; i < nbyte; i++) _putchar(p[i]);
  return (_READ_WRITE_RETURN_TYPE)nbyte;
}

// No input source on bare-metal FPGA.
_READ_WRITE_RETURN_TYPE _read(int file, void *buf, size_t nbyte) {
  (void)file; (void)buf; (void)nbyte;
  return 0;
}

int _close(int file) { (void)file; return 0; }

_off_t _lseek(int file, _off_t offset, int whence) {
  (void)file; (void)offset; (void)whence;
  return 0;
}

// K5-VGG round 2: fixing isatty/fstat/write/etc. wasn't the whole gap --
// __smakebuf_r (see the block comment above) also mallocs the stdio buffer
// it's setting up, and malloc needs _sbrk. Found the same way as before:
// mepc after the first fix was 0x40006ea8, inside _sbrk (a7=0xd6, the brk
// syscall number) -- not guessed.
//
// l2_alloc_base is defined in common/arch.link.fpga.ld, right after .bss/
// .l2, named for exactly this: the heap start, growing up toward _stack_top
// (0x60000000) where the stack grows down from. No collision check against
// the live stack pointer -- every other FPGA app this session has used only
// a few hundred bytes of stack at most, and the gap here is on the order of
// hundreds of MB, so this matches the precision the rest of this bring-up
// has used rather than adding an unmeasured guess at how close is too close.
extern char l2_alloc_base;
static char *heap_end = 0;

void *_sbrk(ptrdiff_t incr) {
  char *prev_heap_end;
  if (heap_end == 0) heap_end = &l2_alloc_base;
  prev_heap_end = heap_end;
  heap_end += incr;
  return (void *)prev_heap_end;
}
#endif
