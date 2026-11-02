#pragma once
// K5-VGG: this file lives in vggnet/kernel/, so a bare #include "printf.h"
// from any kernel/*.c file resolves HERE first (C searches the including
// file's own directory before -I paths) instead of common/printf.h (found
// via -I for main.c, which lives one directory up in vggnet/ and has no
// local printf.h to shadow it with).
//
// That meant every kernel/*.c file -- train.c's per-epoch accuracy/loss
// prints included -- silently got this header's ARA_LINUX-only "map
// printf_ to real printf" shim even when building for FPGA or SPIKE, which
// have no OS backing real printf's newlib stdio machinery (malloc via
// _sbrk, isatty, fstat, write -- see common/util.c and common/crt0.S).
// Confirmed via `llvm-nm -u` on the built .o files: every kernel/*.c
// object referenced undefined "printf" (real newlib), while main.c.o
// referenced only "printf_" (the safe one) -- not guessed.
//
// Only the native/host build (ARA_LINUX) actually has a real OS under it
// and should get real printf. FPGA and SPIKE must get the exact same
// declaration common/printf.h gives every other file.
#if defined(ARA_LINUX)
#include <stdio.h>
#include <stdlib.h>
#define printf_(...) printf(__VA_ARGS__)
#else
#define printf printf_
int printf_(const char *format, ...);
#endif
