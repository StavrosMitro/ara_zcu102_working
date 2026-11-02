# Copyright 2020 ETH Zurich and University of Bologna.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Samuel Riedel, ETH Zurich
#         Matheus Cavalcante, ETH Zurich
SHELL = /usr/bin/env bash

ROOT_DIR := $(patsubst %/,%, $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Use the nearest ancestor that contains Ara's shared configuration files. This
# matters when the apps checkout itself is a Git repository while the real Ara
# repo lives one directory above it.
ARA_DIR := $(shell \
	if [ -f "$(CURDIR)/../config/default.mk" ]; then \
		cd "$(CURDIR)/.." && pwd; \
	elif [ -f "$(CURDIR)/config/default.mk" ]; then \
		cd "$(CURDIR)" && pwd; \
	else \
		git -C "$(CURDIR)" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$$ARA_DIR"; \
	fi)

# Choose Ara's configuration
ifndef config
	ifdef ARA_CONFIGURATION
		config := $(ARA_CONFIGURATION)
	else
		config := default
	endif
endif

# Include configuration
include $(ARA_DIR)/config/$(config).mk

INSTALL_DIR             ?= $(ARA_DIR)/install
GCC_INSTALL_DIR         ?= $(INSTALL_DIR)/riscv-gcc
LLVM_INSTALL_DIR        ?= $(INSTALL_DIR)/riscv-llvm
ISA_SIM_INSTALL_DIR     ?= $(INSTALL_DIR)/riscv-isa-sim
ISA_SIM_MOD_INSTALL_DIR ?= $(INSTALL_DIR)/riscv-isa-sim-mod

RISCV_XLEN    ?= 64
RISCV_ARCH    ?= rv$(RISCV_XLEN)gcv
RISCV_ABI     ?= lp64d
RISCV_TARGET  ?= riscv$(RISCV_XLEN)-unknown-elf

# Use LLVM for bare-metal RVV and GCC for Linux RVV
ifeq ($(LINUX),1)
RISCV_PREFIX  ?= $(ARA_DIR)/cheshire/sw/cva6-sdk/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-
RISCV_CC      ?= $(RISCV_PREFIX)gcc
RISCV_CXX     ?= $(RISCV_PREFIX)g++
RISCV_OBJDUMP ?= $(RISCV_PREFIX)objdump
RISCV_OBJCOPY ?= $(RISCV_PREFIX)objcopy
RISCV_AS      ?= $(RISCV_PREFIX)as
RISCV_AR      ?= $(RISCV_PREFIX)ar
RISCV_LD      ?= $(RISCV_PREFIX)ld.lld
RISCV_STRIP   ?= $(RISCV_PREFIX)strip
else
RISCV_PREFIX  ?= $(LLVM_INSTALL_DIR)/bin/
RISCV_CC      ?= $(RISCV_PREFIX)clang
RISCV_CXX     ?= $(RISCV_PREFIX)clang++
RISCV_OBJDUMP ?= $(RISCV_PREFIX)llvm-objdump
RISCV_OBJCOPY ?= $(RISCV_PREFIX)llvm-objcopy
RISCV_AS      ?= $(RISCV_PREFIX)llvm-as
RISCV_AR      ?= $(RISCV_PREFIX)llvm-ar
RISCV_LD      ?= $(RISCV_PREFIX)ld.lld
RISCV_STRIP   ?= $(RISCV_PREFIX)llvm-strip
endif

# Use gcc to compile scalar riscv-tests
RISCV_CC_GCC  ?= $(GCC_INSTALL_DIR)/bin/$(RISCV_TARGET)-gcc

# Benchmark with spike
spike_env_dir ?= $(ARA_DIR)/apps/riscv-tests
SPIKE_INC     ?= -I$(spike_env_dir)/env -I$(spike_env_dir)/benchmarks/common
SPIKE_CCFLAGS ?= -DPREALLOCATE=1 -DSPIKE=1 $(SPIKE_INC)
SPIKE_LDFLAGS ?= -nostdlib -T$(spike_env_dir)/benchmarks/common/test.ld
FPGA_CCFLAGS  ?= -DFPGA=1
RISCV_SIM     ?= $(ISA_SIM_INSTALL_DIR)/bin/spike
RISCV_SIM_MOD ?= $(ISA_SIM_MOD_INSTALL_DIR)/bin/spike
# VLEN should be lower or equal than 4096 because of spike restrictions
vlen_spike := $(shell vlen=$$(grep vlen $(ARA_DIR)/config/$(config).mk | cut -d" " -f3) && echo "$$(( $$vlen < 4096 ? $$vlen : 4096 ))")
RISCV_SIM_OPT ?= --isa=rv64gcv_zfh --varch="vlen:$(vlen_spike),elen:64"
RISCV_SIM_MOD_OPT ?= --isa=rv64gcv_zfh --varch="vlen:$(vlen_spike),elen:64" -d

# Python
PYTHON ?= python3

# Defines
ENV_DEFINES ?=
ifeq ($(LINUX),1)
ENV_DEFINES += -DARA_LINUX=1
endif
ifeq ($(vcd_dump),1)
ENV_DEFINES += -DVCD_DUMP=1
endif
MAKE_DEFINES = -DNR_LANES=$(nr_lanes) -DVLEN=$(vlen)
DEFINES += $(ENV_DEFINES) $(MAKE_DEFINES)

# Common flags
RISCV_WARNINGS += -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument # -Werror

# LLVM Flags
LLVM_SYSROOT   ?= $(LLVM_INSTALL_DIR)/riscv64-unknown-elf
LLVM_FLAGS     ?= -march=rv64gcv_zfh_zvfh -mabi=$(RISCV_ABI) -mno-relax -fuse-ld=lld --sysroot=$(LLVM_SYSROOT)
AUTOVECTORIZE  ?= 0
LLVM_V_FLAGS   ?= $(if $(filter 1,$(AUTOVECTORIZE)),-mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float,-fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float)
RISCV_FLAGS    ?= $(LLVM_FLAGS) $(LLVM_V_FLAGS) -mcmodel=medany -I$(CURDIR)/common -O3 -g -ffast-math -fno-common -fno-builtin-printf $(DEFINES) $(RISCV_WARNINGS)
ifeq ($(LINUX),1)
RISCV_CCFLAGS  ?= -march=rv64gcv -mabi=$(RISCV_ABI) -I$(CURDIR)/common -O2 $(DEFINES)
RISCV_LDFLAGS  ?= -lm -lstdc++
RISCV_CXXFLAGS ?= -march=rv64gcv -mabi=$(RISCV_ABI) -I$(CURDIR)/common -O2 $(DEFINES)
else
RISCV_CCFLAGS  ?= $(RISCV_FLAGS) -ffunction-sections -fdata-sections -std=gnu99
RISCV_LDFLAGS  ?= -static -nostartfiles -lm -Wl,--gc-sections -T$(CURDIR)/common/link.ld
endif
RISCV_CCFLAGS_SPIKE  ?= $(RISCV_FLAGS) $(SPIKE_CCFLAGS) -ffunction-sections -fdata-sections -std=gnu99
RISCV_CCFLAGS_FPGA   ?= $(RISCV_FLAGS) $(FPGA_CCFLAGS) -ffunction-sections -fdata-sections -std=gnu99
# softfloat branch: no hardware D on either CVA6 (scalar) or Ara (vector).
# F32 stays (RVF=1/FPUSupportHalfSingle on the cheshire_soc fp16_only
# branch) -- confirmed empirically that LLVM's RISC-V ISA dependency graph
# makes Zvfh (vector half, needed for Ara's vectorized fp16 code) require
# Zve32f, which itself requires scalar F; clang kept emitting real
# fadd.s/fmv.w.x for plain scalar float even with an explicit
# -target-feature -f override placed after the march-derived +f. D has no
# such dependency and drops cleanly.
#
# IMPORTANT: -march does NOT behave as last-occurrence-wins the way -mabi
# does (confirmed empirically) -- appending a second -march= to
# RISCV_CCFLAGS_FPGA's (which already has one) merges/unions target
# features rather than replacing them, silently leaving +d enabled despite
# the override. So this is built as an independent flag chain from
# LLVM_FLAGS down, never inheriting RISCV_CCFLAGS_FPGA's own -march, rather
# than appending to it.
#
# Only used by vggnet16/conv_layer16only/fc_layer16only via the .fpga16
# rules/template below -- every other app's normal .fpga build is untouched.
#
# -resource-dir redirects clang to an isolated compiler-rt builtins archive
# (libclang_rt.builtins-riscv64.a) built for this same march/abi -- the
# default one (found via the normal resource-dir) contains real hardware
# fadd.d/fmul.d/etc for D, and critically ld.lld does NOT reject linking it
# in the way it rejects a mismatched libc/libm (confirmed empirically: it
# links clean but the resulting binary's .riscv.attributes shows d present
# and objdump shows real fadd.d-class instructions). Its include/ symlinks
# back to the real resource dir (same intrinsic headers, e.g.
# riscv_vector.h, used for compiling too) so overriding this for both
# compile and link steps is safe.
RISCV_ARCH_FPGA16 ?= rv64imac_zve64f_zvfh
RISCV_ABI_FPGA16   ?= lp64
RISCV_RESOURCE_DIR_FPGA16 ?= $(INSTALL_DIR)/riscv-llvm-fp16only-rt/resource-dir
# -resource-dir (above) only redirects compiler-rt's builtins archive.
# libc.a/libm.a/libgloss.a are resolved separately, via -lc/-lm/-lgloss
# name-search against a -L path the driver adds on its own -- and that
# auto-added -L always points at the flat, non-multilib sysroot lib/
# directory regardless of -march/-mabi (confirmed empirically: this
# toolchain's driver isn't actually multilib-aware for library search,
# despite clang --print-multi-directory reporting differing per-target
# subdirectory names). That flat directory is still the original lp64d/
# full-FPU newlib build. This explicit -L, placed first, makes -lc/-lm/
# -lgloss resolve to the isolated lp64 build (rv64imac/lp64/, produced by
# rebuilding newlib with CC_FOR_TARGET set to this march/abi) before ever
# reaching the flat fallback.
LLVM_FLAGS_FPGA16  ?= -march=$(RISCV_ARCH_FPGA16) -mabi=$(RISCV_ABI_FPGA16) -mno-relax -fuse-ld=lld --sysroot=$(LLVM_SYSROOT) -resource-dir=$(RISCV_RESOURCE_DIR_FPGA16) -L$(LLVM_SYSROOT)/lib/rv64imac/lp64
RISCV_FLAGS_FPGA16 ?= $(LLVM_FLAGS_FPGA16) $(LLVM_V_FLAGS) -mcmodel=medany -I$(CURDIR)/common -O3 -g -ffast-math -fno-common -fno-builtin-printf $(DEFINES) $(RISCV_WARNINGS)
RISCV_CCFLAGS_FPGA16 ?= $(RISCV_FLAGS_FPGA16) $(FPGA_CCFLAGS) -ffunction-sections -fdata-sections -std=gnu99
# RISCV_LDFLAGS_FPGA's "-lm" is a name search (-l), resolved only against
# -L paths the linker has already seen at that point on the command line --
# and clang's driver places user -L flags (from RISCV_CCFLAGS_FPGA16 above)
# AFTER "-lm" in the actual ld.lld invocation (confirmed via -v: "... -lm
# --gc-sections -L<isolated> -L<flat> ... --start-group -lc -lgloss
# --end-group ..."). So -lm silently fell back to the sysroot's default
# (flat, hard-float) libm.a regardless of the -L override, while -lc/
# -lgloss (positioned after the -L flags, inside --start-group) correctly
# picked up the isolated build. An explicit archive path sidesteps -L
# search entirely, so it isn't sensitive to this ordering.
RISCV_LDFLAGS_FPGA16 ?= -static -nostartfiles $(LLVM_SYSROOT)/lib/rv64imac/lp64/libm.a -Wl,--gc-sections -T$(CURDIR)/common/link.fpga.ld
RISCV_CXXFLAGS ?= $(RISCV_FLAGS) -ffunction-sections -fdata-sections
RISCV_LDFLAGS_SPIKE  ?= -static -nostartfiles -lm $(SPIKE_LDFLAGS) -Wl,--gc-sections
RISCV_LDFLAGS_FPGA   ?= -static -nostartfiles -lm -Wl,--gc-sections -T$(CURDIR)/common/link.fpga.ld

# GCC Flags
RISCV_FLAGS_GCC    ?= -mcmodel=medany -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI) -I$(CURDIR)/common -static -O3 -ffast-math -fno-common -fno-builtin-printf $(DEFINES) $(RISCV_WARNINGS)
RISCV_CCFLAGS_GCC  ?= $(RISCV_FLAGS_GCC) -std=gnu99
RISCV_CXXFLAGS_GCC ?= $(RISCV_FLAGS_GCC)
RISCV_LDFLAGS_GCC  ?= -static -nostartfiles -lm -lgcc $(RISCV_FLAGS_GCC) -std=gnu99 -T$(CURDIR)/common/link.ld

ifeq ($(COMPILER),gcc)
	RISCV_OBJDUMP_FLAGS ?=
else
ifneq ($(LINUX),1)
	RISCV_OBJDUMP_FLAGS ?= --mattr=v
endif
endif

# Compile two different versions of the runtime, since we cannot link code compiled with two different toolchains
RUNTIME_GCC   ?= common/crt0-gcc.S.o common/printf-gcc.c.o common/string-gcc.c.o common/serial-gcc.c.o common/util-gcc.c.o
ifeq ($(LINUX),1)
RUNTIME_LLVM  ?= common/util-llvm.c.o
else
RUNTIME_LLVM  ?= common/crt0-llvm.S.o common/printf-llvm.c.o common/string-llvm.c.o common/serial-llvm.c.o common/util-llvm.c.o
endif
RUNTIME_SPIKE ?= $(spike_env_dir)/benchmarks/common/crt.S.o.spike $(spike_env_dir)/benchmarks/common/syscalls.c.o.spike common/util.c.o.spike common/printf.c.o.spike common/serial.c.o.spike
RUNTIME_FPGA  ?= common/crt0.S.o.fpga common/printf.c.o.fpga common/string.c.o.fpga common/serial.c.o.fpga common/util.c.o.fpga
# softfloat branch: private runtime objects for the FPUSupportHalf-only
# apps, built via RISCV_CCFLAGS_FPGA16 (see above) instead of sharing the
# normal .fpga objects -- printf.c's %f handling and util.c's
# similarity_check() do real hardware double arithmetic, so they can't be
# compiled once and shared between a full-FPU app and a no-FPU one: Make
# only tracks file mtimes, not which flags an existing object was built
# with, so a shared path would silently reuse whichever variant happened to
# be built first.
RUNTIME_FPGA16 ?= common/crt0.S.o.fpga16 common/printf.c.o.fpga16 common/string.c.o.fpga16 common/serial.c.o.fpga16 common/util.c.o.fpga16
.INTERMEDIATE: $(RUNTIME_GCC) $(RUNTIME_LLVM)

%-gcc.S.o: %.S
	$(RISCV_CC_GCC) $(RISCV_CCFLAGS_GCC) -c $< -o $@

%-gcc.c.o: %.c
	$(RISCV_CC_GCC) $(RISCV_CCFLAGS_GCC) -c $< -o $@

%-llvm.S.o: %.S
	$(RISCV_CC) $(RISCV_CCFLAGS) -c $< -o $@

%-llvm.c.o: %.c
	$(RISCV_CC) $(RISCV_CCFLAGS) -c $< -o $@

%.S.o: %.S
	$(RISCV_CC) $(RISCV_CCFLAGS) -c $< -o $@

%.c.o: %.c
	$(RISCV_CC) $(RISCV_CCFLAGS) -c $< -o $@

%.S.o.spike: %.S patch-spike-crt0
	$(RISCV_CC) $(RISCV_CCFLAGS_SPIKE) -c $< -o $@

%.c.o.spike: %.c
	$(RISCV_CC) $(RISCV_CCFLAGS_SPIKE) -c $< -o $@

%.S.o.fpga: %.S
	$(RISCV_CC) $(RISCV_CCFLAGS_FPGA) -c $< -o $@

%.c.o.fpga: %.c
	$(RISCV_CC) $(RISCV_CCFLAGS_FPGA) -c $< -o $@

%.S.o.fpga16: %.S
	$(RISCV_CC) $(RISCV_CCFLAGS_FPGA16) -c $< -o $@

%.c.o.fpga16: %.c
	$(RISCV_CC) $(RISCV_CCFLAGS_FPGA16) -c $< -o $@

%.cpp.o: %.cpp
	$(RISCV_CXX) $(RISCV_CXXFLAGS) -c $< -o $@

%.ld: %.ld.c
	$(RISCV_CC) -P -E $(DEFINES) $< -o $@
