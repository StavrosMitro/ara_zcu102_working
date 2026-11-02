# Copyright 2024 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Author: Moritz Imfeld   <moimfeld@student.ethz.ch>
# Author: Matteo Perotti  <mperotti@ethz.ch>
# Author: Mojtaba Rostami <m.rostami1989@gmail.com>

# Chshire root reposiotry
MAKEFILE_DIR            := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ARA_ROOT                := $(MAKEFILE_DIR)/..
BACKREF_CHS_ROOT        := $(realpath $(ARA_ROOT)/../..)
BACKREF_CHS_XIL_SCRIPTS := $(BACKREF_CHS_ROOT)/target/xilinx/scripts
BACKREF_CHS_SIM_SCRIPTS := $(BACKREF_CHS_ROOT)/target/sim/vsim/

# Tools
VIVADO ?= 'vitis-2021.1 vivado'

# Set up Bender targets and defines
# default configuration for Cheshire + Ara is 2_lanes
ARA_CONFIGURATION         ?= 2_lanes
include $(ARA_ROOT)/config/$(ARA_CONFIGURATION).mk

BOARD                     ?= zcu102
VLOG_ARGS                 ?= -suppress 2583 -suppress 13314

# Cheshire LLC data-array size in KiB. This RAM is shared between cache ways
# and SPM ways (the split is a runtime bitmask in cfg_spm at AmLlc), so this
# sets the TOTAL. cheshire_top_xilinx.sv derives LlcNumLines from it so the
# byte size stays fixed as AxiDataWidth changes with the lane count.
llc                       ?= 1024

# RISC-V atomics (AMO + LR/SC) shims on the reg / LLC / dbg / DMA-conf AXI slave
# ports. atomics=0 replaces all four with wires: ~13.2k LUTs and removes the
# LlcMaxReadTxns=16 outstanding-read throttle on the SoC->LLC path. Requires
# CVA6ConfigAExtEn=0 (see cv64a6_imafdcv_sv39_config_pkg.sv) so a stray atomic
# traps rather than silently doing the wrong thing. Set atomics=1 for Linux.
# Default 0 to stay consistent with CVA6ConfigAExtEn=0, which is hardcoded in
# cva6_patched. Building atomics=1 while CVA6 has A disabled is harmless (unused
# hardware); the reverse would be unsafe, and cannot happen from this default.
atomics                   ?= 0

# RISC-V debug module + JTAG TAP. dbg=0 removes it entirely (~10.8k LUTs) and
# terminates the AxiOut.dbg crossbar port with an axi_err_slv. Only valid when
# the board defines no `USE_JTAG pins -- which zcu102 does not. dbg=1 restores.
dbg                       ?= 0
DBG_DEFINE                := $(if $(filter 0,$(dbg)),--define NO_DBG,)
ATOMICS_DEFINE            := $(if $(filter 0,$(atomics)),--define NO_ATOMICS,)

# Ara VECTOR FPU precision. 16 = FP16 only (default, smallest), 32 = FP32 only,
# 1632 = both. Scalar FP is separate (CVA6ConfigRVF/RVD in cva6_patched); note
# that RVD sets FLEN, and ara_dispatcher's NaN-box check derives the box width
# from CVA6Cfg -- a hardcoded 64 there is what made every .vf scalar a NaN.
fpu                       ?= 16
FPU_DEFINE                := $(if $(filter 1632,$(fpu)),--define ARA_FPU_1632,\
                             $(if $(filter 32,$(fpu)),--define ARA_FPU_32,))

# ILA probe groups -- switchable, so a probed and an unprobed build differ in
# nothing else. Empty (the default) = NO mark_debug nets, so common.tcl's
# insert_ilas adds no ILA at all: zero area, zero timing impact.
#   probes=mem        memory path (round 4): AXI points C1/A1/X/L2/L3 in
#                     cheshire_soc.sv + the CVA6 L1-D load port in cva6.sv
#   probes=vec        vector instruction lifecycle: CVA6<->Ara handshake in
#                     cheshire_soc.sv + Ara sequencer issue/done per unit in
#                     ara_patched/hardware/src/ara_sequencer.sv. This is what
#                     times a slide: a slide never touches memory.
#   probes=ddr        point H (dbg_h_*): input of the DRAM CDC, i.e. between
#                     the LLC master port (L3, in mem) and the PS. With mem,
#                     L3->H = AXI converters, H = CDC + HP0 + PS + DRAM.
#                     target/xilinx/src/dram_wrapper_xilinx.sv
#   probes=mem,vec    any combination (comma or space separated)
# NOTE: until 2026-10-26 the mem probes were unconditional, so EVERY bitstream
# built before then carries an ILA (~8k LUT). Pass probes=mem to reproduce that.
comma                     := ,
probes                    ?=
PROBE_LIST                := $(subst $(comma), ,$(probes))
PROBE_BAD                 := $(filter-out mem vec ddr,$(PROBE_LIST))
$(if $(PROBE_BAD),$(error unknown probes= group(s): '$(PROBE_BAD)'. Valid: mem vec ddr))
PROBE_DEFINE              := $(if $(filter mem,$(PROBE_LIST)),--define PROBE_MEM,) \
                             $(if $(filter vec,$(PROBE_LIST)),--define PROBE_VEC,) \
                             $(if $(filter ddr,$(PROBE_LIST)),--define PROBE_DDR,)

# Clock knobs (ZCU102) -- OFF by default: with neither set NO define is passed and
# cheshire_top_xilinx.sv compiles the original clkwiz unchanged (50 MHz soc_clk,
# HP0 on soc_clk). Setting either one switches to an MMCM in that file:
#   soc_mhz=<f>   SoC clock (CVA6, Ara, LLC, ILA). 50 = today. Timing is NOT
#                 guaranteed above ~55-60 MHz, see TIMING_CRITICAL_PATH_20261027.md
#   hp0_mhz=<f>   PS HP0 port + DRAM side of the axi_cdc. Unset = same as soc_clk.
# <f> must divide the 1200 MHz VCO: 10 12 15 16 20 24 25 30 40 48 50 60 75 80 100
# 120 150 200 240 300 (HP0 max 300). xdc/ILA/RTC follow automatically.
soc_mhz                   ?= 50
hp0_mhz                   ?=
CLK_VALID_MHZ             := 10 12 15 16 20 24 25 30 40 48 50 60 75 80 100 120 150 200 240 300
$(if $(filter-out $(CLK_VALID_MHZ),$(soc_mhz)),$(error soc_mhz=$(soc_mhz) is not 1200/N. Valid: $(CLK_VALID_MHZ)))
$(if $(hp0_mhz),$(if $(filter-out $(CLK_VALID_MHZ),$(hp0_mhz)),$(error hp0_mhz=$(hp0_mhz) is not 1200/N. Valid: $(CLK_VALID_MHZ))))
CLK_KNOB                  := $(if $(or $(filter-out 50,$(soc_mhz)),$(hp0_mhz)),1,)
CLK_DEFINE                := $(if $(CLK_KNOB),--define SOC_MHZ=$(soc_mhz),) \
                             $(if $(hp0_mhz),--define HP0_MHZ=$(hp0_mhz),)

COMMON_CUSTOM_TARGETS     := -t cv64a6_imafdcv_sv39 -t cva6 --define ARA --define NR_LANES=$(nr_lanes) --define VLEN=$(vlen) --define LLC_KIB=$(llc) $(ATOMICS_DEFINE) $(DBG_DEFINE) $(FPU_DEFINE) $(PROBE_DEFINE)
CUSTOM_SIM_BENDER_TARGETS := $(COMMON_CUSTOM_TARGETS) -t sim -t test -t rtl --vlog-arg="$(VLOG_ARGS)"
# exclude_first_pass_decoder: prevents CVA6 stub from being included (replaced by Ara real decoder)
CUSTOM_XIL_BENDER_TARGETS := $(COMMON_CUSTOM_TARGETS) -t fpga -t $(BOARD) -t exclude_first_pass_decoder $(CLK_DEFINE)

# Define XILINX FPGA URL and PATH for programming
CHS_XILINX_HWS_URL  ?=
CHS_XILINX_HWS_PATH ?=

.PHONY: ara-chs-xilinx ara-chs-flash ara-chs-xilinx-program update_xilinx_src update_vsim_src clean

ara-chs-xilinx: update_xilinx_src
	make -C $(BACKREF_CHS_ROOT) chs-xilinx-$(BOARD)

ara-chs-image:
	make -C $(BACKREF_CHS_ROOT) $(BACKREF_CHS_ROOT)/sw/boot/linux.$(BOARD).gpt.bin -B

ara-chs-xilinx-flash:
	make -C $(BACKREF_CHS_ROOT) chs-xilinx-flash-$(BOARD) VIVADO="$(VIVADO)" CHS_XILINX_HWS_URL="$(CHS_XILINX_HWS_URL)" CHS_XILINX_HWS_PATH_$(BOARD)="$(CHS_XILINX_HWS_PATH)"

ara-chs-xilinx-program:
	make -C $(BACKREF_CHS_ROOT) chs-xilinx-program-$(BOARD) VIVADO="$(VIVADO)" CHS_XILINX_HWS_URL="$(CHS_XILINX_HWS_URL)" CHS_XILINX_HWS_PATH_$(BOARD)="$(CHS_XILINX_HWS_PATH)"

update_xilinx_src:
	cd $(BACKREF_CHS_ROOT) && \
	bender script vivado $(CUSTOM_XIL_BENDER_TARGETS) > $(BACKREF_CHS_XIL_SCRIPTS)/add_sources.$(BOARD).tcl

update_vsim_src:
	cd $(BACKREF_CHS_ROOT) && \
	bender script vsim $(CUSTOM_SIM_BENDER_TARGETS) > $(BACKREF_CHS_SIM_SCRIPTS)/compile.cheshire_soc.tcl
	echo 'vlog "$(realpath $(BACKREF_CHS_ROOT))/target/sim/src/elfloader.cpp" -ccflags "-std=c++11"' >> $(BACKREF_CHS_SIM_SCRIPTS)/compile.cheshire_soc.tcl

clean:
	rm $(BACKREF_CHS_XIL_SCRIPTS)/add_sources.$(BOARD).tcl
	rm $(MAKEFILE_DIR)/add_sources.$(BOARD).tcl
