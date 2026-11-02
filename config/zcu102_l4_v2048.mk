# Config matching the ZCU102 bitstream actually on the board:
#   cheshire_top_xilinx_L4_V2048_LLC1M_FPUH_..._NaNboxfix_20260830.bit
#
# NOTE: runtime.mk extracts the geometry with
#   grep vlen <config>.mk | cut -d" " -f3
# so exactly ONE line in this file may contain the lowercase token, and the
# same applies to the lane count. Comments below therefore use uppercase.
#
# The stock configs do NOT describe that hardware: config/4_lanes.mk has been
# locally edited to 2 lanes / VLEN 1024, and default.mk is the same. Every app
# build so far has thus been compiled and simulated for a 2-lane VLEN=1024
# machine. The kernels use runtime `vsetvli` throughout and never reference
# VLEN at compile time, so the binaries still RUN correctly on the 4-lane part.
#
# It matters for the Spike REFERENCE. VLMAX at e16/m8 is VLEN*8/16, i.e. 1024
# here versus 512 for VLEN=1024, and batchnorm reduces spatial_size = 1024
# elements per channel. The real hardware therefore performs a SINGLE
# full-VLMAX vfredusum.vs where a VLEN=1024 Spike run performs two chained
# ones -- so the reference never exercised the shape bn1 uses on the board.
#
# Use with: make spike-run-<app> ARA_CONFIGURATION=zcu102_l4_v2048

# Number of vector lanes
nr_lanes ?= 4

# Length of each vector register (in bits)
# Constraints: must be > 128
vlen ?= 2048
