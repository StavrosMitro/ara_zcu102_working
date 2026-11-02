#!/bin/bash
# Build every sldprobe variant: SLD_CASE {1 load, 2 slide} x SLD_LMUL {1,2,4,8}.
# One binary runs on every bitstream (vl = VLMAX is read at runtime), so this
# does NOT depend on lanes or VLEN.
#
# The objects are deleted before EVERY variant: .o.fpga16 does not track
# define-only changes, and that once shipped binaries silently missing their
# probes (memory: apps-fpga16-stale-defines).
#   ./build_variants.sh            # e16 (fpu=16 bitstreams)
#   SEW=32 ./build_variants.sh     # e32 (fpu=32 bitstreams)
set -eu
cd "$(dirname "$0")/.."
SEW=${SEW:-16}
OUT=bin
for c in 1 2; do
  for m in 1 2 4 8; do
    rm -f sldprobe/*.o.fpga16 bin/sldprobe.fpga16 bin/sldprobe.fpga16.bin
    ARA_DIR="$(cd .. && pwd)" make -s bin/sldprobe.fpga16 ARA_CONFIGURATION=zcu102_l4_v2048 \
        ENV_DEFINES="-DSLD_CASE=$c -DSLD_LMUL=$m -DSLD_SEW=$SEW" >/dev/null
    tag=$([ $c = 1 ] && echo ld || echo sl)
    name=sldprobe_${tag}_e${SEW}_m${m}.fpga16.bin
    cp bin/sldprobe.fpga16.bin $OUT/$name
    cp bin/sldprobe.fpga16.dump $OUT/${name%.bin}.dump
    printf "%-34s %s\n" "$name" "$(md5sum < $OUT/$name | cut -c1-32)"
  done
done
