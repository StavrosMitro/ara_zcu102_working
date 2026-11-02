#!/bin/bash
# ============================================================================
# Architectural sweep builder — every bitstream and every report is archived
# under a UNIQUE name, on the server, before anything can overwrite it.
#
#   12 bitstreams = {2 lanes x VLEN 256/1024/4096, 4 lanes x VLEN 512/2048/8192}
#                   x {LLC 1 MiB, LLC 2 MiB}
#
#   name           lanes  AXI   VLEN   VLEN/lane   LLC
#   L2_V256_1M       2    64b    256     128       1 MiB
#   L2_V1024_1M      2    64b   1024     512       1 MiB
#   L2_V4096_1M      2    64b   4096    2048       1 MiB
#   L4_V512_1M       4    128b   512     128       1 MiB
#   L4_V2048_1M      4    128b  2048     512       1 MiB
#   L4_V8192_1M      4    128b  8192    2048       1 MiB
#   (and the same six with _2M = 2 MiB LLC)
#
# The 2- and 4-lane sets have identical VLEN/lane (128/512/2048), which
# separates "per-lane VRF depth" from "lane count". LLC is a per-config
# column, not a global: it is the 5th field of CONFIGS below.
#
# WHY THIS SCRIPT EXISTS AT ALL:
#   Vivado writes ONE output path, target/xilinx/out/cheshire.<board>.bit, and
#   ONE reports.impl/ directory. Every build overwrites the previous one. A
#   build run by hand (`make ara-chs-xilinx nr_lanes=4 vlen=4096 ...`) leaves
#   its result sitting in that single slot with no name and no reports copy,
#   and the NEXT build destroys it. That already happened once: the
#   2026-09-02 04:40 L4_V4096 build was found unarchived and was one `make`
#   away from being lost. rescue_orphan() below exists so it cannot happen
#   again, even if you build by hand and never call this script's build path.
#
# NOTE ON AXI WIDTH: it is NOT a free parameter. cheshire_top_xilinx.sv derives
# AxiDataWidth = max(32*NrLanes, 64), so the 2-lane builds are 64b and the
# 4-lane builds are 128b. At 64b the DRAM axi_dw_converter REAPPEARS (PS HP0 is
# 128b), costing ~20.6k LUTs. So a 2-lane build is NOT half the area, and a
# 2-vs-4 lane delta measures lanes AND memory bandwidth together. That is the
# realistic comparison, but say so when reporting it.
#
# ---------------------------------------------------------------------------
# USAGE (run on the server, from anywhere -- the script pins its own directory)
#
#   ./build_arch_sweep.sh                  # build every config still missing
#   ./build_arch_sweep.sh L4_V8192_2M       # build just that one
#   ./build_arch_sweep.sh L2_V1024_1M L2_V4096_2M
#   ./build_arch_sweep.sh --list           # show configs + what is archived
#   ./build_arch_sweep.sh --rescue         # ONLY archive a stray out/*.bit
#   FORCE=1 ./build_arch_sweep.sh L4_V2048_1M # rebuild even if already archived
#   ARCHIVE=/some/other/dir ./build_arch_sweep.sh
#
# Each config already present in the archive is SKIPPED unless FORCE=1, so an
# interrupted sweep resumes by just re-running the same command.
# ============================================================================
set -u

# FPU is held CONSTANT across all configs. LLC varies, but only in matched
# pairs (1M/2M at every lanes/VLEN point), so each delta stays attributable.
# Overridable from the environment only for a DELIBERATE sweep elsewhere:
#   LLC=1024 ./build_arch_sweep.sh           # ONLY configs with a 1 MiB LLC
#   LLC=2048 ./build_arch_sweep.sh           # ONLY configs with a 2 MiB LLC
# (LLC unset = all configs. It is a FILTER; each config carries its own LLC.)
# The LLC size lands in the archived filename (LLC1M / LLC512K / ...), so two
# sweeps at different LLC coexist in the archive without colliding.
# The Vivado launcher. This is NOT optional plumbing: three different defaults
# disagree -- target/xilinx/xilinx.mk says vitis-2020.2, deps/ara/cheshire's own
# Makefile says vitis-2021.1 but never forwards it to the sub-make, and a login
# shell may export something else again (a fresh ssh session on 2026-09-04 had
# vitis-2022.1, which is not installed, and the build died instantly). Every
# bitstream in the archive was in fact built with 2021.1, so pin that and pass
# it down explicitly. Override only to deliberately change tool version.
VIVADO=${VIVADO:-vitis-2021.1 vivado}

LLC_FILTER=${LLC:-}  # if set, build only configs with this LLC (KiB)
LLC=               # NOT a setting: each config in CONFIGS sets its own LLC (KiB) in the loops
FPU=${FPU:-16}       # Ara vector FPU: 16 = FP16 only, 32 = FP32 only, 1632 = both
BOARD=${BOARD:-zcu102}

# `make ara-chs-xilinx` is defined in the Makefile next to this script, and the
# recipe is CWD-sensitive -- so pin the working directory rather than trusting
# wherever this was invoked from (nohup, cron, tmux, another shell...).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || { echo "cannot cd to $SCRIPT_DIR"; exit 1; }

CHS_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

# Archive lives INSIDE the repo on the server by default, so it travels with
# the checkout and is visible over the sshfs mount from the laptop. $HOME is
# deliberately NOT the default: the laptop and the server have different homes
# and bitstreams kept turning up on whichever machine happened to run the make.
ARCHIVE=${ARCHIVE:-$CHS_ROOT/bitstream_archive}

BUILD=$CHS_ROOT/target/xilinx/build
OUTBIT=$CHS_ROOT/target/xilinx/out/cheshire.$BOARD.bit
REPORTS=$BUILD/$BOARD.cheshire/reports.impl
SYNTHLOG=$BUILD/cheshire.$BOARD.log
SRCTCL=$CHS_ROOT/target/xilinx/scripts/add_sources.$BOARD.tcl

mkdir -p "$ARCHIVE"

#            name          lanes vlen  axi  llc(KiB)
CONFIGS=(
  "L2_V256_1M      2     256   64   1024"
  "L2_V1024_1M     2     1024  64   1024"
  "L2_V4096_1M     2     4096  64   1024"
  "L4_V512_1M      4     512   128  1024"
  "L4_V2048_1M     4     2048  128  1024"
  "L4_V8192_1M     4     8192  128  1024"
  "L2_V256_2M      2     256   64   2048"
  "L2_V1024_2M     2     1024  64   2048"
  "L2_V4096_2M     2     4096  64   2048"
  "L4_V512_2M      4     512   128  2048"
  "L4_V2048_2M     4     2048  128  2048"
  "L4_V8192_2M     4     8192  128  2048"
)

# --- name helpers. Derive every label from the ACTUAL value, never hardcode:
#     a hardcoded "LLC1M" silently lies the moment LLC changes, and a
#     mislabelled archived bitstream is worse than no bitstream at all. ---
llctag() { if [ $(($1 % 1024)) -eq 0 ]; then echo "LLC$(($1 / 1024))M"; else echo "LLC$1K"; fi; }
fputag() {
  case "$1" in
    16)   echo FPUH  ;;   # FP16 only
    32)   echo FPUS  ;;   # FP32 only
    1632) echo FPUHS ;;   # both
    *)    echo "FPU$1" ;;
  esac
}
# RTLTAG marks a build whose RTL differs from the checked-in tree, so the
# archive says WHAT was built and not merely at what geometry. Without it two
# builds of the same lanes/VLEN/LLC are distinguishable only by date, which is
# not a record anyone can read six weeks later.
#   RTLTAG=upsizerfix ./build_arch_sweep.sh L2_V2048
# Leave it unset for a stock-RTL build. See AGENT_NOTES_FP16_4LANE.md 53.
RTLTAG=${RTLTAG:-}
mktag() {  # mktag <lanes> <vlen> <llc> <fpu> <axi> <date>
  local extra=""
  [ -n "$RTLTAG" ] && extra="_$RTLTAG"
  echo "L$1_V$2_$(llctag "$3")_$(fputag "$4")_DMA1_AXI$5_noAtomics_noDbg_allfixes${extra}_$6"
}

# Read the geometry that was ACTUALLY compiled, from the generated bender tcl.
# This is the only trustworthy source: make-level variables say what was asked
# for, this file says what the tools were handed.
built_val() { grep -m1 -oE "$1=[0-9]+" "$SRCTCL" 2>/dev/null | cut -d= -f2; }
built_fpu() {
  if   grep -q "ARA_FPU_1632" "$SRCTCL" 2>/dev/null; then echo 1632
  elif grep -q "ARA_FPU_32"   "$SRCTCL" 2>/dev/null; then echo 32
  else echo 16; fi
}

# Is this exact bitstream already in the archive? Compare by CONTENT, not name,
# so a hand-run build and a script-run build of the same config are recognised
# as the same artifact.
already_archived() {
  local md5; md5=$(md5sum "$1" 2>/dev/null | cut -d' ' -f1)
  [ -z "$md5" ] && return 1
  grep -qls "^$md5" "$ARCHIVE"/*.md5 2>/dev/null
}

# Pick a name that is not taken. Never overwrite: the standing rule is that
# nothing which works is discarded, so a collision gets an _rN suffix.
unique_base() {
  local tag=$1 base="$ARCHIVE/cheshire_top_xilinx_$1" n=2
  while [ -e "$base.bit" ]; do base="$ARCHIVE/cheshire_top_xilinx_${tag}_r$n"; n=$((n + 1)); done
  echo "$base"
}

# Copy the bitstream AND everything needed to interpret it later. A .bit with
# no utilization/timing report is nearly useless six weeks later, and the
# add_sources.tcl is the only record of what was actually compiled.
archive_artifacts() {  # archive_artifacts <base-path-without-extension>
  local base=$1 f
  # NEVER overwrite anything already in the archive: refuse if any file of
  # this base name exists, and use cp -n on top of that.
  for f in "$base".* "${base}"_*; do
    [ -e "$f" ] && { echo "*** archive: $f already exists -- refusing to overwrite"; return 1; }
  done
  cp -n "$OUTBIT"                      "$base.bit"                  || return 1
  # verify the copy is byte-identical before the source can be rebuilt over
  [ "$(md5sum < "$OUTBIT")" = "$(md5sum < "$base.bit")" ] || { echo "*** archive: md5 mismatch on copy"; return 1; }
  md5sum "$base.bit" | sed "s|$ARCHIVE/||" > "$base.md5"
  cp -n "$REPORTS/timing_summary.rpt"  "${base}_timing.rpt"          2>/dev/null
  cp -n "$REPORTS/utilization.rpt"     "${base}_utilization.rpt"     2>/dev/null
  cp -n "$SYNTHLOG"                    "${base}_synth.log"           2>/dev/null
  cp -n "$SRCTCL"                      "${base}_add_sources.tcl"     2>/dev/null
  chmod a-w "$base.bit" "$base.md5" 2>/dev/null
  return 0
}

# --- pull the headline numbers out of the reports ---
read_metrics() {  # sets globals WNS TNS FEP LUT FF BRAM DSP
  local T=$REPORTS/timing_summary.rpt U=$REPORTS/utilization.rpt
  WNS=$(awk '/Design Timing Summary/{f=1} f && /^ *-?[0-9]+\.[0-9]+ /{print $1; exit}' "$T" 2>/dev/null)
  TNS=$(awk '/Design Timing Summary/{f=1} f && /^ *-?[0-9]+\.[0-9]+ /{print $2; exit}' "$T" 2>/dev/null)
  FEP=$(awk '/Design Timing Summary/{f=1} f && /^ *-?[0-9]+\.[0-9]+ /{print $3; exit}' "$T" 2>/dev/null)
  read -r LUT FF BRAM DSP < <(awk -F'|' '$2 ~ /^ *cheshire_top_xilinx *$/ {
      for(i=4;i<=12;i++) gsub(/ /,"",$i); print $4, $8, $9, $12; exit}' "$U" 2>/dev/null)
}

# A Synth 8-689 port-width mismatch named a real RTL bug that went unnoticed
# for ~10 hardware rounds (issue_read_operands.sv fp_wdata_pack). This check is
# free; run it on every build.
# Count DISTINCT warnings, not lines. The benign ex_stage.sv:445 one is emitted
# once per elaboration pass (twice per build), so counting lines flags every
# clean build as suspicious and the flag stops meaning anything.
KNOWN_BENIGN_WIDTHWARN='ex_stage.sv:445'
check_width_warnings() {
  WIDTHWARN=0
  [ -f "$SYNTHLOG" ] || return 0
  local all novel
  all=$(grep "Synth 8-689" "$SYNTHLOG" 2>/dev/null | sed 's/^.*WARNING/WARNING/' | sort -u)
  [ -z "$all" ] && return 0
  novel=$(printf '%s\n' "$all" | grep -v "$KNOWN_BENIGN_WIDTHWARN")
  WIDTHWARN=$([ -n "$novel" ] && printf '%s\n' "$novel" | wc -l || echo 0)

  echo ""
  echo "*** port-width mismatches [Synth 8-689], distinct:"
  printf '%s\n' "$all" | sed 's/^/***   /'
  if [ "$WIDTHWARN" -eq 0 ]; then
    echo "*** All known-benign (ex_stage.sv:445, fpu_wrap result_o). OK."
  else
    echo "*** $WIDTHWARN NOT known-benign -- REAL BUG, investigate before trusting"
    echo "*** this bitstream. A Synth 8-689 named the FPR commit-port-1 defect that"
    echo "*** went unnoticed for ~10 hardware rounds."
  fi
  echo ""
}

SUMMARY=$ARCHIVE/ARCH_SWEEP_SUMMARY.txt
summary_header() {
  [ -f "$SUMMARY" ] && return
  {
    echo "# Architectural sweep results. Appended to, never rewritten."
    echo "# Device xczu9eg: 274080 LUT, 912 RAMB36, 2520 DSP. Clock 50 MHz (20.0 ns)."
    echo "# AXI width is DERIVED as max(32*lanes,64) -- not a free parameter."
    printf "%-12s %5s %6s %5s %5s %4s %9s %9s %7s %6s %7s %6s  %s\n" \
      NAME LANES VLEN LLCK FPU AXI LUT FF RAMB36 DSP WNS_ns MIN STATUS
  } > "$SUMMARY"
}
summary_row() {  # summary_row <name> <lanes> <vlen> <axi> <mins> <status>
  printf "%-12s %5s %6s %5s %5s %4s %9s %9s %7s %6s %7s %6s  %s\n" \
    "$1" "$2" "$3" "$LLC" "$FPU" "$4" "${LUT:--}" "${FF:--}" "${BRAM:--}" \
    "${DSP:--}" "${WNS:--}" "$5" "$6" >> "$SUMMARY"
}

# ---------------------------------------------------------------------------
# Rescue: archive a bitstream sitting in out/ that nothing has claimed yet.
# Runs before any build, so a hand-run result is preserved instead of clobbered.
# ---------------------------------------------------------------------------
rescue_orphan() {
  [ -f "$OUTBIT" ] || return 0
  if already_archived "$OUTBIT"; then
    echo "[rescue] out/ bitstream is already archived (md5 match) -- nothing to do."
    return 0
  fi

  local lanes vlen llc fpu axi date tag base
  lanes=$(built_val NR_LANES); vlen=$(built_val VLEN); llc=$(built_val LLC_KIB)
  fpu=$(built_fpu)
  if [ -z "$lanes" ] || [ -z "$vlen" ] || [ -z "$llc" ]; then
    echo "[rescue] !! out/cheshire.$BOARD.bit exists but $SRCTCL does not say what"
    echo "[rescue]    it was built with. Archiving under UNKNOWNCFG so it is not lost."
    tag="UNKNOWNCFG_$(date -r "$OUTBIT" +%Y%m%d_%H%M)"
  else
    axi=$(( 32 * lanes )); [ $axi -lt 64 ] && axi=64
    date=$(date -r "$OUTBIT" +%Y%m%d)
    tag=$(mktag "$lanes" "$vlen" "$llc" "$fpu" "$axi" "$date")
  fi

  base=$(unique_base "$tag")
  echo ""
  echo "[rescue] unarchived bitstream found in out/ -- saving it before building."
  echo "[rescue]   $OUTBIT  ($(date -r "$OUTBIT" '+%Y-%m-%d %H:%M'))"
  echo "[rescue]   -> $base.bit"
  if archive_artifacts "$base"; then
    read_metrics
    summary_header
    summary_row "L${lanes:-?}_V${vlen:-?}" "${lanes:-?}" "${vlen:-?}" "${axi:-?}" "-" "RESCUED"
    echo "[rescue]   LUT=${LUT:-?} FF=${FF:-?} RAMB36=${BRAM:-?} DSP=${DSP:-?} WNS=${WNS:-?}ns"
  else
    echo "[rescue] !! copy FAILED -- refusing to continue, your bitstream is still in out/"
    exit 1
  fi
}

# ---------------------------------------------------------------------------
run_one() {
  local name=$1 lanes=$2 vlen=$3 axi=$4
  local date; date=$(date +%Y%m%d)
  local tag; tag=$(mktag "$lanes" "$vlen" "$LLC" "$FPU" "$axi" "$date")
  local log=$ARCHIVE/build_${name}_$(llctag "$LLC")_$(fputag "$FPU")_${date}.log
  local ln=2
  while [ -e "$log" ]; do log=${log%.log}; log=${log%_r[0-9]*}_r$ln.log; ln=$((ln + 1)); done

  echo ""
  echo "=============================================================="
  echo " BUILD $name : nr_lanes=$lanes vlen=$vlen llc=$LLC fpu=$FPU (AXI ${axi}b)"
  echo " started  $(date '+%Y-%m-%d %H:%M:%S')"
  echo " log      -> $log"
  echo " archive  -> $ARCHIVE/cheshire_top_xilinx_${tag}.bit"
  echo "=============================================================="

  # Re-check before EVERY build: anything in out/ that is not archived is saved
  # first, so the rm below can never destroy an unarchived bitstream.
  rescue_orphan

  # Remove the previous bitstream so a FAILED build cannot be mistaken for a
  # successful one by the archive step below. Safe: rescue_orphan() already ran.
  rm -f "$OUTBIT"

  local t0; t0=$(date +%s)
  make ara-chs-xilinx BOARD=$BOARD MODE=batch VIVADO="$VIVADO" \
       nr_lanes="$lanes" vlen="$vlen" llc="$LLC" fpu="$FPU" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  local mins=$(( ($(date +%s) - t0) / 60 ))

  summary_header
  if [ $rc -ne 0 ] || [ ! -f "$OUTBIT" ]; then
    echo "*** $name FAILED (rc=$rc, bitstream present: $([ -f "$OUTBIT" ] && echo yes || echo no))"
    LUT=; FF=; BRAM=; DSP=; WNS=
    summary_row "$name" "$lanes" "$vlen" "$axi" "$mins" "FAILED"
    return 1
  fi

  # --- sanity: did the defines actually reach the build? ---
  local got_lanes got_vlen got_llc
  got_lanes=$(built_val NR_LANES); got_vlen=$(built_val VLEN); got_llc=$(built_val LLC_KIB)
  if [ "$got_lanes" != "$lanes" ] || [ "$got_vlen" != "$vlen" ] || [ "$got_llc" != "$LLC" ]; then
    echo "*** $name CONFIG MISMATCH: built NR_LANES=$got_lanes VLEN=$got_vlen LLC_KIB=$got_llc"
    echo "*** Archiving anyway, under the geometry it ACTUALLY built."
    tag=$(mktag "$got_lanes" "$got_vlen" "$got_llc" "$(built_fpu)" \
                "$(( 32 * got_lanes < 64 ? 64 : 32 * got_lanes ))" "$date")
  fi

  # --- archive BEFORE the next build overwrites out/ and reports.impl/ ---
  local base; base=$(unique_base "$tag")
  if ! archive_artifacts "$base"; then
    echo "*** $name ARCHIVE FAILED -- bitstream still in $OUTBIT, stopping."
    exit 1
  fi
  mv -n "$log" "${base}_build.log" 2>/dev/null && log=${base}_build.log

  check_width_warnings
  read_metrics

  local status="OK"
  [ "${FEP:-0}" != "0" ] && status="TIMING_FAIL(${FEP}ep)"
  [ "${WIDTHWARN:-0}" -gt 0 ] && status="$status,WIDTHWARN($WIDTHWARN)"
  [ "$got_lanes" != "$lanes" ] && status="$status,CONFIG_MISMATCH"

  summary_row "$name" "$got_lanes" "$got_vlen" "$axi" "$mins" "$status"

  echo ""
  echo ">>> $name done in ${mins} min: LUT=$LUT FF=$FF RAMB36=$BRAM DSP=$DSP WNS=${WNS}ns TNS=$TNS failing_ep=$FEP [$status]"
  echo ">>> archived: $base.bit"
  return 0
}

# ---------------------------------------------------------------------------
do_list() {
  echo "Archive: $ARCHIVE"
  echo ""
  printf "%-12s %5s %6s %5s %5s  %s\n" NAME LANES VLEN LLCK FPU ARCHIVED
  for cfg in "${CONFIGS[@]}"; do
    read -r name lanes vlen axi LLC <<< "$cfg"
    [ -n "$LLC_FILTER" ] && [ "$LLC_FILTER" != "$LLC" ] && continue
    local_hits=$(ls "$ARCHIVE"/cheshire_top_xilinx_L${lanes}_V${vlen}_$(llctag "$LLC")_$(fputag "$FPU")_*.bit 2>/dev/null | wc -l)
    printf "%-12s %5s %6s %5s %5s  %s\n" "$name" "$lanes" "$vlen" "$LLC" "$FPU" \
      "$([ "$local_hits" -gt 0 ] && echo "yes ($local_hits)" || echo "no")"
  done
  echo ""
  echo "Bitstreams currently archived:"
  ls -1 "$ARCHIVE"/*.bit 2>/dev/null | sed 's|.*/|  |' || echo "  (none)"
}

WANT=("$@")
case "${1:-}" in
  --list|-l)   do_list; exit 0 ;;
  --rescue|-r) rescue_orphan; exit 0 ;;
  --help|-h)   sed -n '2,50p' "$0"; exit 0 ;;
esac

# Check the toolchain BEFORE anything else. Discovering that the Vivado wrapper
# does not exist is a 2-second check; discovering it from inside make wastes the
# run and buries the reason under a make error cascade.
VIVADO_CMD=${VIVADO%% *}
if ! command -v "$VIVADO_CMD" >/dev/null 2>&1; then
  echo "*** '$VIVADO_CMD' is not on PATH -- cannot build."
  echo "*** VIVADO is currently: $VIVADO"
  echo "*** Launchers that DO exist here:"
  compgen -c 2>/dev/null | grep -E '^(vitis|vivado)' | sort -u | sed 's/^/***   /' \
    || echo "***   (none found matching vitis*/vivado*)"
  echo "*** Every archived bitstream was built with 2021.1. Re-run as e.g.:"
  echo "***   VIVADO='vitis-2021.1 vivado' $0 $*"
  exit 1
fi

# ★ ALWAYS run this first. It is the whole point of the script.
rescue_orphan

fail=0; built=0; skipped=0
for cfg in "${CONFIGS[@]}"; do
  read -r name lanes vlen axi LLC <<< "$cfg"
  [ -n "$LLC_FILTER" ] && [ "$LLC_FILTER" != "$LLC" ] && continue

  if [ ${#WANT[@]} -gt 0 ]; then
    match=0
    for w in "${WANT[@]}"; do [ "$w" = "$name" ] && match=1; done
    [ $match -eq 0 ] && continue
  fi

  if [ "${FORCE:-0}" != "1" ] && \
     ls "$ARCHIVE"/cheshire_top_xilinx_L${lanes}_V${vlen}_$(llctag "$LLC")_$(fputag "$FPU")_*.bit >/dev/null 2>&1; then
    echo "--- $name already archived, skipping (FORCE=1 to rebuild)"
    skipped=$((skipped + 1))
    continue
  fi

  run_one "$name" "$lanes" "$vlen" "$axi" || fail=$((fail + 1))
  built=$((built + 1))
done

echo ""
echo "=============================== SUMMARY ==============================="
cat "$SUMMARY" 2>/dev/null
echo "======================================================================="
echo "built=$built  skipped=$skipped  failed=$fail"
echo "Archive:      $ARCHIVE"
echo "Summary file: $SUMMARY"
[ $fail -gt 0 ] && exit 1
exit 0
