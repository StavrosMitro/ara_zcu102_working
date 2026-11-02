#!/bin/bash
# ============================================================================
# Probe sweep builder -- bitstreams WITH ILA probes, for cycle-exact timing of
# vector instructions (vle vs vfslide1down). Same discipline as
# build_arch_sweep.sh: every artifact is archived under a unique name before
# anything can overwrite it, and nothing in the archive is ever overwritten.
#
#   4 bitstreams = {2 lanes, 4 lanes} x {VLEN 1024, VLEN 4096},  LLC 1 MiB
#
#   name          lanes  AXI    VLEN   VLEN/lane
#   L2_V1024        2    64b    1024     512
#   L2_V4096        2    64b    4096    2048
#   L4_V1024        4    128b   1024     256
#   L4_V4096        4    128b   4096    1024
#
# PROBES -- which ILA probe groups are compiled in (see the `probes` knob in
# the Makefile next to this script; the RTL blocks are `ifdef PROBE_MEM /
# `ifdef PROBE_VEC, so a probed and an unprobed build differ in nothing else):
#   mem   memory path: AXI points C1/A1/X/L2/L3 + CVA6 L1-D load port (round 4)
#   vec   vector-instruction lifecycle: CVA6<->Ara handshake + Ara sequencer
#         issue/done per unit (VLDU, SLDU, ...). Required to time a slide.
#   Default: mem,vec  (a load is then timed end to end AND split at the AXI
#   port; a slide is timed by the sequencer points).
#
# DIFFERENCES FROM build_arch_sweep.sh, all deliberate:
#   * the .ltx probe file is archived next to the .bit -- without it the ILA
#     cannot be read, and the arch sweep never needed one;
#   * after each build the .ltx is opened and its probes counted, so a build
#     whose ILA silently did not get inserted is flagged, not archived as good;
#   * the probe groups are part of the archived name and of the skip check, so
#     an existing UNprobed L2_V1024 does not make this script skip L2_V1024;
#   * results go to PROBE_SWEEP_SUMMARY.txt, not ARCH_SWEEP_SUMMARY.txt -- an
#     ILA adds ~8k+ LUT, and mixing those rows into the area table would make
#     every area comparison wrong.
#
# Builds run ONE AFTER ANOTHER in the single cheshire_soc tree (exactly like
# build_arch_sweep.sh): Vivado has one out/ slot and one reports.impl/. Expect
# ~2 h each, ~8 h for all four. Resumable: re-run the same command.
#
# ---------------------------------------------------------------------------
# USAGE (run on the server)
#   ./build_probe_sweep.sh                    # build every config not yet archived
#   ./build_probe_sweep.sh L2_V1024 L4_V4096  # just these
#   ./build_probe_sweep.sh --list             # configs + what is archived
#   ./build_probe_sweep.sh --rescue           # ONLY archive a stray out/*.bit
#   PROBES=vec ./build_probe_sweep.sh         # only the vector-lifecycle probes
#   PROBES=mem,vec FPU=32 ./build_probe_sweep.sh
#   FORCE=1 ./build_probe_sweep.sh L2_V1024   # rebuild even if archived
# ============================================================================
set -u

VIVADO=${VIVADO:-vitis-2021.1 vivado}   # pinned: every archived bitstream is 2021.1
PROBES=${PROBES:-mem,vec}
LLC=${LLC:-1024}
FPU=${FPU:-16}           # 16 = FP16 (vfslide1down at e16), 32 = FP32, 1632 = both
BOARD=${BOARD:-zcu102}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || { echo "cannot cd to $SCRIPT_DIR"; exit 1; }
CHS_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
ARCHIVE=${ARCHIVE:-$CHS_ROOT/bitstream_archive}

BUILD=$CHS_ROOT/target/xilinx/build
OUTBIT=$CHS_ROOT/target/xilinx/out/cheshire.$BOARD.bit
OUTLTX=$CHS_ROOT/target/xilinx/out/cheshire.$BOARD.ltx
REPORTS=$BUILD/$BOARD.cheshire/reports.impl
SYNTHLOG=$BUILD/cheshire.$BOARD.log
SRCTCL=$CHS_ROOT/target/xilinx/scripts/add_sources.$BOARD.tcl
SUMMARY=$ARCHIVE/PROBE_SWEEP_SUMMARY.txt
mkdir -p "$ARCHIVE"

#            name       lanes vlen  axi
CONFIGS=(
  "L2_V1024    2     1024  64"
  "L2_V4096    2     4096  64"
  "L4_V1024    4     1024  128"
  "L4_V4096    4     4096  128"
)

# --- validate PROBES here too, so a typo fails in 1 s instead of after make ---
for p in ${PROBES//,/ }; do
  case "$p" in mem|vec) ;; *) echo "*** PROBES: unknown group '$p' (valid: mem vec)"; exit 1 ;; esac
done
[ -z "${PROBES//[ ,]/}" ] && { echo "*** PROBES is empty -- use build_arch_sweep.sh for unprobed builds"; exit 1; }

# --- name helpers: every label derived from the value it describes ---------
llctag() { if [ $(($1 % 1024)) -eq 0 ]; then echo "LLC$(($1 / 1024))M"; else echo "LLC$1K"; fi; }
fputag() { case "$1" in 16) echo FPUH;; 32) echo FPUS;; 1632) echo FPUHS;; *) echo "FPU$1";; esac; }
# canonical, order-independent probe tag:  "vec,mem" and "mem vec" -> probes-mem-vec
probetag() { echo "probes-$(echo "$1" | tr ', ' '\n\n' | grep -v '^$' | sort -u | paste -sd- -)"; }
mktag() {  # mktag <lanes> <vlen> <llc> <fpu> <axi> <probes> <date>
  echo "L$1_V$2_$(llctag "$3")_$(fputag "$4")_DMA1_AXI$5_noAtomics_noDbg_allfixes_$(probetag "$6")_$7"
}

# What the tools were ACTUALLY handed -- read back from the generated tcl.
built_val()    { grep -m1 -oE "$1=[0-9]+" "$SRCTCL" 2>/dev/null | cut -d= -f2; }
built_fpu()    { if grep -q ARA_FPU_1632 "$SRCTCL" 2>/dev/null; then echo 1632;
                 elif grep -q ARA_FPU_32 "$SRCTCL" 2>/dev/null; then echo 32; else echo 16; fi; }
built_probes() { local l=""; grep -q PROBE_MEM "$SRCTCL" 2>/dev/null && l="$l mem"
                 grep -q PROBE_VEC "$SRCTCL" 2>/dev/null && l="$l vec"; echo "${l# }"; }
norm_probes()  { echo "$1" | tr ', ' '\n\n' | grep -v '^$' | sort -u | paste -sd' ' -; }

# Count the probes the .ltx actually carries, per group. An ltx is JSON-ish
# text listing every probe by net name; dbg_<point>_... names identify groups.
ltx_counts() {  # prints "mem=<n> vec=<n> total=<n>"
  local f=$1
  [ -f "$f" ] || { echo "mem=0 vec=0 total=0"; return; }
  local mem vec tot
  # names are hierarchical ("i_cheshire_soc/dbg_a1_arr", or "gen_llc.dbg_l2_arv" inside a
  # generate block -- hence [/.]); dbg_hub is the debug
  # hub core itself, not a probe, and the regexes below never match it.
  mem=$(grep -oE '"name": *"[^"]*[/.]dbg_(a1|c1|x|l2|l3|d1)_[a-z]+"' "$f" | sort -u | wc -l)
  vec=$(grep -oE '"name": *"[^"]*[/.]dbg_(q|v)_[a-z_]+"'             "$f" | sort -u | wc -l)
  tot=$(grep -oE '"name": *"[^"]*[/.]dbg_[a-z0-9_]+"'                "$f" | sort -u | wc -l)
  echo "mem=$mem vec=$vec total=$tot"
}

already_archived() { local m; m=$(md5sum "$1" 2>/dev/null | cut -d' ' -f1)
                     [ -n "$m" ] && grep -qls "^$m" "$ARCHIVE"/*.md5 2>/dev/null; }
unique_base() { local tag=$1 base="$ARCHIVE/cheshire_top_xilinx_$1" n=2
                while [ -e "$base.bit" ]; do base="$ARCHIVE/cheshire_top_xilinx_${tag}_r$n"; n=$((n+1)); done
                echo "$base"; }

archive_artifacts() {  # archive_artifacts <base>
  local base=$1 f
  for f in "$base".* "${base}"_*; do
    [ -e "$f" ] && { echo "*** archive: $f already exists -- refusing to overwrite"; return 1; }
  done
  cp -n "$OUTBIT" "$base.bit" || return 1
  [ "$(md5sum < "$OUTBIT")" = "$(md5sum < "$base.bit")" ] || { echo "*** archive: md5 mismatch"; return 1; }
  md5sum "$base.bit" | sed "s|$ARCHIVE/||" > "$base.md5"
  # the probe file is NOT optional here: a probed .bit without its .ltx is unreadable
  if [ -f "$OUTLTX" ]; then
    cp -n "$OUTLTX" "$base.ltx"
    md5sum "$base.ltx" | sed "s|$ARCHIVE/||" >> "$base.md5"
  else
    echo "*** archive: $OUTLTX MISSING -- the ILA of this bitstream cannot be read"
  fi
  cp -n "$REPORTS/timing_summary.rpt" "${base}_timing.rpt"      2>/dev/null
  cp -n "$REPORTS/utilization.rpt"    "${base}_utilization.rpt" 2>/dev/null
  cp -n "$SYNTHLOG"                   "${base}_synth.log"       2>/dev/null
  cp -n "$SRCTCL"                     "${base}_add_sources.tcl" 2>/dev/null
  chmod a-w "$base.bit" "$base.md5" "$base.ltx" 2>/dev/null
  return 0
}

read_metrics() {
  local T=$REPORTS/timing_summary.rpt U=$REPORTS/utilization.rpt
  WNS=$(awk '/Design Timing Summary/{f=1} f && /^ *-?[0-9]+\.[0-9]+ /{print $1; exit}' "$T" 2>/dev/null)
  FEP=$(awk '/Design Timing Summary/{f=1} f && /^ *-?[0-9]+\.[0-9]+ /{print $3; exit}' "$T" 2>/dev/null)
  read -r LUT FF BRAM DSP < <(awk -F'|' '$2 ~ /^ *cheshire_top_xilinx *$/ {
      for(i=4;i<=12;i++) gsub(/ /,"",$i); print $4, $8, $9, $12; exit}' "$U" 2>/dev/null)
  ILA_LUT=$(awk -F'|' '$2 ~ /^ *i_ila *$/ {gsub(/ /,"",$4); print $4; exit}' "$U" 2>/dev/null)
}

KNOWN_BENIGN_WIDTHWARN='ex_stage.sv:445'
check_width_warnings() {
  WIDTHWARN=0; [ -f "$SYNTHLOG" ] || return 0
  local all novel
  all=$(grep "Synth 8-689" "$SYNTHLOG" 2>/dev/null | sed 's/^.*WARNING/WARNING/' | sort -u)
  [ -z "$all" ] && return 0
  novel=$(printf '%s\n' "$all" | grep -v "$KNOWN_BENIGN_WIDTHWARN")
  WIDTHWARN=$([ -n "$novel" ] && printf '%s\n' "$novel" | wc -l || echo 0)
  echo "*** Synth 8-689 port-width mismatches (distinct):"; printf '%s\n' "$all" | sed 's/^/***   /'
  [ "$WIDTHWARN" -gt 0 ] && echo "*** $WIDTHWARN NOT known-benign -- investigate before trusting this bitstream"
}

summary_header() {
  [ -f "$SUMMARY" ] && return
  { echo "# Probe sweep (ILA-carrying bitstreams). Appended to, never rewritten."
    echo "# Area here INCLUDES the ILA -- do not compare against ARCH_SWEEP_SUMMARY.txt."
    printf "%-10s %5s %6s %5s %5s %4s %-9s %9s %8s %7s %6s %7s %5s %-24s %s\n" \
      NAME LANES VLEN LLCK FPU AXI PROBES LUT ILA_LUT RAMB36 DSP WNS_ns MIN LTX STATUS
  } > "$SUMMARY"
}
summary_row() {  # <name> <lanes> <vlen> <axi> <probes> <mins> <ltx> <status>
  printf "%-10s %5s %6s %5s %5s %4s %-9s %9s %8s %7s %6s %7s %5s %-24s %s\n" \
    "$1" "$2" "$3" "$LLC" "$FPU" "$4" "$5" "${LUT:--}" "${ILA_LUT:--}" "${BRAM:--}" \
    "${DSP:--}" "${WNS:--}" "$6" "$7" "$8" >> "$SUMMARY"
}

rescue_orphan() {
  [ -f "$OUTBIT" ] || return 0
  already_archived "$OUTBIT" && { echo "[rescue] out/ bitstream already archived -- nothing to do."; return 0; }
  local lanes vlen llc fpu axi pr tag base
  lanes=$(built_val NR_LANES); vlen=$(built_val VLEN); llc=$(built_val LLC_KIB)
  fpu=$(built_fpu); pr=$(built_probes)
  if [ -z "$lanes" ] || [ -z "$vlen" ] || [ -z "$llc" ]; then
    tag="UNKNOWNCFG_$(date -r "$OUTBIT" +%Y%m%d_%H%M)"
  else
    axi=$(( 32 * lanes )); [ $axi -lt 64 ] && axi=64
    if [ -n "$pr" ]; then tag=$(mktag "$lanes" "$vlen" "$llc" "$fpu" "$axi" "$pr" "$(date -r "$OUTBIT" +%Y%m%d)")
    else tag="L${lanes}_V${vlen}_$(llctag "$llc")_$(fputag "$fpu")_DMA1_AXI${axi}_noAtomics_noDbg_allfixes_$(date -r "$OUTBIT" +%Y%m%d)"; fi
  fi
  base=$(unique_base "$tag")
  echo "[rescue] unarchived bitstream in out/ -> $base.bit"
  archive_artifacts "$base" || { echo "[rescue] !! copy FAILED -- stopping, your bitstream is still in out/"; exit 1; }
}

run_one() {
  local name=$1 lanes=$2 vlen=$3 axi=$4
  local date; date=$(date +%Y%m%d)
  local want; want=$(norm_probes "$PROBES")
  local tag; tag=$(mktag "$lanes" "$vlen" "$LLC" "$FPU" "$axi" "$PROBES" "$date")
  local log=$ARCHIVE/build_${name}_$(llctag "$LLC")_$(fputag "$FPU")_$(probetag "$PROBES")_${date}.log
  local n=2; while [ -e "$log" ]; do log=${log%.log}; log=${log%_r[0-9]*}_r$n.log; n=$((n+1)); done

  echo ""
  echo "=============================================================="
  echo " BUILD $name : lanes=$lanes vlen=$vlen llc=$LLC fpu=$FPU probes=[$want] (AXI ${axi}b)"
  echo " started  $(date '+%F %T')    log -> $log"
  echo "=============================================================="
  rescue_orphan
  rm -f "$OUTBIT" "$OUTLTX"      # a failed build must not leave a stale .bit/.ltx behind

  local t0; t0=$(date +%s)
  make ara-chs-xilinx BOARD=$BOARD MODE=batch VIVADO="$VIVADO" \
       nr_lanes="$lanes" vlen="$vlen" llc="$LLC" fpu="$FPU" probes="$PROBES" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  local mins=$(( ($(date +%s) - t0) / 60 ))
  summary_header

  if [ $rc -ne 0 ] || [ ! -f "$OUTBIT" ]; then
    echo "*** $name FAILED (rc=$rc)"; LUT=; ILA_LUT=; BRAM=; DSP=; WNS=
    summary_row "$name" "$lanes" "$vlen" "$axi" "$want" "$mins" "-" "FAILED"; return 1
  fi

  # --- did the requested configuration AND probes actually reach the build? ---
  local gl gv gll gp
  gl=$(built_val NR_LANES); gv=$(built_val VLEN); gll=$(built_val LLC_KIB); gp=$(built_probes)
  local status=OK
  if [ "$gl" != "$lanes" ] || [ "$gv" != "$vlen" ] || [ "$gll" != "$LLC" ]; then
    echo "*** $name CONFIG MISMATCH: built NR_LANES=$gl VLEN=$gv LLC_KIB=$gll -- archiving under that"
    tag=$(mktag "$gl" "$gv" "$gll" "$(built_fpu)" "$(( 32*gl < 64 ? 64 : 32*gl ))" "${gp:-none}" "$date")
    status="$status,CONFIG_MISMATCH"
  fi
  [ "$gp" != "$want" ] && { echo "*** $name PROBE MISMATCH: asked [$want], add_sources has [$gp]"; status="$status,PROBE_MISMATCH"; }

  # --- did the ILA actually get inserted, with the expected groups? ---
  local lc; lc=$(ltx_counts "$OUTLTX")
  echo ">>> ltx: $lc"
  local lmem lvec; lmem=$(echo "$lc" | sed -E 's/.*mem=([0-9]+).*/\1/'); lvec=$(echo "$lc" | sed -E 's/.*vec=([0-9]+).*/\1/')
  [ ! -f "$OUTLTX" ]                                 && status="$status,NO_LTX"
  [[ " $want " == *" mem "* ]] && [ "$lmem" -eq 0 ]  && status="$status,MEM_PROBES_ABSENT"
  [[ " $want " == *" vec "* ]] && [ "$lvec" -eq 0 ]  && status="$status,VEC_PROBES_ABSENT"

  local base; base=$(unique_base "$tag")
  archive_artifacts "$base" || { echo "*** $name ARCHIVE FAILED -- bitstream still in $OUTBIT, stopping."; exit 1; }
  mv -n "$log" "${base}_build.log" 2>/dev/null

  check_width_warnings; read_metrics
  [ "${FEP:-0}" != "0" ] && status="$status,TIMING_FAIL(${FEP}ep)"
  [ "${WIDTHWARN:-0}" -gt 0 ] && status="$status,WIDTHWARN($WIDTHWARN)"
  summary_row "$name" "$gl" "$gv" "$axi" "$want" "$mins" "$lc" "$status"
  echo ">>> $name done in ${mins} min: LUT=$LUT (ILA $ILA_LUT) RAMB36=$BRAM WNS=${WNS}ns [$status]"
  echo ">>> archived: $base.bit  +  $base.ltx"
}

do_list() {
  echo "Archive: $ARCHIVE    PROBES=$(norm_probes "$PROBES")  FPU=$FPU  LLC=$LLC"
  printf "%-10s %5s %6s  %s\n" NAME LANES VLEN ARCHIVED
  for cfg in "${CONFIGS[@]}"; do
    read -r name lanes vlen axi <<< "$cfg"
    local h; h=$(ls "$ARCHIVE"/cheshire_top_xilinx_L${lanes}_V${vlen}_$(llctag "$LLC")_$(fputag "$FPU")_*_$(probetag "$PROBES")_*.bit 2>/dev/null | wc -l)
    printf "%-10s %5s %6s  %s\n" "$name" "$lanes" "$vlen" "$([ "$h" -gt 0 ] && echo "yes ($h)" || echo no)"
  done
}

WANT=("$@")
case "${1:-}" in
  --list|-l)   do_list; exit 0 ;;
  --rescue|-r) rescue_orphan; exit 0 ;;
  --help|-h)   sed -n '2,52p' "$0"; exit 0 ;;
esac

VIVADO_CMD=${VIVADO%% *}
command -v "$VIVADO_CMD" >/dev/null 2>&1 || { echo "*** '$VIVADO_CMD' not on PATH (VIVADO=$VIVADO)"; exit 1; }

rescue_orphan
fail=0; built=0; skipped=0
for cfg in "${CONFIGS[@]}"; do
  read -r name lanes vlen axi <<< "$cfg"
  if [ ${#WANT[@]} -gt 0 ]; then m=0; for w in "${WANT[@]}"; do [ "$w" = "$name" ] && m=1; done; [ $m -eq 0 ] && continue; fi
  if [ "${FORCE:-0}" != "1" ] && ls "$ARCHIVE"/cheshire_top_xilinx_L${lanes}_V${vlen}_$(llctag "$LLC")_$(fputag "$FPU")_*_$(probetag "$PROBES")_*.bit >/dev/null 2>&1; then
    echo "--- $name ($(probetag "$PROBES")) already archived, skipping (FORCE=1 to rebuild)"; skipped=$((skipped+1)); continue
  fi
  run_one "$name" "$lanes" "$vlen" "$axi" || fail=$((fail+1))
  built=$((built+1))
done

echo ""; echo "=============================== SUMMARY ==============================="
cat "$SUMMARY" 2>/dev/null
echo "built=$built skipped=$skipped failed=$fail    archive: $ARCHIVE"
[ $fail -gt 0 ] && exit 1
exit 0
