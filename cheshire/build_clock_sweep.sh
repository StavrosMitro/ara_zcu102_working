#!/bin/bash
# ============================================================================
# Clock sweep builder -- the SAME architecture at different clocks, with the
# DDR-path ILA probes, to see how the DDR response depends on the clock.
# Question (2026-10-27): what does the DDR response look like at 50 vs 100 MHz?
# The SoC itself cannot run at 100 MHz (critical path 16.6-19.1 ns, see
# TIMING_CRITICAL_PATH_20261027.md), so the cases below vary the two clocks
# that CAN move:
#
#   case                 soc_clk  HP0 clk  isolates
#   L2_V1024_SOC50_HP0100   50     100     the HP0 port / CDC-dst clock alone
#   L2_V1024_SOC25          25      25     fixed-ns vs fixed-cycle split (2nd point)
#   L2_V1024_SOC75          75      75     3rd point + real Fmax attempt (may fail timing)
#   L2_V1024_SOC50          50      50     baseline with the SAME probes (point H is new)
#
# Everything goes through the make knobs, nothing is hard-wired here:
#   probes=mem,ddr  soc_mhz=<f>  hp0_mhz=<f>      (deps/ara/cheshire/Makefile)
# With no knob the RTL is the original one, so this script cannot change what
# build_arch_sweep.sh / build_probe_sweep.sh produce. It shares only the
# archive directory, never overwrites anything in it, and writes its own
# CLOCK_SWEEP_SUMMARY.txt.
#
# Builds run ONE AFTER ANOTHER in the single cheshire_soc tree (Vivado has one
# out/ slot). ~2 h each. Resumable: re-run the same command.
#
# ---------------------------------------------------------------------------
# USAGE (run on coroni, preferably inside tmux)
#   ./build_clock_sweep.sh --list                         # cases + what is archived
#   ./build_clock_sweep.sh 2>&1 | tee clock_sweep.log     # all cases not yet archived
#   ./build_clock_sweep.sh L2_V1024_SOC25 L2_V1024_SOC75  # just these
#   PROBES=mem,ddr,vec ./build_clock_sweep.sh             # other probe groups
#   FORCE=1 ./build_clock_sweep.sh L2_V1024_SOC75         # rebuild even if archived
#   ./build_clock_sweep.sh --rescue                       # ONLY archive a stray out/*.bit
# ============================================================================
set -u

VIVADO=${VIVADO:-vitis-2021.1 vivado}   # pinned: every archived bitstream is 2021.1
PROBES=${PROBES:-mem,ddr}
LLC=${LLC:-1024}
FPU=${FPU:-16}
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
SUMMARY=$ARCHIVE/CLOCK_SWEEP_SUMMARY.txt
mkdir -p "$ARCHIVE"

#         name                    lanes vlen axi soc hp0 ("-" = same as soc)
CASES=(
  "L2_V1024_SOC50_HP0100            2  1024  64  50  100"
  "L2_V1024_SOC25                   2  1024  64  25  -"
  "L2_V1024_SOC75                   2  1024  64  75  -"
  "L2_V1024_SOC50                   2  1024  64  50  -"
)

for p in ${PROBES//,/ }; do
  case "$p" in mem|vec|ddr) ;; *) echo "*** PROBES: unknown group '$p' (valid: mem vec ddr)"; exit 1 ;; esac
done

# --- name helpers ------------------------------------------------------------
llctag()   { if [ $(($1 % 1024)) -eq 0 ]; then echo "LLC$(($1 / 1024))M"; else echo "LLC$1K"; fi; }
fputag()   { case "$1" in 16) echo FPUH;; 32) echo FPUS;; 1632) echo FPUHS;; *) echo "FPU$1";; esac; }
probetag() { local t; t=$(echo "$1" | tr ', ' '\n\n' | grep -v '^$' | sort -u | paste -sd- -)
             [ -n "$t" ] && echo "probes-$t" || echo "noprobes"; }
clktag()   { if [ "$2" = "-" ]; then echo "SOC$1"; else echo "SOC$1_HP0$2"; fi; }   # <soc> <hp0>
mktag() {  # mktag <lanes> <vlen> <axi> <soc> <hp0> <probes> <date>
  echo "L$1_V$2_$(llctag "$LLC")_$(fputag "$FPU")_DMA1_AXI$3_noAtomics_noDbg_allfixes_$(probetag "$6")_$(clktag "$4" "$5")_$7"
}
# exact-match glob: "..._SOC50_2026..." never matches "..._SOC50_HP0100_2026..."
archived_glob() {  # <lanes> <vlen> <axi> <soc> <hp0>
  echo "$ARCHIVE/cheshire_top_xilinx_L$1_V$2_$(llctag "$LLC")_$(fputag "$FPU")_DMA1_AXI$3_noAtomics_noDbg_allfixes_$(probetag "$PROBES")_$(clktag "$4" "$5")_[0-9]*.bit"
}

# What the tools were ACTUALLY handed -- read back from the generated tcl.
built_val()    { grep -m1 -oE "$1=[0-9]+" "$SRCTCL" 2>/dev/null | cut -d= -f2; }
built_probes() { local l=""; grep -q PROBE_DDR "$SRCTCL" 2>/dev/null && l="$l ddr"
                 grep -q PROBE_MEM "$SRCTCL" 2>/dev/null && l="$l mem"
                 grep -q PROBE_VEC "$SRCTCL" 2>/dev/null && l="$l vec"; echo "${l# }"; }
norm_probes()  { echo "$1" | tr ', ' '\n\n' | grep -v '^$' | sort -u | paste -sd' ' -; }

ltx_counts() {  # prints "mem=<n> ddr=<n> vec=<n> total=<n>"
  local f=$1
  [ -f "$f" ] || { echo "mem=0 ddr=0 vec=0 total=0"; return; }
  local mem ddr vec tot
  mem=$(grep -oE '"name": *"[^"]*[/.]dbg_(a1|c1|x|l2|l3|d1)_[a-z]+"' "$f" | sort -u | wc -l)
  ddr=$(grep -oE '"name": *"[^"]*[/.]dbg_h_[a-z]+"'                  "$f" | sort -u | wc -l)
  vec=$(grep -oE '"name": *"[^"]*[/.]dbg_(q|v)_[a-z_]+"'             "$f" | sort -u | wc -l)
  tot=$(grep -oE '"name": *"[^"]*[/.]dbg_[a-z0-9_]+"'                "$f" | sort -u | wc -l)
  echo "mem=$mem ddr=$ddr vec=$vec total=$tot"
}

# Is there a clock of <f> MHz in the timing report's Clock Summary? (+-0.1 MHz)
has_clock_mhz() {  # <rpt> <mhz>
  awk -v f="$2" '/\| Clock Summary/{c=1} /\| Intra Clock Table/{c=0}
                 c && NF>=4 && $NF ~ /^[0-9]+\.[0-9]+$/ {d=$NF-f; if (d<0) d=-d; if (d<0.1) ok=1}
                 END{exit !ok}' "$1" 2>/dev/null
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
  if [ -f "$OUTLTX" ]; then
    cp -n "$OUTLTX" "$base.ltx"
    md5sum "$base.ltx" | sed "s|$ARCHIVE/||" >> "$base.md5"
  else
    echo "*** archive: $OUTLTX missing -- no ILA in this bitstream"
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
  { echo "# Clock sweep (same arch, different soc_clk / HP0 clock). Appended to, never rewritten."
    echo "# Area INCLUDES the ILA. WNS is over ALL clocks; at soc_mhz>50 a negative WNS is"
    echo "# expected on the CVA6<->Ara path -- do not trust measurements from TIMING_FAIL rows."
    printf "%-24s %4s %4s %-12s %9s %8s %7s %6s %5s %-30s %s\n" \
      NAME SOC HP0 PROBES LUT ILA_LUT WNS_ns FEP MIN LTX STATUS
  } > "$SUMMARY"
}
summary_row() {  # <name> <soc> <hp0> <probes> <mins> <ltx> <status>
  printf "%-24s %4s %4s %-12s %9s %8s %7s %6s %5s %-30s %s\n" \
    "$1" "$2" "$3" "$4" "${LUT:--}" "${ILA_LUT:--}" "${WNS:--}" "${FEP:--}" "$5" "$6" "$7" >> "$SUMMARY"
}

rescue_orphan() {
  [ -f "$OUTBIT" ] || return 0
  already_archived "$OUTBIT" && { echo "[rescue] out/ bitstream already archived -- nothing to do."; return 0; }
  local tag base
  tag="UNKNOWNCFG_clocksweep_$(date -r "$OUTBIT" +%Y%m%d_%H%M)"
  base=$(unique_base "$tag")
  echo "[rescue] unarchived bitstream in out/ -> $base.bit  (read ${base}_add_sources.tcl to see what it is)"
  archive_artifacts "$base" || { echo "[rescue] !! copy FAILED -- stopping, your bitstream is still in out/"; exit 1; }
}

run_one() {
  local name=$1 lanes=$2 vlen=$3 axi=$4 soc=$5 hp0=$6
  local date; date=$(date +%Y%m%d)
  local want; want=$(norm_probes "$PROBES")
  local tag; tag=$(mktag "$lanes" "$vlen" "$axi" "$soc" "$hp0" "$PROBES" "$date")
  local log=$ARCHIVE/build_${name}_$(probetag "$PROBES")_${date}.log
  local n=2; while [ -e "$log" ]; do log=${log%.log}; log=${log%_r[0-9]*}_r$n.log; n=$((n+1)); done
  local hp0arg=""; [ "$hp0" != "-" ] && hp0arg=$hp0

  echo ""
  echo "=============================================================="
  echo " BUILD $name : lanes=$lanes vlen=$vlen soc=${soc}MHz hp0=${hp0arg:-soc} probes=[$want]"
  echo " started  $(date '+%F %T')    log -> $log"
  echo "=============================================================="
  rescue_orphan
  rm -f "$OUTBIT" "$OUTLTX"      # a failed build must not leave a stale .bit/.ltx behind

  local t0; t0=$(date +%s)
  make ara-chs-xilinx BOARD=$BOARD MODE=batch VIVADO="$VIVADO" \
       nr_lanes="$lanes" vlen="$vlen" llc="$LLC" fpu="$FPU" probes="$PROBES" \
       soc_mhz="$soc" hp0_mhz="$hp0arg" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  local mins=$(( ($(date +%s) - t0) / 60 ))
  summary_header

  if [ $rc -ne 0 ] || [ ! -f "$OUTBIT" ]; then
    echo "*** $name FAILED (rc=$rc)"; LUT=; ILA_LUT=; WNS=; FEP=
    summary_row "$name" "$soc" "${hp0arg:-=}" "$want" "$mins" "-" "FAILED"; return 1
  fi

  # --- did the requested configuration, clocks AND probes reach the build? ---
  local status=OK gl gv gs gh gp
  gl=$(built_val NR_LANES); gv=$(built_val VLEN); gp=$(built_probes)
  gs=$(built_val SOC_MHZ);  gs=${gs:-50}          # no define = original clkwiz = 50
  gh=$(built_val HP0_MHZ)
  [ "$gl" != "$lanes" ] || [ "$gv" != "$vlen" ] && { echo "*** CONFIG MISMATCH: NR_LANES=$gl VLEN=$gv"; status="$status,CONFIG_MISMATCH"; }
  [ "$gs" != "$soc" ] || [ "${gh:--}" != "$hp0" ] && { echo "*** CLOCK MISMATCH: built SOC_MHZ=$gs HP0_MHZ=${gh:--}"; status="$status,CLOCK_MISMATCH"; }
  [ "$gp" != "$want" ] && { echo "*** PROBE MISMATCH: asked [$want], add_sources has [$gp]"; status="$status,PROBE_MISMATCH"; }
  # the clocks must also be in the implemented design, not just in the defines
  has_clock_mhz "$REPORTS/timing_summary.rpt" "$soc" || { echo "*** no ${soc} MHz clock in the timing report"; status="$status,NO_${soc}MHZ_CLOCK"; }
  [ "$hp0" != "-" ] && { has_clock_mhz "$REPORTS/timing_summary.rpt" "$hp0" || { echo "*** no ${hp0} MHz clock in the timing report"; status="$status,NO_${hp0}MHZ_CLOCK"; }; }

  local lc; lc=$(ltx_counts "$OUTLTX")
  echo ">>> ltx: $lc"
  local g
  for g in mem ddr vec; do
    [[ " $want " == *" $g "* ]] && [ "$(echo "$lc" | sed -E "s/.*$g=([0-9]+).*/\1/")" -eq 0 ] && status="$status,$(echo $g | tr a-z A-Z)_PROBES_ABSENT"
  done
  [ -n "$want" ] && [ ! -f "$OUTLTX" ] && status="$status,NO_LTX"

  local base; base=$(unique_base "$tag")
  archive_artifacts "$base" || { echo "*** $name ARCHIVE FAILED -- bitstream still in $OUTBIT, stopping."; exit 1; }
  mv -n "$log" "${base}_build.log" 2>/dev/null

  check_width_warnings; read_metrics
  [ "${FEP:-0}" != "0" ] && status="$status,TIMING_FAIL(${FEP}ep)"
  [ "${WIDTHWARN:-0}" -gt 0 ] && status="$status,WIDTHWARN($WIDTHWARN)"
  summary_row "$name" "$gs" "${gh:-=}" "$want" "$mins" "$lc" "$status"
  echo ">>> $name done in ${mins} min: LUT=$LUT (ILA $ILA_LUT) WNS=${WNS}ns [$status]"
  echo ">>> archived: $base.bit  +  $base.ltx"
}

do_list() {
  echo "Archive: $ARCHIVE    PROBES=$(norm_probes "$PROBES")  FPU=$FPU  LLC=$LLC"
  printf "%-24s %5s %6s %4s %4s  %s\n" NAME LANES VLEN SOC HP0 ARCHIVED
  for c in "${CASES[@]}"; do
    read -r name lanes vlen axi soc hp0 <<< "$c"
    local h; h=$(ls $(archived_glob "$lanes" "$vlen" "$axi" "$soc" "$hp0") 2>/dev/null | wc -l)
    printf "%-24s %5s %6s %4s %4s  %s\n" "$name" "$lanes" "$vlen" "$soc" "$hp0" "$([ "$h" -gt 0 ] && echo "yes ($h)" || echo no)"
  done
}

WANT=("$@")
case "${1:-}" in
  --list|-l)   do_list; exit 0 ;;
  --rescue|-r) rescue_orphan; exit 0 ;;
  --help|-h)   sed -n '2,38p' "$0"; exit 0 ;;
esac
for w in "${WANT[@]}"; do   # a typo in a case name must not silently build nothing
  printf '%s\n' "${CASES[@]}" | awk '{print $1}' | grep -qx "$w" || { echo "*** unknown case '$w' (see --list)"; exit 1; }
done

VIVADO_CMD=${VIVADO%% *}
command -v "$VIVADO_CMD" >/dev/null 2>&1 || { echo "*** '$VIVADO_CMD' not on PATH (VIVADO=$VIVADO)"; exit 1; }

rescue_orphan
fail=0; built=0; skipped=0
for c in "${CASES[@]}"; do
  read -r name lanes vlen axi soc hp0 <<< "$c"
  if [ ${#WANT[@]} -gt 0 ]; then m=0; for w in "${WANT[@]}"; do [ "$w" = "$name" ] && m=1; done; [ $m -eq 0 ] && continue; fi
  if [ "${FORCE:-0}" != "1" ] && ls $(archived_glob "$lanes" "$vlen" "$axi" "$soc" "$hp0") >/dev/null 2>&1; then
    echo "--- $name ($(probetag "$PROBES")) already archived, skipping (FORCE=1 to rebuild)"; skipped=$((skipped+1)); continue
  fi
  run_one "$name" "$lanes" "$vlen" "$axi" "$soc" "$hp0" || fail=$((fail+1))
  built=$((built+1))
done

echo ""; echo "=============================== SUMMARY ==============================="
cat "$SUMMARY" 2>/dev/null
echo "built=$built skipped=$skipped failed=$fail    archive: $ARCHIVE"
[ $fail -gt 0 ] && exit 1
exit 0
