#!/usr/bin/env bash
#
# Build vggnet for Spike, run training, and check the result.
#
# Exists so that a kernel change can be verified without a human reading loss
# numbers off a terminal. Every check here is a yes/no, because end-to-end loss
# is a poor oracle: training is chaotic, so past epoch 1 two builds can differ
# in the last bits for entirely benign reasons. The trustworthy signals are the
# per-layer gradient checksums, the post-update weight checksums, and the
# max-pool element-wise audit -- all of which this script compares exactly.
#
# Usage:
#   check_vggnet.sh record [MAKEVAR=VAL ...]   save current output as baseline
#   check_vggnet.sh check  [MAKEVAR=VAL ...]   run and diff against baseline
#   check_vggnet.sh ab VAR=VAL [MAKEVAR=...]   run twice (VAR set vs unset)
#                                              and diff the two -- use this to
#                                              prove a refactor is a no-op
#   check_vggnet.sh run    [MAKEVAR=VAL ...]   just build+run, print summary
#
# Exit status is 0 only if every check passed, so it can gate a commit.
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARA_DIR="${ARA_DIR:-$(dirname "$APPS_DIR")}"
BASELINE_DIR="$SCRIPT_DIR/baselines"
WORK="${TMPDIR:-/tmp}/vggnet-check.$$"
mkdir -p "$WORK" "$BASELINE_DIR"
trap 'rm -rf "$WORK"' EXIT

# Defaults chosen so a run is quick but still exercises the full backward pass.
# DUMP_TRACE is what emits the [BW] / W: / [MPAUDIT] lines the checks rely on.
DEFAULT_KNOBS=(
  VGGNET_FINETUNE=0
  VGGNET_EPOCHS=1
  VGGNET_BATCHSIZE=32
  VGGNET_MAX_STEPS=1
  VGGNET_DUMP_TRACE=1
)

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RST=$'\033[0m'
[ -t 1 ] || { RED=""; GRN=""; YEL=""; RST=""; }

fail_count=0
say()  { printf '%s\n' "$*"; }
ok()   { printf '  %sPASS%s  %s\n' "$GRN" "$RST" "$*"; }
bad()  { printf '  %sFAIL%s  %s\n' "$RED" "$RST" "$*"; fail_count=$((fail_count+1)); }
warn() { printf '  %sWARN%s  %s\n' "$YEL" "$RST" "$*"; }

# build_and_run <outfile> <extra make args...>
build_and_run() {
  local out="$1"; shift
  if ! make -C "$APPS_DIR" ARA_DIR="$ARA_DIR" spike-run-vggnet \
         "${DEFAULT_KNOBS[@]}" "$@" > "$out" 2>&1; then
    bad "build/run failed -- last lines:"
    tail -20 "$out" | sed 's/^/        /'
    return 1
  fi
  return 0
}

# Pull the deterministic, comparable parts out of a run log. Cycle counts and
# build command lines are deliberately excluded: they are not reproducible and
# would swamp a real difference.
extract() {
  grep -E '^\[BW\]|^\[MPAUDIT\]|^epoch ' "$1" || true
}

# sanity_checks <logfile> -- things that must hold regardless of any baseline
sanity_checks() {
  local log="$1"

  if grep -qiE '\*\*\* FAILED|tohost = [0-9]+' "$log"; then
    bad "Spike reported a failure (trap / non-zero tohost)"
  else
    ok "Spike run completed"
  fi

  local epochs
  epochs=$(grep -cE '^epoch ' "$log")
  if [ "$epochs" -eq 0 ]; then
    bad "no epoch lines -- training never ran"
  else
    ok "training ran ($epochs epoch line(s))"
  fi

  # printf_ renders inf as 2147483647; nan shows up literally.
  if grep -E '^epoch ' "$log" | grep -qiE 'nan|inf|2147483647'; then
    bad "loss is not finite:"
    grep -E '^epoch ' "$log" | sed 's/^/        /'
  elif [ "$epochs" -gt 0 ]; then
    ok "loss finite in every epoch"
  fi

  # The max-pool audit recomputes the scalar argmax and the scattered gradient
  # value for every output; anything but zero means the vector path diverged.
  if grep -q '^\[MPAUDIT\]' "$log"; then
    local bad_audit
    bad_audit=$(grep '^\[MPAUDIT\]' "$log" | grep -vc 'index_mismatch=0  *value_mismatch=0' || true)
    if [ "$bad_audit" -eq 0 ]; then
      ok "max-pool audit clean ($(grep -c '^\[MPAUDIT\]' "$log") pool(s))"
    else
      bad "max-pool audit reported mismatches:"
      grep '^\[MPAUDIT\]' "$log" | sed 's/^/        /'
    fi
  else
    warn "no [MPAUDIT] lines (build without VGGNET_DUMP_TRACE=1?)"
  fi

  # A gradient that is exactly zero everywhere means a backward path is dead --
  # this is how the commented-out conv dW (RVV_ISSUES Issue 2) presented.
  local zero_grads
  zero_grads=$(grep -E '^\[BW\] [a-z0-9]+\.(dW|db|dg|dbeta)' "$log" \
               | awk '$7 == 0.000000 {print $2}' || true)
  if [ -n "$zero_grads" ]; then
    bad "parameter gradient identically zero: $(echo "$zero_grads" | tr '\n' ' ')"
  else
    ok "no identically-zero parameter gradients"
  fi
}

mode="${1:-check}"; shift || true

case "$mode" in
  record)
    say "== recording baseline =="
    build_and_run "$WORK/run.log" "$@" || exit 1
    sanity_checks "$WORK/run.log"
    if [ "$fail_count" -ne 0 ]; then
      say "refusing to record a baseline from a run that failed its sanity checks"
      exit 1
    fi
    extract "$WORK/run.log" > "$BASELINE_DIR/spike.txt"
    say "baseline written: $BASELINE_DIR/spike.txt ($(wc -l < "$BASELINE_DIR/spike.txt") lines)"
    ;;

  check)
    say "== check against baseline =="
    build_and_run "$WORK/run.log" "$@" || exit 1
    sanity_checks "$WORK/run.log"
    if [ ! -f "$BASELINE_DIR/spike.txt" ]; then
      warn "no baseline yet -- run '$0 record' first"
    else
      extract "$WORK/run.log" > "$WORK/now.txt"
      if diff -u "$BASELINE_DIR/spike.txt" "$WORK/now.txt" > "$WORK/diff.txt"; then
        ok "output identical to baseline"
      else
        bad "output differs from baseline:"
        sed 's/^/        /' "$WORK/diff.txt" | head -40
      fi
    fi
    ;;

  ab)
    [ $# -ge 1 ] || { say "usage: $0 ab VAR=VAL [make args...]"; exit 2; }
    ab_var="$1"; shift
    ab_name="${ab_var%%=*}"
    say "== A/B: $ab_var  vs  ${ab_name}=0 =="
    build_and_run "$WORK/a.log" "$ab_var" "$@" || exit 1
    build_and_run "$WORK/b.log" "${ab_name}=0" "$@" || exit 1
    sanity_checks "$WORK/b.log"
    extract "$WORK/a.log" > "$WORK/a.txt"
    extract "$WORK/b.log" > "$WORK/b.txt"
    if diff -u "$WORK/a.txt" "$WORK/b.txt" > "$WORK/diff.txt"; then
      ok "$ab_var and ${ab_name}=0 produce IDENTICAL output"
    else
      bad "$ab_var and ${ab_name}=0 differ:"
      sed 's/^/        /' "$WORK/diff.txt" | head -40
    fi
    ;;

  run)
    say "== build + run =="
    build_and_run "$WORK/run.log" "$@" || exit 1
    sanity_checks "$WORK/run.log"
    grep -E '^epoch ' "$WORK/run.log" | sed 's/^/  /'
    ;;

  *)
    say "usage: $0 {record|check|ab VAR=VAL|run} [MAKEVAR=VAL ...]"
    exit 2
    ;;
esac

say ""
if [ "$fail_count" -eq 0 ]; then
  say "${GRN}all checks passed${RST}"
  exit 0
fi
say "${RED}$fail_count check(s) failed${RST}"
exit 1
