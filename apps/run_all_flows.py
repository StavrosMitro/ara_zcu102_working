#!/usr/bin/env python3
"""
Run the Ara benchmark sweeps SEQUENTIALLY, one Python flow after another.

Why sequential (and not parallel):
  run_ara_flow.py (conv) and run_fc_layer.py (FC) share three build resources
  that are NOT config-scoped, so running them at the same time corrupts results:
    1. the Verilator model dir  hardware/build/verilator/  (one model on disk;
       whoever verilates last wins, the other simv silently uses the wrong VLEN)
    2. common/link.ld           (regenerated from $(nr_lanes) on every app build)
    3. common/*-llvm.o          (shared runtime objects carrying -DNR_LANES/-DVLEN)
  Each flow already forces a full per-config clean+rebuild internally, so running
  them back-to-back is correct: the second flow rebuilds everything it needs.

This wrapper just launches each flow as its own process, streams its output
live, times it, and prints a summary. It adds no build logic of its own -- all
the sweep axes (lanes, vlen, apps, dims) live inside each flow script.

Usage:
  python3 run_all_flows.py                 # conv sweep, then FC sweep
  python3 run_all_flows.py --stop-on-fail  # abort if a flow exits non-zero
  python3 run_all_flows.py --only fc       # run just one flow (substring match)
"""
import argparse
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordered list of flows to run. Conv first (longer per-config verilation is the
# same either way; order is arbitrary), then FC. Edit here to add a flow.
FLOWS = [
    ("conv (conv_layer32 / conv_layer16only)", "run_ara_flow.py"),
    ("fc   (fc_layer32 / fc_layer16only)",     "run_fc_layer.py"),
]


def run_flow(label, script):
    """Run one flow script as a child process, streaming its stdio live.

    Returns (return_code, elapsed_seconds). stdout/stderr are inherited so the
    child's own live-streaming (it already uses stdbuf -oL) reaches the terminal
    unbuffered without this wrapper re-buffering it.
    """
    path = os.path.join(BASE_DIR, script)
    if not os.path.isfile(path):
        print(f"❌ {label}: {path} not found — skipping.")
        return (127, 0.0)

    print("\n" + "=" * 72)
    print(f">>> ▶️  STARTING FLOW: {label}")
    print(f">>>    {sys.executable} {script}")
    print("=" * 72 + "\n", flush=True)

    t0 = time.time()
    # Same interpreter as this wrapper; cwd = apps/ so the child's __file__-based
    # paths and its relative make invocations resolve exactly as when run alone.
    proc = subprocess.run([sys.executable, path], cwd=BASE_DIR)
    elapsed = time.time() - t0

    status = "✅ OK" if proc.returncode == 0 else f"❌ FAILED (rc={proc.returncode})"
    print(f"\n<<< {status}: {label}  [{elapsed/60:.1f} min]")
    return (proc.returncode, elapsed)


def main():
    ap = argparse.ArgumentParser(description="Run Ara benchmark flows sequentially.")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="Abort the remaining flows if one exits non-zero.")
    ap.add_argument("--only", metavar="SUBSTR", default=None,
                    help="Run only flows whose label or script matches SUBSTR.")
    args = ap.parse_args()

    flows = FLOWS
    if args.only:
        flows = [f for f in FLOWS if args.only in f[0] or args.only in f[1]]
        if not flows:
            print(f"❌ --only '{args.only}' matched no flow. Known: "
                  f"{[s for _, s in FLOWS]}")
            sys.exit(2)

    print("=" * 72)
    print("      ARA BENCHMARK — SEQUENTIAL FLOW RUNNER")
    print(f"      flows: {', '.join(s for _, s in flows)}")
    print("=" * 72)

    results = []
    wall0 = time.time()
    for label, script in flows:
        rc, elapsed = run_flow(label, script)
        results.append((label, rc, elapsed))
        if rc != 0 and args.stop_on_fail:
            print(f"\n⛔ --stop-on-fail: aborting after '{label}'.")
            break

    total = time.time() - wall0
    print("\n" + "=" * 72)
    print("      SUMMARY")
    print("=" * 72)
    for label, rc, elapsed in results:
        mark = "✅" if rc == 0 else "❌"
        print(f"  {mark}  {label:48s} rc={rc:<4d} {elapsed/60:6.1f} min")
    print(f"\n  total wall time: {total/60:.1f} min")

    failures = sum(1 for _, rc, _ in results if rc != 0)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
