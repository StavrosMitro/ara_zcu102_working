import os
import subprocess
import re
import csv
import sys
import glob

# ==========================================
# CONFIGURATION  — edit the lists/knobs here, everything below is derived
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "apps"))
CONFIG_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "config"))

# Apps to benchmark — mixed-precision (fp16 in/weights, fp32 accum) vs pure fp32
SIM_APPS = ['fc_layer16only' , 'fc_layer32']

# --- Sweep axes -------------------------------------------------------------
LANES_LIST = [2, 4]
VLEN_BYTES_LIST = [64, 128, 256]

# --- App build knobs (passed as make flags, never left to Makefile defaults) -
#
# BATCHSIZE must match the number of samples baked into generated_data/*.bin.
# You do NOT need to regenerate the data by hand: apps/Makefile regenerates it
# from a stamp whenever SAMPLES/IN_UNITS/OUT_UNITS/SEED change (the *_DATA_STAMP
# rule), and app_make_flags() below passes those same values, so the .bin files
# and the compiled binary can never disagree.
#
# IN/OUT default to 256 (not 128): on the m4 fmatmul path VLMAX(e16) = 2*VL_bytes,
# so 256 keeps FP16 fully lane-occupied through VL=128 B and yields the honest
# 2:1 strip ratio vs FP32. At 128, FP16 idles half its lanes from 128 B upward.
# (For a VL=256 B point you would need 512, which is also FC_MAX_INTERNAL.)
BATCHSIZE = 4
EPOCHS = 1
SAMPLES = BATCHSIZE          # FC_TOTAL_SAMPLES; must be <= samples in the .bin files
IN_UNITS = 256               # FC layer input  units (FC_*_IN_UNITS)
OUT_UNITS = 256              # FC layer output units (FC_*_OUT_UNITS)
SEED = 42                    # data-generation seed; keep fixed across the pair
LOSS = 'MSE'                 # MSE | CEL

# Per-app Makefile variable prefixes. Everything is derived from these so adding
# an app is a one-line change.
APP_VAR_PREFIX = {
    'fc_layer16only': 'FC_LAYER16ONLY',
    'fc_layer32':     'FC_LAYER32',
}

# --- Run behaviour ----------------------------------------------------------
USE_NOHUP = True             # capture sim output via nohup (lab servers)
KEEP_NOHUP_LOG = True        # keep per-run log; needed to debug "no metrics"
OVERWRITE_CSV = True         # fresh CSV per run
FORCE_CLEAN_OBJECTS = True   # see clean_app_objects() for why this matters
UNIQUE_CONFIG_PER_PID = True # avoid clashes between concurrent sweeps

# Per-app, per-run output file name template
OUTPUT_CSV_TEMPLATE = "ara_{app}_l{lanes}_vlen{vlen_bytes}_batch{batch}_FP16.csv"

PID = os.getpid()

# ==========================================
# MAKE FLAG CONSTRUCTION
# ==========================================

def hw_make_flags(config_name, lanes, vlen_bits):
    """
    Flags for the hardware/ Makefile (verilate, simv).

    config= is what actually drives re-verilation: the verilate rule depends on
    the config .mk file, so a fresh file per (lanes,vlen) is what makes make
    rebuild the model. nr_lanes/vlen are passed too so the values are explicit
    on the command line rather than only living inside the generated file.
    """
    return f"config={config_name} nr_lanes={lanes} vlen={vlen_bits}"


def app_make_flags(app, config_name, lanes, vlen_bits):
    """
    Flags for the apps/ Makefile.

    Every knob is passed explicitly. The batch size in particular MUST be on the
    command line: main.c.o is .INTERMEDIATE (rebuilt every time) while
    kernel/*.c.o are not, so if the two disagree about ALEXNET_STATIC_MAX_BATCH
    the binary aborts at startup.

    config= matters because nr_lanes feeds align_sections.sh, which sets the
    linker-script section alignment (4 * nr_lanes) baked into the ELF.
    (VLEN itself is unused by fc_layer/fc_layer32 — they are vector-length
    agnostic — but it is passed for consistency with apps that do use it.)
    """
    p = APP_VAR_PREFIX[app]
    return (
        f"{p}_BATCHSIZE={BATCHSIZE} "
        f"{p}_STATIC_MAX_BATCH={BATCHSIZE} "
        f"{p}_SAMPLES={SAMPLES} "
        f"{p}_EPOCHS={EPOCHS} "
        f"{p}_IN_UNITS={IN_UNITS} "
        f"{p}_OUT_UNITS={OUT_UNITS} "
        f"{p}_SEED={SEED} "
        f"{p}_LOSS={LOSS} "
        f"config={config_name} nr_lanes={lanes} vlen={vlen_bits}"
    )

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def write_config_file(lanes, vlen_bits, config_file_path):
    content = f"""# Auto-generated configuration for testing
nr_lanes ?= {lanes}
vlen ?= {vlen_bits}
"""
    try:
        with open(config_file_path, 'w') as f:
            f.write(content)
        return True
    except IOError as e:
        print(f"❌ Error writing config file: {e}")
        return False


def clean_app_objects(app):
    """
    Remove the app's compiled objects and linked binary.

    The apps Makefile has no dependency from *.c.o on the Makefile-supplied -D
    flags, so an object built with a different ALEXNET_BATCHSIZE /
    ALEXNET_STATIC_MAX_BATCH is silently reused. Only main.c.o is .INTERMEDIATE
    (always rebuilt), which yields a binary whose translation units disagree
    about the batch size -> "batchsize N exceeds static max batch M" -> exit(1)
    before a single vector instruction runs. Force a clean rebuild instead.
    """
    if not FORCE_CLEAN_OBJECTS:
        return
    app_dir = os.path.join(APPS_DIR, app)
    patterns = [
        os.path.join(app_dir, "*.c.o"),
        os.path.join(app_dir, "*.S.o"),
        os.path.join(app_dir, "kernel", "*.c.o"),
        os.path.join(app_dir, "kernel", "*.S.o"),
    ]
    removed = 0
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    for path in (os.path.join(APPS_DIR, "bin", app),
                 os.path.join(APPS_DIR, "bin", app + ".dump")):
        if os.path.exists(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    print(f"🧹 Cleaned {removed} stale object/binary files for {app}")


def run_command(command, cwd=None):
    """
    Runs a shell command and streams output immediately using stdbuf logic.
    Returns (output, returncode).
    """
    print(f"⚡ Running: {command}")

    unbuffered_command = f"stdbuf -oL -eL {command}"

    process = subprocess.Popen(
        unbuffered_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=cwd
    )

    captured_output = []

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured_output.append(line)

    rc = process.poll()
    if rc != 0:
        print(f"❌ Command failed with exit code {rc}: {command}")

    return "".join(captured_output), rc


def run_command_nohup(command, cwd=None, nohup_out=None):
    """
    Runs a shell command via nohup and captures output from nohup.out.
    """
    print(f"⚡ Running (nohup): {command}")
    print(f"📁 Log file path: {nohup_out}")

    if os.path.exists(nohup_out):
        os.remove(nohup_out)

    unbuffered_command = f"nohup stdbuf -oL -eL {command} > {nohup_out} 2>&1"

    process = subprocess.Popen(unbuffered_command, shell=True, cwd=cwd)
    process.wait()

    if not os.path.exists(nohup_out):
        print(f"❌ CRITICAL ERROR: The file {nohup_out} was STILL not created. Check folder permissions.")
        return ""

    with open(nohup_out, "r", encoding="utf-8", errors="replace") as f:
        output = f.read()

    if not KEEP_NOHUP_LOG:
        os.remove(nohup_out)

    return output


def sanity_check_output(output_text):
    """
    Surface the common failure modes instead of silently reporting 'no metrics'.
    """
    problems = []
    if "exceeds static max batch" in output_text:
        problems.append(
            "app aborted: batchsize > ALEXNET_STATIC_MAX_BATCH "
            "(stale objects with different -D flags, or BATCHSIZE too large)"
        )
    if "weight array shape mismatch" in output_text:
        problems.append("app aborted: weight array shape mismatch (regenerate weights.c)")
    if "[SCALAR]" in output_text:
        problems.append("kernel fell back to the scalar path (dims exceed FMATMUL_MAX_*)")
    m = re.search(r'^batch size:\s*(\d+)', output_text, re.MULTILINE)
    if m and int(m.group(1)) != BATCHSIZE:
        problems.append(f"app reports batch size {m.group(1)}, expected {BATCHSIZE}")
    if not problems:
        problems.append("simulation produced no 'cycles[epoch ...]' line — check the log")

    for p in problems:
        print(f"   ⚠️  {p}")


def parse_results(output_text, lanes, vlen_bytes):
    line_pattern = re.compile(
        r'^cycles\[epoch\s+(\d+)\s+batch\s+(\d+)/(\d+)\]:\s+(.+)$',
        re.MULTILINE
    )
    kv_pattern = re.compile(r'([a-zA-Z0-9_+\-]+)=([0-9]+)')

    breakdown_pattern = re.compile(r'^conv backward breakdown:\s+(.+)$', re.MULTILINE)

    parsed_data = []

    for match in line_pattern.finditer(output_text):
        epoch, batch, batch_total, kv_blob = match.groups()
        kvs = dict(kv_pattern.findall(kv_blob))

        breakdown_kvs = {}
        breakdown_match = breakdown_pattern.search(output_text, match.end())
        if breakdown_match:
            breakdown_kvs = dict(kv_pattern.findall(breakdown_match.group(1)))

        parsed_data.append({
            'Lanes': lanes,
            'VLEN (Bytes)': vlen_bytes,
            'VLEN (Bits)': vlen_bytes * 8,
            'Batchsize': BATCHSIZE,
            'Epoch': int(epoch),
            'Batch': int(batch),
            'Batch Total': int(batch_total),
            'Prep Cycles': kvs.get('prep', ''),
            'Forward Cycles': kvs.get('forward', ''),
            'Pred+Metric Cycles': kvs.get('pred+metric', ''),
            'Loss Cycles': kvs.get('loss', ''),
            'Zero D Input Cycles': kvs.get('zero_d_input', ''),
            'Backward Cycles': kvs.get('backward', ''),
            'Update Cycles': kvs.get('update', ''),
            'D Input Cycles': breakdown_kvs.get('d_input', kvs.get('backward_d_input', kvs.get('d_input', ''))),
            'D Bias Cycles': breakdown_kvs.get('d_bias', kvs.get('backward_d_bias', kvs.get('d_bias', ''))),
            'D Weights Im2col Cycles': breakdown_kvs.get('d_weights_im2col', ''),
            'D Weights Total Cycles': breakdown_kvs.get('d_weights_total', kvs.get('backward_d_weights', kvs.get('d_weight', ''))),
            'Total Backward Cycles': breakdown_kvs.get('total', kvs.get('backward_total', '')),
            'Backward Wrapper Cycles': kvs.get('backward_wrapper', '')
        })

    return parsed_data


def append_to_csv(data, filename):
    file_exists = os.path.isfile(filename)
    fieldnames = [
        'Lanes', 'VLEN (Bytes)', 'VLEN (Bits)', 'Batchsize', 'Epoch', 'Batch', 'Batch Total',
        'Prep Cycles', 'Forward Cycles', 'Pred+Metric Cycles', 'Loss Cycles',
        'Zero D Input Cycles', 'Backward Cycles', 'Update Cycles', 'D Input Cycles',
        'D Bias Cycles', 'D Weights Im2col Cycles', 'D Weights Total Cycles',
        'Total Backward Cycles', 'Backward Wrapper Cycles'
    ]

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================

def main():
    print("========================================")
    print("    ARA AUTOMATED BENCHMARK SUITE")
    print(f"    PID={PID}  batch={BATCHSIZE}  epochs={EPOCHS}  samples={SAMPLES}  loss={LOSS}")
    print(f"    lanes={LANES_LIST}  vlen_bytes={VLEN_BYTES_LIST}")
    print(f"    apps={', '.join(SIM_APPS)}")
    print("========================================\n")

    unknown = [a for a in SIM_APPS if a not in APP_VAR_PREFIX]
    if unknown:
        print(f"❌ No Makefile variable prefix known for: {unknown}")
        sys.exit(1)

    for lanes in LANES_LIST:
        for vlen_bytes in VLEN_BYTES_LIST:
            vlen_bits = vlen_bytes * 8

            suffix = f"_{PID}" if UNIQUE_CONFIG_PER_PID else ""
            temp_config_name = f"autogen_test_l{lanes}_v{vlen_bytes}{suffix}"
            temp_config_file = os.path.join(CONFIG_DIR, f"{temp_config_name}.mk")

            if not write_config_file(lanes, vlen_bits, temp_config_file):
                sys.exit(1)

            print(f"\n{'='*70}\n=== VERILATING: lanes={lanes}, VLEN={vlen_bytes}B ({vlen_bits} bits)\n{'='*70}")
            _, rc = run_command(f"make verilate {hw_make_flags(temp_config_name, lanes, vlen_bits)}")
            if rc != 0:
                print(f"❌ Verilation failed for lanes={lanes} vlen={vlen_bits}. Skipping this config.")
                if os.path.exists(temp_config_file):
                    os.remove(temp_config_file)
                continue

            for sim_app in SIM_APPS:
                output_csv = os.path.join(
                    BASE_DIR,
                    OUTPUT_CSV_TEMPLATE.format(app=sim_app, lanes=lanes,
                                               vlen_bytes=vlen_bytes, batch=BATCHSIZE)
                )
                if OVERWRITE_CSV and os.path.exists(output_csv):
                    os.remove(output_csv)

                print(f"\n>>> 🚀 TEST [{sim_app}]: lanes={lanes}, VLEN={vlen_bytes}B, batch={BATCHSIZE} <<<")

                current_nohup = os.path.join(
                    BASE_DIR, f"nohup_{sim_app}_l{lanes}_v{vlen_bytes}_{PID}.out")

                # Build the app. `make simv` only loads the ELF, it never builds it.
                clean_app_objects(sim_app)
                build_cmd = f"make {sim_app} {app_make_flags(sim_app, temp_config_name, lanes, vlen_bits)}"
                _, rc = run_command(build_cmd, cwd=APPS_DIR)
                if rc != 0:
                    print(f"❌ Build failed for {sim_app}. Skipping this run.")
                    continue

                # Verify the ELF actually landed before handing it to simv. `make`
                # can return 0 without producing bin/<app> (e.g. an all-goals-up-to-
                # date no-op after a partial clean), and simv would then fail with a
                # bare "Failed to load ELF" that looks identical to a crashed run.
                app_bin = os.path.join(APPS_DIR, "bin", sim_app)
                if not os.path.isfile(app_bin):
                    print(f"❌ Build reported success but {app_bin} is missing. Skipping this run.")
                    continue

                sim_cmd = f"make simv app={sim_app} {hw_make_flags(temp_config_name, lanes, vlen_bits)}"
                if USE_NOHUP:
                    sim_output = run_command_nohup(sim_cmd, nohup_out=current_nohup)
                else:
                    sim_output, _ = run_command(sim_cmd)

                # A failed ELF load or an RTL crash yields NO 'cycles[...]' line,
                # which is indistinguishable from a silent success unless we look
                # for it explicitly (mirrors run_ara_flow.py's guard).
                if re.search(r'Failed to load ELF|Segmentation fault|Assertion.*failed',
                             sim_output):
                    print(f"❌ SIMULATION FAILED ({sim_app}, lanes={lanes}, vlen={vlen_bits}):")
                    print(sim_output[-1500:])
                    if KEEP_NOHUP_LOG:
                        print(f"   📄 Full simulation log: {current_nohup}")
                    continue

                metrics = parse_results(sim_output, lanes, vlen_bytes)
                if metrics:
                    print(f"✅ Found {len(metrics)} data points. Saving to {output_csv}")
                    append_to_csv(metrics, output_csv)
                else:
                    print("⚠️  No metrics found.")
                    sanity_check_output(sim_output)
                    if KEEP_NOHUP_LOG:
                        print(f"   📄 Full simulation log: {current_nohup}")

            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)

    print("\n\n========================================")
    print("🎉 ALL TESTS COMPLETED.")
    print(f"📊 Results saved in: {os.path.abspath(BASE_DIR)}")
    print("========================================")


if __name__ == "__main__":
    main()
