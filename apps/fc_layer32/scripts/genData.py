import numpy as np
import os

# --- Dimensions ---
INPUT_SIZE = 2048
OUTPUT_SIZE = 512
NUM_SAMPLES = 4  # 2 batches of 2 samples = 4 total samples

# Fixed seed so you get the same numbers every time you run it
np.random.seed(42)

output_dir = "generated_data"
os.makedirs(output_dir, exist_ok=True)

print("Generating data...")

# 1. Generate Weights and Biases (Float32)
# Weights: Random values between -0.1 and 0.1
# Bias: All 1.00 as requested
W = np.random.uniform(-0.1, 0.1, (INPUT_SIZE * OUTPUT_SIZE)).astype(np.float32)
B = np.ones(OUTPUT_SIZE, dtype=np.float32)

# 2. Generate Inputs (Float32)
# Random floats between -1.0 and 1.0
X = np.random.uniform(-1.0, 1.0, (NUM_SAMPLES, INPUT_SIZE)).astype(np.float32)

# 3. Generate Labels/Predictions (Int32)
# For Cross Entropy, we need one integer class label per sample (0 to 511)
labels = np.random.randint(0, OUTPUT_SIZE, (NUM_SAMPLES,)).astype(np.int32)


# --- 1. Export to Binary Files (.bin) for .incbin ---
print("Exporting binary files for .incbin...")
# .tofile() writes raw C-compatible memory bytes to disk
X.tofile(os.path.join(output_dir, "inputs_data.bin"))
labels.tofile(os.path.join(output_dir, "labels_data.bin"))


# --- 2. Export Weights to C File (weights.c) ---
print("Exporting weights.c (this may take a few seconds for 1M+ parameters)...")
c_file_path = os.path.join(output_dir, "weights.c")

with open(c_file_path, "w") as f:
    f.write("// Auto-generated Fully Connected Layer Weights\n\n")
    
    # Write Weights Array
    f.write(f"float fc1_weights[{INPUT_SIZE * OUTPUT_SIZE}] = {{\n")
    # Write in chunks of 8 values per line for readability
    for i in range(0, len(W), 8):
        chunk = W[i:i+8]
        # Append 'f' to ensure C compiler treats them as single-precision floats
        line = ", ".join([f"{val:.6f}f" for val in chunk])
        if i + 8 >= len(W):
            f.write(f"    {line}\n")
        else:
            f.write(f"    {line},\n")
    f.write("};\n\n")

    # Write Bias Array
    f.write(f"float fc1_bias[{OUTPUT_SIZE}] = {{\n")
    for i in range(0, len(B), 8):
        chunk = B[i:i+8]
        line = ", ".join([f"{val:.2f}f" for val in chunk])
        if i + 8 >= len(B):
            f.write(f"    {line}\n")
        else:
            f.write(f"    {line},\n")
    f.write("};\n")

print(f"Success! Files saved in '{output_dir}'")