import numpy as np
from PIL import Image
import subprocess
import os
import torch
import torch.nn as nn

# -------------------------------------------------------
# Set TEST_MODE = True to run a tiny 4x4 visual test
# instead of loading the real image from disk.
# Set to False to run the full comparison with the C binary.
# -------------------------------------------------------
TEST_MODE = True

IMAGE_PATH = "/home/stavros/scripts/cifar10_images/cat/train_batch_2_04269.jpg"

def im2col_torch(img_array, patch_size=3, stride=1):
    """
    Perform im2col using PyTorch's Unfold.
    Input: img_array of shape (C, H, W), dtype=np.float32
    Returns: matrix of shape (C * patch_size * patch_size, num_patches)
             with patches in column-major order (x first, then y).
    """
    # Convert to torch tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, C, H, W)
    C, H, W = img_array.shape

    # Compute output dimensions
    out_h = (H - patch_size) // stride + 1
    out_w = (W - patch_size) // stride + 1

    # Unfold (produces row-major patches: (1, C*K*K, out_h*out_w))
    unfold = nn.Unfold(kernel_size=patch_size, dilation=1, padding=0, stride=stride)
    patches = unfold(img_tensor)  # (1, C*K*K, L) with L = out_h * out_w

    # Reshape to separate kernel dimensions and spatial grid
    patches = patches.view(1, C, patch_size, patch_size, out_h, out_w)

    # Permute to convert row-major -> column-major: swap out_h and out_w
    patches = patches.permute(0, 1, 2, 3, 5, 4).contiguous()  # (1, C, K, K, out_w, out_h)

    # Flatten back to (1, C*K*K, out_w*out_h) – now in column-major order
    patches = patches.view(1, C * patch_size * patch_size, -1)

    # Remove batch dimension and return as numpy array
    return patches.squeeze(0).numpy()

def python_im2col(image_path, patch_size=3, stride=1):
    # This function is no longer used – kept for reference
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.transpose(img_array, (2, 0, 1))
    return im2col_torch(img_array, patch_size, stride)

def test_im2col_visual():
    """
    Runs im2col on a hand-crafted 4x4 single-channel image (values 1..16)
    and prints every step so you can verify the result by eye.
    Also runs the C binary with -t and compares.
    """
    patch_size, stride = 2, 1
    img_array = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4)  # (C=1, H=4, W=4)
    C, H, W = img_array.shape

    out_h = (H - patch_size) // stride + 1  # 3
    out_w = (W - patch_size) // stride + 1  # 3

    print("=" * 55)
    print("TEST MODE: 4x4 image, 1 channel, values 1..16")
    print("=" * 55)
    print(f"\nInput image (channel 0, shape {H}x{W}):")
    print(img_array[0].astype(int))
    print(f"\nkernel={patch_size}  stride={stride}  out_w={out_w}  out_h={out_h}")
    print(f"{out_w*out_h} patches, each with {C*patch_size*patch_size} elements\n")

    # Use PyTorch im2col
    col_matrix = im2col_torch(img_array, patch_size, stride)  # shape (4, 9)

    # Print non-transposed matrix (rows = kernel elements, cols = patches)
    print("PyTorch im2col output (NOT transposed):")
    print("(rows=kernel elements, cols=patches)")
    header = "       " + "  ".join(f"p{i:02d}" for i in range(col_matrix.shape[1]))
    print(header)
    for e in range(col_matrix.shape[0]):
        row_str = "  ".join(f"{v:5.1f}" for v in col_matrix[e])
        print(f"elem {e}: {row_str}")

    # Transpose: rows=patches, cols=elements (matches C output order)
    py_T = col_matrix.T  # shape (9, 4)

    print("\nPyTorch im2col output (TRANSPOSED - rows=patches, cols=kernel elements):")
    header = "       " + "  ".join(f"el{e:02d}" for e in range(col_matrix.shape[0]))
    print(header)
    for p in range(py_T.shape[0]):
        row_str = "  ".join(f"{v:5.1f}" for v in py_T[p])
        print(f"Patch {p}: {row_str}")

    # --- Run C binary with -t and compare ---
    print("\n" + "-" * 55)
    if not os.path.exists("./im2col_test"):
        print("C binary './im2col_test' not found — skipping comparison.")
        return

    result = subprocess.run(["./im2col_test", "-t"], capture_output=True, text=True)
    if result.returncode != 0:
        print("C binary failed:", result.stderr)
        return

    print("\nC im2col output (from stderr — same pretty-print):")
    for line in result.stderr.splitlines():
        print(" ", line)

    c_flat = np.array([float(v) for v in result.stdout.split()], dtype=np.float32)
    c_matrix = c_flat.reshape(py_T.shape)

    print("\nComparison (rounded to 1 decimal):")
    if np.allclose(py_T, c_matrix, atol=1e-4):
        print("✅ Python (PyTorch) and C results are identical.")
    else:
        diff = np.abs(py_T - c_matrix)
        bad = np.argwhere(diff > 1e-4)
        print(f"❌ {len(bad)} element(s) differ:")
        for (r, c) in bad:
            print(f"   Patch {r}, el {c}: Python={py_T[r,c]:.4f}  C={c_matrix[r,c]:.4f}")
    print("=" * 55)

def main():
    if TEST_MODE:
        test_im2col_visual()
        return

    print(f"Εικόνα: {IMAGE_PATH}\n")

    # --- ΒΗΜΑ 1: Υπολογισμός Python & Transpose ---
    print("⏳ Υπολογισμός μέσω PyTorch...")
    py_matrix = python_im2col(IMAGE_PATH)

    # Transpose για να συγκρίνουμε τις 900 γραμμές με τις 27 στήλες
    py_transposed = py_matrix.T

    # --- ΒΗΜΑ 2: Λήψη αποτελεσμάτων από C ---
    print("⏳ Εκτέλεση του C script...")
    if not os.path.exists("./im2col_test"):
        print("❌ Σφάλμα: Δεν βρέθηκε το αρχείο './im2col_test'. Κάνε compile ξανά.")
        return

    result = subprocess.run(["./im2col_test"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Το C script απέτυχε:")
        print(result.stderr)
        return

    c_flat = np.array([float(val) for val in result.stdout.split()], dtype=np.float32)

    # Μετατρέπουμε το 1D αποτέλεσμα της C στις ίδιες διαστάσεις (900, 27)
    try:
        c_matrix = c_flat.reshape(py_transposed.shape)
    except ValueError:
        print(f"❌ Αναντιστοιχία μεγέθους: Η Python περιμένει {py_transposed.shape[0] * py_transposed.shape[1]} στοιχεία, αλλά η C επέστρεψε {c_flat.shape[0]}.")
        return

    print(f"\n📊 Διαστάσεις Πινάκων (Γραμμές=Patches, Στήλες=Στοιχεία): {py_transposed.shape}")

    # --- ΒΗΜΑ 3: Σύγκριση & Report ---
    py_rounded = np.round(py_transposed, 2)
    c_rounded = np.round(c_matrix, 2)

    # Βρίσκουμε πού υπάρχουν διαφορές
    diff_mask = py_rounded != c_rounded

    if not np.any(diff_mask):
        print("\n✅ ΕΠΙΤΥΧΙΑ! Όλες οι γραμμές και οι στήλες ταυτίζονται απόλυτα (στα 2 δεκαδικά).")
        return

    # --- MINI REPORT ---
    print("\n❌ ΑΠΟΤΥΧΙΑ! Βρέθηκαν διαφορές. Ακολουθεί το Mini Report:\n")
    print("-" * 65)

    rows, cols = np.where(diff_mask)
    unique_rows = np.unique(rows)

    print(f"Συνολικά βρέθηκαν διαφορές σε {len(unique_rows)} από τα {py_transposed.shape[0]} patches (γραμμές).\n")

    max_rows_to_print = 15
    for r in unique_rows[:max_rows_to_print]:
        cols_in_row = cols[rows == r]
        print(f"🔹 Γραμμή {r} (Patch {r}):")
        print(f"   Βρέθηκαν {len(cols_in_row)} διαφορετικά στοιχεία στις στήλες: {list(cols_in_row)}")

        for c in cols_in_row[:3]:
            print(f"   -> Στήλη {c}: Python = {py_rounded[r, c]:.2f} | C = {c_rounded[r, c]:.2f}")
        if len(cols_in_row) > 3:
            print(f"   -> ... (και άλλα {len(cols_in_row) - 3} στοιχεία)")
        print("")

    if len(unique_rows) > max_rows_to_print:
        print(f"... Το report σταμάτησε στις {max_rows_to_print} γραμμές για εξοικονόμηση χώρου.")

    print("-" * 65)
    print("💡 Tip: Αν τα νούμερα διαφέρουν ελάχιστα (π.χ. 0.01), τότε δεν φταίει η img2col, αλλά η διαφορά στη μέθοδο resize της Python (PIL) σε σχέση με την resize_image της C.")

if __name__ == "__main__":
    main()