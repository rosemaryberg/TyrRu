import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter

# -------------------------
# 1. SETUP & PARAMETERS
# -------------------------
Directory = r"C:\Users\bergi\OneDrive - UTS\Onedrive\OneDrive - UTS\Tyr-Ru paper\paper figures\SRR\new SRR"
horz_file = os.path.join(Directory, "2. B1 1.1000 10min Vert.csv")
vert_file = os.path.join(Directory, "3. B1 1.1000 10min Hort (TtB).csv")

# Experimental Parameters (Define the Aspect Ratio)
Speed = 20  # µm/sec (v)
Scantime = 0.05 #1 / 200  # 0.005 sec (tsc)
SpotSize = 1  # µm (D)
Layers = 2

# --- CALCULATIONS BASED ON PAPER'S FORMULAS ---

# Lateral Sampling Interval (Δx)
Delta_x = Speed * Scantime  # 20 * 0.005 = 0.1 µm

# Expansion Factor / Aspect Ratio (M)
# M is the factor by which the raw image must be expanded for SRR: M = Spot Size / Δx
M = SpotSize / Delta_x  # 1.0 / 0.1 = 10.0
M_int = int(M)

# Shift Offset (N)
# N is the shift required (half the magnitude of the spot size): N = M / 2
N = M_int // 2  # 10 // 2 = 5 pixels
print(M_int, N)

# -------------------------
# 2. LOAD RAW DATA (FIXED FOR PERSISTENT BOM ERROR)
# -------------------------
print("--- LOADING DATA ---")

# Step 1: Read the file, ensuring the encoding is handled.
# If 'utf-8' didn't work, try reading the file content directly and stripping the BOM
with open(horz_file, 'r', encoding='utf-8') as f:
    content_H = f.read()

# Manually strip the BOM character from the beginning of the file content
if content_H.startswith('\ufeff'):
    content_H = content_H.lstrip('\ufeff')

# Now, read the cleaned content using StringIO
import io
H_df = pd.read_csv(io.StringIO(content_H), sep=None, engine='python', header=None)

# Repeat for the Vertical file (V)
with open(vert_file, 'r', encoding='utf-8') as f:
    content_V = f.read()

if content_V.startswith('\ufeff'):
    content_V = content_V.lstrip('\ufeff')

V_df = pd.read_csv(io.StringIO(content_V), sep=None, engine='python', header=None)


# Step 2: Convert to NumPy array
# This step is now safer as the BOM has been stripped from the file start.
H = H_df.values.astype(np.float64)
V = V_df.values.astype(np.float64)

print(f"Loaded H Shape: {H.shape}")
print(f"Calculated Aspect Ratio (M): {M}")
# print("--- LOADING DATA ---")
# # Robust CSV loading (sep=None) is critical to prevent the '710x1' error
# H = pd.read_csv(horz_file, sep=None, engine='python', header=None).values
# V = pd.read_csv(vert_file, sep=None, engine='python', header=None).values
# print(f"Loaded H Shape: {H.shape}")
# print(f"Calculated Aspect Ratio (M): {M}")


# -------------------------
# 3. CORE PROCESSING FUNCTION (Kronecker, Upsample, Shift/Pad)
# -------------------------
def process_layer(data, M_factor, shift_pixels, is_vertical):
    # A. Kronecker Product (Expands width/columns by M)
    data_exp = np.kron(data, np.ones((1, M_factor)))

    # B. Upsample (Expands height/rows by M to create square pixels)
    data_upsampled = np.repeat(data_exp, M_factor, axis=0)

    # C. Shift and Pad (Aligns the orthogonal scans)
    pad_width = shift_pixels

    if is_vertical:
        # Pre-Shift (Top/Left) for the orthogonal layer
        pad = ((pad_width, 0), (pad_width, 0))
    else:
        # Post-Shift (Bottom/Right) for the primary layer (to match size)
        pad = ((0, pad_width), (0, pad_width))

    # Pad with NaNs (null values) for interpolation
    data_shifted = np.pad(data_upsampled, pad, mode='constant', constant_values=np.nan)

    return data_shifted


# -------------------------
# 4. EXECUTE RECONSTRUCTION (Full Image Processing)
# -------------------------

# 4a. Process Layers
H_final = process_layer(H, M_int, N, is_vertical=False)
V_final = process_layer(V, M_int, N, is_vertical=True)

# 4b. Crop to Common Size (Handles 1-pixel differences from padding)
min_r = min(H_final.shape[0], V_final.shape[0])
min_c = min(H_final.shape[1], V_final.shape[1])
H_final = H_final[:min_r, :min_c]
V_final = V_final[:min_r, :min_c]

# 4c. Stack and Sum (Creates the final 2D image)
stack = np.dstack([H_final, V_final])
SRR_Image = np.nansum(stack, axis=2)

# 4d. Gaussian Smoothing (Mitigates square pixel artifacts)
# Sigma derived from FWHM relationship with Spot Size (M)
calculated_sigma = M / 2.355
Gaussian_Image = gaussian_filter(SRR_Image, sigma=calculated_sigma)

# -------------------------
# 5. SAVE OUTPUTS
# -------------------------
print(f"Saving SRR Image (Shape: {SRR_Image.shape})...")

# Use fixed-point formatting ('%.5f') for compatibility with Fiji/ImageJ
np.savetxt(os.path.join(Directory, "SRR_2D.csv"), SRR_Image, delimiter=",", fmt='%.5f')
np.savetxt(os.path.join(Directory, "SRR_2D_Gaussian.csv"), Gaussian_Image, delimiter=",", fmt='%.5f')

print("✅ Full image processed and saved.")