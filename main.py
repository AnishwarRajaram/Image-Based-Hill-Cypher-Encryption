#!/usr/bin/env python3
"""
Two-Stage Hill Cipher + Arnold Scramble — Simple Script Version

Just edit the CONFIG section below and run:
  python main.py
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path
import os

# Local imports
from read_from_img import PaddedRGBImageLoader
from hill_cipher_engine import generate_key, encrypt_channel, decrypt_channel, _mod_det
from arnold_transform import ArnoldScrambler
from decryption_validator import DecryptionValidator


# ==================== CONFIGURATION ====================
# Edit these values to change behavior

# Paths
INPUT_IMAGE = Path("Images/Original/input.jpg")        # Image to encrypt/decrypt
OUTPUT_ENCRYPTED = Path("Images/Encrypted/encrypted.png")
OUTPUT_DECRYPTED = Path("Images/decrypted.jpg")
KEY_FILE = Path("Images/key.npy")             # Where to save/load the key

# Cryptographic parameters
N = 4                  # Hill block size / key matrix dimension (e.g., 4 for 4x4)
Q = 251                # Prime modulus (must be prime, ≤256 for uint8 pixels)
PRE_MULTIPLY = True    # Hill cipher: True = K@B, False = B@K
ARNOLD_ITERS = 5       # Arnold map iterations per stage
RANDOM_SEED = 42       # For reproducible key generation (None = random)

# Mode: set to "encrypt" or "decrypt"
MODE = "decrypt"       # Change to "decrypt" to run decryption pipeline
# =======================================================


def encrypt_tshc(r, g, b, K, n, q, pre, arnold_iters):
    """Two-Stage Hill + Arnold encryption per channel."""
    encrypted = []
    channel_names = ['RED', 'GREEN', 'BLUE']
    idx=0
    for ch in [r, g, b]:
        name = channel_names[idx]
        print(f"\n  --- Encrypting {name} channel ---")
         # Stage 1
        print(f"    Stage1 Hill...")
        h1 = encrypt_channel(ch, K, n, q, pre=pre)
        print(f"      After Hill1: max={h1.max()}")
        
        print(f"    Stage1 Arnold...")
        a1 = ArnoldScrambler.scramble(h1, iterations=arnold_iters)
        print(f"      After Arnold1: max={a1.max()}")
        
        # Stage 2
        print(f"    Stage2 Hill...")
        h2 = encrypt_channel(a1, K, n, q, pre=pre)
        print(f"      After Hill2: max={h2.max()}")
        
        print(f"    Stage2 Arnold...")
        a2 = ArnoldScrambler.scramble(h2, iterations=arnold_iters)
        print(f"      After Arnold2: max={a2.max()}")
        
        # Check for issues
        if a2.max() >= q:
            print(f"    ⚠️ WARNING: {channel_name} has values >= {q}!")
        
        encrypted.append(a2)
    
    return tuple(encrypted)


def save_rgb(r, g, b, path):
    """Save RGB channels as PNG."""
    img = np.dstack([r, g, b])
    img = np.clip(img, 0, 255).astype(np.uint8)
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(path)
    print(f"✓ Saved: {path}")


def run_encrypt():
    
    print(f"🔐 ENCRYPTING: {INPUT_IMAGE} → {OUTPUT_ENCRYPTED}")
    
    # Load image
    loader = PaddedRGBImageLoader(k=N, images_dir=INPUT_IMAGE.parent)
    r, g, b = loader.load(INPUT_IMAGE.name)
    print(f"  Original range - R: {r.min()}-{r.max()}, G: {g.min()}-{g.max()}, B: {b.min()}-{b.max()}")
    
    # Values must be 0-250 for Q=251 to work properly
    r = np.clip(r, 0, Q-1)
    g = np.clip(g, 0, Q-1)
    b = np.clip(b, 0, Q-1)
    
    print(f"  Clipped range - R: {r.min()}-{r.max()}, G: {g.min()}-{g.max()}, B: {b.min()}-{b.max()}")
    print(f"  NOTE: Pure white (255) becomes {Q-1} ({Q-1})")
    print(f"  Loaded: {loader.original_shape} → padded to {loader.padded_shape}")
    
    # Generate key
    rng = np.random.default_rng(RANDOM_SEED) if RANDOM_SEED is not None else None
    print(f"  Generating {N}×{N} key over F_{Q}...")
    K = generate_key(N, Q, rng=rng)
    print(f"  Key det mod {Q} = {_mod_det(K, Q)} ✓")
    
    # Encrypt
    print("  Running TSHC pipeline: Hill₁ → Arnold₁ → Hill₂ → Arnold₂")
    er, eg, eb = encrypt_tshc(r, g, b, K, N, Q, PRE_MULTIPLY, ARNOLD_ITERS)

    # ===== DEBUG: Check values BEFORE saving =====
    print(f"\n  BEFORE SAVING:")
    print(f"    R - max: {er.max()}, min: {er.min()}")
    print(f"    G - max: {eg.max()}, min: {eg.min()}")
    print(f"    B - max: {eb.max()}, min: {eb.min()}")
    print(f"    Values >= 251 in R: {(er >= 251).sum()}")
    # ============================================
    # Save outputs


    save_rgb(er, eg, eb, OUTPUT_ENCRYPTED)

    # ===== DEBUG: Check AFTER saving and reloading =====

    reloaded = Image.open(OUTPUT_ENCRYPTED)
    reloaded_array = np.asarray(reloaded)
    print(f"\n  AFTER RELOAD:")
    print(f"    Reloaded - max: {reloaded_array.max()}, min: {reloaded_array.min()}")
    print(f"    Values >= 251: {(reloaded_array >= 251).sum()}")


    np.save(KEY_FILE, K)
    print(f"  ⚠️  Key saved to {KEY_FILE} — keep this secure!\n")
    return K, loader.original_shape


def run_decrypt(K, original_shape=None):
    print(f"🔓 DECRYPTING: {OUTPUT_ENCRYPTED} → {OUTPUT_DECRYPTED}")
    
    # Load encrypted image
    loader = PaddedRGBImageLoader(k=N, images_dir=OUTPUT_ENCRYPTED.parent)
    er, eg, eb = loader.load(OUTPUT_ENCRYPTED.name)
    cipher_tensor = np.dstack([er, eg, eb])
    
    # Decrypt using validator (handles reverse pipeline)
    validator = DecryptionValidator(
        key_matrix=K,
        n=N,
        q=Q,
        pre=PRE_MULTIPLY,
        arnold_iters=ARNOLD_ITERS,
    )
    decrypted = validator.decrypt_full_image(cipher_tensor)
    
    # Crop to original dimensions if known
    if original_shape:
        h, w = original_shape
        decrypted = decrypted[:h, :w, :]
        print(f"  Cropped to original size: {(h, w)}")
    
    # Save
    dr, dg, db = decrypted[:, :, 0], decrypted[:, :, 1], decrypted[:, :, 2]
    save_rgb(dr, dg, db, OUTPUT_DECRYPTED)
    
    # Validate if original is available
    orig_path = INPUT_IMAGE if MODE == "decrypt" and INPUT_IMAGE.exists() else None
    if orig_path and original_shape:
        print("\n🔍 Validating...")
        orig_loader = PaddedRGBImageLoader(k=N, images_dir=orig_path.parent)
        or_, og, ob = orig_loader.load(orig_path.name)
        orig_tensor = np.dstack([or_, og, ob])
        orig_cropped = orig_tensor[:original_shape[0], :original_shape[1], :]
        
        mse, psnr = validator.validate_matrices(orig_cropped, decrypted)    
    print()


def debug_encryption_stages():
    """Find where values exceed 251"""
    from read_from_img import PaddedRGBImageLoader
    from hill_cipher_engine import generate_key, encrypt_channel
    
    print("\n=== DEBUGGING ENCRYPTION STAGES ===\n")
    
    # Load a small test image
    loader = PaddedRGBImageLoader(k=N, images_dir=INPUT_IMAGE.parent)
    r, g, b = loader.load(INPUT_IMAGE.name)
    
    # Generate a key for debugging
    rng = np.random.default_rng(RANDOM_SEED) if RANDOM_SEED is not None else None
    K_debug = generate_key(N, Q, rng=rng)
    print(f"Generated debug key")
    
    # Clip input
    r = np.clip(r, 0, Q-1)
    print(f"After clipping - max: {r.max()}")
    
    # Stage 1: Hill encrypt
    h1 = encrypt_channel(r, K_debug, N, Q, PRE_MULTIPLY)
    print(f"After Hill1 - max: {h1.max()}")
    
    # Stage 2: Arnold
    from arnold_transform import ArnoldScrambler
    a1 = ArnoldScrambler.scramble(h1, ARNOLD_ITERS)
    print(f"After Arnold1 - max: {a1.max()}")
    
    # Stage 3: Hill encrypt again
    h2 = encrypt_channel(a1, K_debug, N, Q, PRE_MULTIPLY)
    print(f"After Hill2 - max: {h2.max()}")
    
    # Stage 4: Arnold again
    a2 = ArnoldScrambler.scramble(h2, ARNOLD_ITERS)
    print(f"After Arnold2 - max: {a2.max()}")
    
    return a2

# Call this in run_encrypt before saving

def main():
    print(f"\n{'='*60}")
    print(f"Two-Stage Hill Cipher + Arnold Scramble")
    print(f"Params: n={N}, q={Q}, pre={PRE_MULTIPLY}, arnold_iters={ARNOLD_ITERS}")
    print(f"{'='*60}\n")
    
    if MODE == "encrypt":
        
        K, orig_shape = run_encrypt()
        debug_encryption_stages()
        # Optional: auto-decrypt to test round-trip
        # run_decrypt(K, orig_shape)

        print("\n" + "="*60)
        print("VERIFYING ENCRYPTION OUTPUT")
        print("="*60)
    
        # Check if encryption produced valid files
        
    
        if os.path.exists(OUTPUT_ENCRYPTED):
            enc_img = Image.open(OUTPUT_ENCRYPTED)
            enc_array = np.asarray(enc_img)
            print(f"✓ Encrypted image saved: {enc_array.shape}")
            print(f"  Value range: {enc_array.min()}-{enc_array.max()}")
        
            # Check if values are within modulus
            if enc_array.max() < Q:
                print(f"✓ All encrypted values < {Q} (good)")
            else:
                print(f"⚠️ Warning: Encrypted values exceed {Q}")
        else:
            print(f"❌ ERROR: Encrypted image not saved!")
        
        if os.path.exists(KEY_FILE):
            key = np.load(KEY_FILE)
            print(f"✓ Key saved: shape {key.shape}")
        else:
            print(f"❌ ERROR: Key not saved!")
        
    elif MODE == "decrypt":
        if not KEY_FILE.exists():
            print(f"❌ Key file not found: {KEY_FILE}")
            print("   Run encryption first, or set RANDOM_SEED to regenerate key.")
            return 1
        K = np.load(KEY_FILE)
        print(f"  Loaded key from {KEY_FILE}")
        # Get original shape from encrypted image
        temp_loader = PaddedRGBImageLoader(k=N, images_dir=OUTPUT_ENCRYPTED.parent)
        temp_loader.load(OUTPUT_ENCRYPTED.name)
        
        run_decrypt(K, original_shape=temp_loader.original_shape)
        
    else:
        print(f"❌ Unknown MODE: '{MODE}'. Use 'encrypt' or 'decrypt'.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())