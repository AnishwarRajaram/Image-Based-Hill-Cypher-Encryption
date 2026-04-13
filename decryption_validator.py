"""
Decryption and Validation Engine

Primary Responsibilities:
1. Matrix Simplification: Calculate the modular inverse of the Hill Cipher key via Gauss-Jordan elimination.
2. Decryption Pipeline: Orchestrate the LIFO reversal of the Two-Stage Hill Cipher (TSHC).
3. Statistical Validation: Prove lossless data recovery using MSE and PSNR.

Dependencies: numpy, hill_cipher_engine, arnold_transform
"""

from __future__ import annotations
import numpy as np

# Imports from the Cryptography Engine and Image Processor modules
from hill_cipher_engine import decrypt_channel, _mod_inv_matrix
from arnold_transform import ArnoldScrambler

class DecryptionValidator:
    def __init__(self, key_matrix: np.ndarray, n: int, q: int, pre: bool = True, arnold_iters: int = 1):
        """
        Initializes the decryption engine and calculates the modular inverse.
        Parameters match the 'n' (block size) and 'q' (modulo) from the encryption engine.
        """
        self.K = key_matrix
        self.n = n
        self.q = q
        self.pre = pre
        self.arnold_iters = arnold_iters
        
        # Core Task: Find the modular inverse matrix over F_q
        self.K_inv = _mod_inv_matrix(self.K, self.q)

    def decrypt_full_image(self, cipher_tensor: np.ndarray) -> np.ndarray:
        """
        Reverses the Two-Stage Hill Cipher (TSHC) pipeline.
        Expects a padded (N, N, 3) tensor.
        Order: Inverse Arnold 2 -> Inverse Hill 2 -> Inverse Arnold 1 -> Inverse Hill 1
        """
        print("--- Initiating Decryption Pipeline ---")
        print(f"Calculated Modular Inverse of Key (mod {self.q}):\n{self.K_inv}")
        
        # Step 1: Reverse Stage 2 Arnold Transform
        print("Step 1: Reversing Stage 2 Arnold Transform...")
        step1_tensor = ArnoldScrambler.inverse(cipher_tensor, iterations=self.arnold_iters)
        
        # Step 2: Reverse Stage 2 Hill Cipher
        print("Step 2: Reversing Stage 2 Hill Cipher...")
        step2_tensor = np.empty_like(step1_tensor)
        for c in range(3): # Iterate over R, G, B channels independently
            step2_tensor[:, :, c] = decrypt_channel(
                step1_tensor[:, :, c], self.K, self.n, self.q, self.pre
            )
            
        # Step 3: Reverse Stage 1 Arnold Transform
        print("Step 3: Reversing Stage 1 Arnold Transform...")
        step3_tensor = ArnoldScrambler.inverse(step2_tensor, iterations=self.arnold_iters)
        
        # Step 4: Reverse Stage 1 Hill Cipher
        print("Step 4: Reversing Stage 1 Hill Cipher...")
        decrypted_tensor = np.empty_like(step3_tensor)
        for c in range(3):
            decrypted_tensor[:, :, c] = decrypt_channel(
                step3_tensor[:, :, c], self.K, self.n, self.q, self.pre
            )
            
        print("Decryption Complete.\n")
        return decrypted_tensor

    def calculate_mse(self, original_tensor: np.ndarray, decrypted_tensor: np.ndarray) -> float:
        """Calculates the Mean Square Error between the original and decrypted tensors."""
        orig = original_tensor.astype(np.float64)
        dec = decrypted_tensor.astype(np.float64)
        return float(np.mean((orig - dec) ** 2))

    def calculate_psnr(self, mse: float, max_pixel_value: float = 255.0) -> float:
        """Calculates the Peak Signal-to-Noise Ratio."""
        if mse == 0.0:
            return float('inf') # Perfect reconstruction
        return float(10 * np.log10((max_pixel_value ** 2) / mse))

    def validate_matrices(self, original_tensor: np.ndarray, decrypted_tensor: np.ndarray) -> tuple[float, float]:
        """
        Runs statistical validation to prove that the matrix transformations were completely lossless.
        """
        print("--- Validation & Statistical Analysis ---")
        mse = self.calculate_mse(original_tensor, decrypted_tensor)
        psnr = self.calculate_psnr(mse)
        
        print(f"Mean Square Error (MSE): {mse}")
        print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")
        
        if mse == 0.0:
            print("SUCCESS: The decrypted matrix perfectly matches the original matrix. Zero data loss.")
        else:
            print("WARNING: Data loss or corruption detected. Matrices do not match.")
            
        return mse, psnr
