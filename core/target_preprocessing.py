from __future__ import annotations

import cv2
import numpy as np


def preprocess_target_roi(gray: np.ndarray) -> np.ndarray:
    """Enhance target-object contrast for non-distance measurement modes."""
    if gray.size == 0:
        return gray.astype(np.uint8, copy=False)
    source = gray.astype(np.uint8, copy=False)
    lo, hi = np.percentile(source, (1.0, 99.0))
    if hi <= lo + 1e-6:
        normalized = source.copy()
    else:
        normalized = np.clip((source.astype(np.float32) - float(lo)) * 255.0 / float(hi - lo), 0, 255).astype(np.uint8)
    denoised = cv2.medianBlur(normalized, 3)
    denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=18, sigmaSpace=5)
    sigma = max(8.0, min(float(min(source.shape[:2])) / 8.0, 42.0))
    background = cv2.GaussianBlur(denoised, (0, 0), sigmaX=sigma, sigmaY=sigma)
    corrected = denoised.astype(np.float32) - background.astype(np.float32) + 128.0
    corrected = denoised.astype(np.float32) * 0.65 + corrected * 0.35
    return np.clip(corrected, 0, 255).astype(np.uint8)
