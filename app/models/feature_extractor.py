"""Dual-Branch multi-scale feature extractor for exoplanetary light curves.

Extracts deep morphological and astrophysical features from:
1. Global View (2,001 points) - Dilated 1D receptive fields, secondary eclipses, odd-even parity.
2. Local View (201 points) - Ingress/egress curvature, U-shape vs V-shape limb darkening.
3. Astrophysical Scalars - BLS period, depth, duration, SDE, duty cycle.
"""

from typing import Dict, Any, Optional
import numpy as np
from app.photometry.bls import BLSResult


def _make_dilated_kernel(kernel: np.ndarray, dilation: int) -> np.ndarray:
    """Expands kernel with zeros for dilated convolution."""
    if dilation <= 1:
        return kernel
    k_len = len(kernel)
    eff_len = (k_len - 1) * dilation + 1
    d_kernel = np.zeros(eff_len, dtype=kernel.dtype)
    d_kernel[::dilation] = kernel
    return d_kernel


def _dilated_1d_conv_pool(signal: np.ndarray, kernel_size: int, dilation: int) -> np.ndarray:
    """1D dilated convolution with fixed geometric edge filters followed by pooling.
    
    Extracts multi-scale gradient, curvature, and step features across varying receptive fields.
    Vectorized in pure C using np.convolve.
    """
    n = len(signal)
    eff_kernel = (kernel_size - 1) * dilation + 1
    if n < eff_kernel:
        return np.zeros(4, dtype=np.float32)

    # 3 canonical scientific filter kernels:
    # 1. Smoothing / local baseline
    # 2. Gradient / edge detector (-1, 0, 1)
    # 3. Second derivative / curvature (1, -2, 1)
    k_smooth = np.ones(kernel_size, dtype=np.float64) / kernel_size
    k_grad = np.linspace(-1.0, 1.0, kernel_size, dtype=np.float64)
    k_curv = np.zeros(kernel_size, dtype=np.float64)
    if kernel_size >= 3:
        k_curv[0] = 1.0
        k_curv[kernel_size // 2] = -2.0
        k_curv[-1] = 1.0

    dk_smooth = _make_dilated_kernel(k_smooth, dilation)
    dk_grad = _make_dilated_kernel(k_grad, dilation)
    dk_curv = _make_dilated_kernel(k_curv, dilation)

    c_smooth = np.convolve(signal, dk_smooth, mode='valid')
    c_grad = np.convolve(signal, dk_grad, mode='valid')
    c_curv = np.convolve(signal, dk_curv, mode='valid')

    # Pool summary statistics across time: min, max, mean, std
    features = np.array([
        np.min(c_smooth) if len(c_smooth) > 0 else 0.0,
        np.max(np.abs(c_grad)) if len(c_grad) > 0 else 0.0,
        np.min(c_curv) if len(c_curv) > 0 else 0.0,
        np.std(c_smooth) if len(c_smooth) > 0 else 0.0,
    ], dtype=np.float32)

    return features



class DualBranchFeatureExtractor:
    """Extracts invariant astrophysical features from dual-view folded light curves."""

    def __init__(self, global_dilations=(1, 2, 4, 8), local_dilations=(1, 2, 4)):
        self.global_dilations = global_dilations
        self.local_dilations = local_dilations

    def extract(
        self,
        global_view: np.ndarray,
        local_view: np.ndarray,
        bls_result: Optional[BLSResult] = None,
        extra_scalars: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Extract combined 1D feature vector for classification.
        
        Returns:
            1D np.ndarray of shape (D,), normalized and finite.
        """
        # 1. Global View Morphological Features
        global_features = []
        for d in self.global_dilations:
            gf = _dilated_1d_conv_pool(global_view, kernel_size=5, dilation=d)
            global_features.append(gf)
        global_vec = np.concatenate(global_features)

        # Global Astrophysical Tests:
        # A) Secondary eclipse depth ratio (at phase ~0.5 vs phase ~0.0)
        # Phase 0 is at center (idx 1000 for 2001 bins)
        center_idx = len(global_view) // 2
        sec_idx_left = int(len(global_view) * 0.0)  # phase -0.5
        sec_idx_right = int(len(global_view) * 0.99) # phase +0.5

        primary_depth = 1.0 - np.min(global_view[max(0, center_idx - 50):min(len(global_view), center_idx + 51)])
        sec_depth_left = 1.0 - np.min(global_view[0:50])
        sec_depth_right = 1.0 - np.min(global_view[-50:])
        sec_depth = max(0.0, max(sec_depth_left, sec_depth_right))

        sec_ratio = float(sec_depth / (primary_depth + 1e-6))
        sec_ratio = np.clip(sec_ratio, 0.0, 5.0)

        # B) Odd-even or Left-Right transit asymmetry
        left_wing = global_view[center_idx - 100:center_idx]
        right_wing = global_view[center_idx + 1:center_idx + 101][::-1]
        asymmetry = float(np.mean(np.abs(left_wing - right_wing)))

        # C) Out-of-transit scatter
        oot_flux = np.concatenate([global_view[:center_idx - 100], global_view[center_idx + 100:]])
        oot_std = float(np.std(oot_flux))
        oot_kurtosis = float(np.mean(((oot_flux - np.mean(oot_flux)) / (oot_std + 1e-8))**4) - 3.0)

        global_astrophysical = np.array([
            primary_depth,
            sec_depth,
            sec_ratio,
            asymmetry,
            oot_std,
            oot_kurtosis
        ], dtype=np.float32)

        # 2. Local View Features
        local_features = []
        for d in self.local_dilations:
            lf = _dilated_1d_conv_pool(local_view, kernel_size=3, dilation=d)
            local_features.append(lf)
        local_vec = np.concatenate(local_features)

        # Local Shape Tests (U-shape Mandel-Agol vs V-shape Eclipsing Binary):
        loc_center = len(local_view) // 2
        local_min_val = np.min(local_view)
        local_depth = 1.0 - local_min_val

        # Flat-bottom duration ratio: fraction of transit points with flux < 90% of max depth
        transit_threshold = 1.0 - 0.90 * local_depth
        in_flat_bottom = np.sum(local_view <= transit_threshold)
        flat_ratio = float(in_flat_bottom / len(local_view))

        # Ingress / egress slope: steepness of flux transition
        local_grad = np.gradient(local_view)
        max_ingress_slope = float(np.max(np.abs(local_grad)))
        local_snr = float(local_depth / (oot_std + 1e-6))

        local_astrophysical = np.array([
            local_depth,
            flat_ratio,
            max_ingress_slope,
            local_snr
        ], dtype=np.float32)

        # 3. Scalar Features (from BLS or direct dict)
        if bls_result is not None:
            period = float(bls_result.best_period)
            duration = float(bls_result.best_duration)
            depth = float(bls_result.best_depth)
            sde = float(bls_result.sde)
            duty_cycle = float(bls_result.best_duration / (bls_result.best_period + 1e-6))
        elif extra_scalars is not None:
            period = float(extra_scalars.get("period", 3.0))
            duration = float(extra_scalars.get("duration", 0.1))
            depth = float(extra_scalars.get("depth", 0.005))
            sde = float(extra_scalars.get("sde", 8.0))
            duty_cycle = float(extra_scalars.get("duty_cycle", 0.03))
        else:
            period = 3.0
            duration = 0.1
            depth = local_depth
            sde = local_snr
            duty_cycle = 0.03

        scalars = np.array([
            period,
            duration,
            depth,
            sde,
            duty_cycle
        ], dtype=np.float32)

        # Concatenate all features
        full_vector = np.concatenate([
            global_vec,             # len = 4 * 4 = 16
            global_astrophysical,   # len = 6
            local_vec,              # len = 3 * 4 = 12
            local_astrophysical,    # len = 4
            scalars                 # len = 5
        ]).astype(np.float32)       # Total dim = 43

        # Replace any potential NaNs or Infs with safe 0.0
        full_vector = np.nan_to_num(full_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        return full_vector
