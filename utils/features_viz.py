# features_viz.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Sequence, Tuple, Optional
from sklearn.decomposition import PCA


def emg_td_features(
    X: np.ndarray,
    zc_thresh: float = 0.0,
    ssc_thresh: float = 0.0,
    batch_size: int = 20000,
    flatten: bool = False,
    dtype_out=np.float32,
) -> np.ndarray:
    """
    Compute TD features per channel over windows X of shape (N, C, L).

    Features per channel (K=5): MAV, RMS, WL, ZC, SSC

    Returns
    -------
    F : np.ndarray
        (N, C, 5) if flatten=False, else (N, C*5). dtype=dtype_out.
    """
    assert X.ndim == 3, "X must be (N, C, L)"
    N, C, L = X.shape
    F = np.empty((N, C, 5), dtype=dtype_out)

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        xb = np.asarray(X[start:end], dtype=np.float32, order="C")  # (B,C,L)

        # MAV, RMS
        mav = np.mean(np.abs(xb), axis=-1)
        rms = np.sqrt(np.mean(xb * xb, axis=-1))

        # First difference and WL
        d1 = np.diff(xb, axis=-1)
        wl = np.sum(np.abs(d1), axis=-1)

        # Zero crossings
        x0, x1 = xb[..., :-1], xb[..., 1:]
        zc_cond = (x0 * x1) < 0.0
        if zc_thresh > 0.0:
            zc_cond &= (np.abs(x0 - x1) >= zc_thresh)
        zc = np.sum(zc_cond, axis=-1)

        # Slope sign changes
        d10, d11 = d1[..., :-1], d1[..., 1:]
        ssc_cond = (d10 * d11) < 0.0
        if ssc_thresh > 0.0:
            ssc_cond &= (np.abs(d10) >= ssc_thresh) & (np.abs(d11) >= ssc_thresh)
        ssc = np.sum(ssc_cond, axis=-1)

        # Pack
        F[start:end, :, 0] = mav
        F[start:end, :, 1] = rms
        F[start:end, :, 2] = wl
        F[start:end, :, 3] = zc
        F[start:end, :, 4] = ssc

    return F.reshape(N, C * 5) if flatten else F


def features_to_df(
    F: np.ndarray,
    y: np.ndarray,
    subject_ids: Optional[np.ndarray] = None,
    rep_ids: Optional[np.ndarray] = None,
    feat_names: Sequence[str] = ("MAV", "RMS", "WL", "ZC", "SSC"),
) -> pd.DataFrame:
    """
    Convert features to a long/tidy DataFrame.

    Parameters
    ----------
    F : (N, C, K) features
    y : (N,) gesture labels
    subject_ids : (N,), optional
    rep_ids : (N,), optional

    Returns
    -------
    df : DataFrame with columns:
        ['gesture','subject','rep','channel','feature','value']
    """
    assert F.ndim == 3, "F must be (N, C, K)"
    N, C, K = F.shape
    assert K == len(feat_names), "feat_names length must match F.shape[2]"

    subj = subject_ids if subject_ids is not None else np.full(N, -1, dtype=int)
    rep = rep_ids if rep_ids is not None else np.full(N, -1, dtype=int)

    parts = []
    for k, fname in enumerate(feat_names):
        # values_k: (N, C) -> melt by channel
        values_k = F[:, :, k]
        # build one DataFrame per channel to avoid giant copies
        for ch in range(C):
            parts.append(pd.DataFrame(
                {
                    "gesture": y,
                    "subject": subj,
                    "rep": rep,
                    "channel": ch,
                    "feature": fname,
                    "value": values_k[:, ch],
                }
            ))
    return pd.concat(parts, ignore_index=True)
