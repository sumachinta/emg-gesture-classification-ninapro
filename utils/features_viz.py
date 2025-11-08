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



def plot_feature_by_gesture(
    df: pd.DataFrame,
    feature: str = "RMS",
    channel: int = 0,
    showfliers: bool = False,
    figsize: Tuple[float, float] = (9, 4),
):
    """Boxplot of one feature across gestures for a single channel."""
    dsub = df[(df["feature"] == feature) & (df["channel"] == channel)]
    gestures = np.sort(dsub["gesture"].unique())
    data = [dsub[dsub["gesture"] == g]["value"].values for g in gestures]

    plt.figure(figsize=figsize)
    plt.boxplot(data, labels=gestures, showfliers=showfliers)
    plt.xlabel("Gesture")
    plt.ylabel(feature)
    plt.title(f"{feature} by gesture (channel {channel})")
    plt.tight_layout()
    plt.show()


def plot_feature_mean_sem(
    df: pd.DataFrame,
    feature: str = "MAV",
    channel: int = 0,
    figsize: Tuple[float, float] = (9, 4),
):
    """Bar plot of mean±SEM of one feature per gesture for a single channel."""
    dsub = df[(df["feature"] == feature) & (df["channel"] == channel)]
    groups = dsub.groupby("gesture")["value"]
    gestures = np.array(sorted(groups.groups.keys()))
    means = groups.mean().reindex(gestures).values
    sems = groups.sem().reindex(gestures).values

    x = np.arange(len(gestures))
    plt.figure(figsize=figsize)
    plt.bar(x, means, yerr=sems, capsize=3)
    plt.xticks(x, gestures)
    plt.xlabel("Gesture")
    plt.ylabel(f"{feature} (mean ± SEM)")
    plt.title(f"{feature} by gesture (channel {channel})")
    plt.tight_layout()
    plt.show()


def plot_feature_heatmap(
    df: pd.DataFrame,
    feature: str = "WL",
    agg: str = "mean",
    figsize: Tuple[float, float] = (10, 6),
):
    """Heatmap of gestures × channels for a chosen feature (agg = mean/median)."""
    pivot = df[df["feature"] == feature].pivot_table(
        index="gesture", columns="channel", values="value",
        aggfunc="median" if agg == "median" else "mean"
    ).sort_index()

    plt.figure(figsize=figsize)
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label=f"{feature} ({agg})")
    plt.yticks(np.arange(pivot.shape[0]), pivot.index)
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns)
    plt.xlabel("Channel")
    plt.ylabel("Gesture")
    plt.title(f"{feature} {agg} by gesture × channel")
    plt.tight_layout()
    plt.show()


def radar_gesture_signature(
    df: pd.DataFrame,
    channel: int = 0,
    gestures: Iterable[int] = (),
    normalize: bool = True,
    figsize: Tuple[float, float] = (6, 6),
):
    """
    Radar plot of multi-feature signature per gesture for one channel.
    If `gestures` is empty, uses all gestures present.
    """
    feats = list(sorted(df["feature"].unique()))
    theta = np.linspace(0, 2 * np.pi, len(feats), endpoint=False)
    theta = np.concatenate([theta, theta[:1]])

    if not gestures:
        gestures = np.sort(df["gesture"].unique())

    plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)

    for g in gestures:
        dsub = df[(df["channel"] == channel) & (df["gesture"] == g)]
        m = dsub.groupby("feature")["value"].mean().reindex(feats).values
        if normalize:
            m = (m - m.mean()) / (m.std(ddof=1) + 1e-9)
        m = np.concatenate([m, m[:1]])
        ax.plot(theta, m, label=f"g{g}")
        ax.fill(theta, m, alpha=0.15)

    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(feats)
    ax.set_title(f"Gesture signatures (channel {channel})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()


def pca_scatter(
    F: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
    standardize: bool = True,
    figsize: Tuple[float, float] = (6, 5),
):
    """
    PCA scatter of concatenated channel features.
    F: (N, C, K) or (N, C*K)
    """
    if F.ndim == 3:
        N, C, K = F.shape
        Xf = F.reshape(N, C * K)
    else:
        Xf = F
    if standardize:
        Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)

    pcs = PCA(n_components=n_components).fit_transform(Xf)

    plt.figure(figsize=figsize)
    sc = plt.scatter(pcs[:, 0], pcs[:, 1], c=y, s=6, alpha=0.4, cmap="tab20")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA of EMG TD features")
    cb = plt.colorbar(sc); cb.set_label("Gesture")
    plt.tight_layout()
    plt.show()
