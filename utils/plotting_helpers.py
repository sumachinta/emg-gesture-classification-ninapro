import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Iterable
from sklearn.decomposition import PCA


def _time_axis(n_samples, Fs):
    return np.arange(n_samples) / float(Fs)

def _mean_sem(arr, axis=0):
    """Return mean and standard error along `axis`."""
    arr = np.asarray(arr)
    mean = arr.mean(axis=axis)
    sem  = arr.std(axis=axis, ddof=1) / np.sqrt(arr.shape[axis])
    return mean, sem

def _subject_balanced_mean_sem(X_sel, subj_sel):
    """
    Return mean±SEM across subjects, each subject contributing its mean trace once.
    Inputs:
    X_sel: (N, L) windows for one channel
    subj_sel: (N,) subject ids aligned with X_sel
    Returns mean±SEM across subjects (each subject contributes its mean trace once).
    """
    means_by_subj = []
    for s in np.unique(subj_sel):
        m = X_sel[subj_sel == s].mean(axis=0)  # mean over that subject's windows
        means_by_subj.append(m)
    means_by_subj = np.stack(means_by_subj, axis=0)  # (S, L)
    return _mean_sem(means_by_subj, axis=0)          # across subjects

def plot_gesture_trace(
    npz_path,
    gesture_label,
    subject=None,              # int (e.g., 7) or None for all subjects
    channel=0,                 # channel index to plot
    subject_balanced=True,     # for all-subjects: average per subject first
    color=None,                # optional matplotlib color
    title_prefix="EMG"
):
    """
    Load NPZ (with X[N,C,L], y[N], subject_ids[N], Fs) and plot mean±SEM
    for a given gesture label. If `subject` is None, aggregates across
    all subjects (subject-balanced by default).
    """
    d = np.load(npz_path, allow_pickle=False)
    X = d["X"]                     # (N, C, L)
    y = d["y"]                     # (N,)
    subj = d["subject_ids"]        # (N,)
    Fs = float(d["Fs"])
    L  = int(d["L"])

    # Filter by gesture (and subject if given)
    mask = (y == int(gesture_label))
    if subject is not None:
        mask &= (subj == int(subject))

    if not np.any(mask):
        label = f"gesture={gesture_label}, subject={subject}" if subject else f"gesture={gesture_label}"
        raise ValueError(f"No windows found for {label}")

    # Select channel and reshape to (N, L)
    X_ch = X[mask, channel, :]  # (N, L)
    subj_sel = subj[mask]

    # Compute mean and SEM
    if subject is None and subject_balanced:
        mean, sem = _subject_balanced_mean_sem(X_ch, subj_sel)
        subtitle = f"All subjects (subject-balanced), gesture={gesture_label}, ch={channel}"
    else:
        mean, sem = _mean_sem(X_ch, axis=0)  # window-level aggregation
        who = f"subject={int(subject)}" if subject is not None else "all subjects (pooled windows)"
        subtitle = f"{who}, gesture={gesture_label}, ch={channel}"

    # Plot
    t = _time_axis(L, Fs)
    plt.figure(figsize=(4, 3))
    line = plt.plot(t, mean, lw=2, color=color)[0]
    c = line.get_color() if color is None else color
    plt.fill_between(t, mean - sem, mean + sem, alpha=0.25, linewidth=0, color=c)
    plt.xlabel("Time (s)")
    plt.ylabel("EMG (normalized)")
    plt.title(f"{title_prefix}: mean ± SEM\n{subtitle}")
    plt.tight_layout()
    plt.show()


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
