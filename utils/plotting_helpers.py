import numpy as np
import matplotlib.pyplot as plt

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