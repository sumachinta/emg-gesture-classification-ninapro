# Functions to load data and prepreocess EMG signals from the Ninapro database.

from typing import Tuple, Optional
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

def load_ninapro_data(subject_number: int, exercise_number: int, dataPath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load EMG data, stimulus labels, and repetition info from the Ninapro database for a given subject and exercise.
    Parameters:
    subject_number (int): The subject number (e.g., 1, 2, ..., 40).
    exercise_number (int): The exercise number (e.g., 1, 2, 3).
    dataPath (Path): The base path to the Ninapro database.
    Returns:
    Tuple containing:
        - emg (np.ndarray): The EMG signal data.
        - stimulus (np.ndarray): The stimulus labels.
        - repetition (np.ndarray): The repetition information.
        - time (np.ndarray): The time vector corresponding to the EMG data.
        - Fs (int): The sampling frequency. # 2000 Hz for Ninapro DB2.
    """
    subject_path = dataPath / f"DB2_s{subject_number}" / f"S{subject_number}_E{exercise_number}_A1.mat"
    mat_data = loadmat(subject_path)
    emg = mat_data['emg']
    stimulus = mat_data['restimulus']
    repetition = mat_data['rerepetition']
    Fs = 2000  # Sampling frequency
    time = np.arange(emg.shape[0]) / Fs  
    return emg, stimulus, repetition, time, Fs

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth bandpass filter to the data"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_emg = filtfilt(b, a, data)
    return filtered_emg

def rectify_signal(data: np.ndarray) -> np.ndarray:
    return np.abs(data)

def smooth_signal(data: np.ndarray, window_size: int) -> np.ndarray:
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def preprocess_emg(emg: np.ndarray, 
                   fs: float, 
                   smoothen: Optional[bool] = True, 
                   lowcut: Optional[float] = 20.0, 
                   highcut: Optional[float] = 450.0, 
                   smooth_window: Optional[float] = 0.05,
                   normalize: Optional[bool] = True) -> np.ndarray:
    """Preprocess EMG signal by applying bandpass filter, rectification, and smoothing.
    Args:
        emg (np.ndarray): Raw EMG signal.
        fs (float): Sampling frequency.
        smoothen (bool, optional): Whether to apply smoothing. Defaults to True.
        lowcut (float, optional): Low cutoff frequency for bandpass filter. Defaults to 20.0 Hz.
        highcut (float, optional): High cutoff frequency for bandpass filter. Defaults to 450.0 Hz.
        smooth_window (float, optional): Smoothing window size in seconds. Defaults to 0.05 s.

    Returns:
        np.ndarray: Preprocessed EMG signal.
    """
    # Apply bandpass filter
    filtered_emg = bandpass_filter(emg, lowcut=lowcut, highcut=highcut, fs=fs, order=4)
    # Full-wave rectification
    rectified_emg = rectify_signal(filtered_emg)
    # Smoothing to obtain the envelope for muscle activation
    if smoothen:
        window_size = int(0.05 * fs)  # 50 ms window
        processed_emg = smooth_signal(rectified_emg, window_size)
    else:
        processed_emg = rectified_emg
    # z-score normalization
    if normalize:
        processed_emg = (processed_emg - np.mean(processed_emg)) / np.std(processed_emg)
    return processed_emg

def preprocess_all_channels(emg, Fs, lowcut=20.0, highcut=450.0, smoothen=True, smooth_window=0.05, normalize=True):
    """Preprocess each channel and return array of same shape."""
    n_samples, n_channels = emg.shape
    filtered = np.zeros_like(emg, dtype=float)
    for ch in range(n_channels):
        filtered[:, ch] = preprocess_emg(emg[:, ch], Fs,
            smoothen=smoothen, lowcut=lowcut, highcut=highcut, smooth_window=smooth_window, normalize=normalize)
    return filtered


def majority_label_in_window(stim_win, majority_thr=0.90):
    """
    Returns (label, frac) where label is the majority class in the window
    if its fraction >= majority_thr; otherwise returns (None, max_frac).
    """
    vals, counts = np.unique(stim_win, return_counts=True)
    idx = np.argmax(counts)
    label = int(vals[idx])
    frac = counts[idx] / stim_win.size
    if frac >= majority_thr:
        return label, frac
    return None, frac


def epoch_data_sliding_window(dataPath: str, Fs: float, exercise_number: int, subjects: list, window_s: float, step_s: float, majority_threshold: float, include_rest: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Extract epochs from EMG data using sliding window approach with majority voting for labels.
    
    Parameters:
    dataPath (str): Path to the Ninapro database.
    Fs (float): Sampling frequency of the EMG data.
    exercise_number (int): Exercise number to load data from.
    subjects (list): List of subject numbers to process.
    window_s (float): Window length in seconds.
    step_s (float): Step size in seconds.
    majority_threshold (float): Majority threshold for labeling.
    include_rest (bool): Whether to include rest (label 0) windows.

    Returns:
    - X: np.ndarray, shape (num_epochs, C, L), extracted epochs
    - y: np.ndarray, shape (num_epochs,), labels for each epoch
    - subject_ids: np.ndarray, shape (num_epochs,), subject IDs for each epoch
    - rep_ids: np.ndarray, shape (num_epochs,), repetition IDs for each epoch
    - t0: np.ndarray, shape (num_epochs,), start time of each epoch
    - coverage: np.ndarray, shape (num_epochs,), majority fraction used for labeling
    - gesture_ids_full: np.ndarray, all gesture IDs present in the dataset
    - n_channels: int, number of EMG channels
    """

    # Probe dataset to get Fs, C, gesture IDs
    emg0, stim0, rep0, time0, Fs0 = load_ninapro_data(subject_number=subjects[0], exercise_number=exercise_number, dataPath=dataPath)
    Fs = Fs0
    n_channels = emg0.shape[1]
    gesture_ids_full = np.sort(np.unique(stim0)).astype(int)   # includes 0=rest
    if not include_rest:
        gesture_ids_full = gesture_ids_full[gesture_ids_full > 0]

    print(f"Fs={Fs} Hz, channels={n_channels}, gestures(all)={gesture_ids_full.tolist()}")

    # Main: sliding-window extraction
    win = int(round(window_s * Fs))          # L = T * F  (in samples; T in seconds here)
    step = int(round(step_s * Fs))           # L_d = S * F 
    assert win > 0 and step > 0, "Window and step must be >=1 sample"

    X_list = []             # (N,C,L) (Number of epochs, Channels, Length of samples)
    y_list = []             # gesture label per window (int)
    subj_list = []          # subject id (1-based)
    rep_list = []           # repetition id (if determinable by majority)
    t0_list = []            # window start time (seconds, for traceability)
    coverage_list = []      # majority fraction used for label

    for subject_number in subjects:
        print(f"[T={int(window_s*1000)} ms] Processing subject {subject_number}...")
        emg, stimulus, repetition, time, Fs_check = load_ninapro_data(
            subject_number=subject_number, exercise_number=exercise_number, dataPath=dataPath
        )
        if Fs_check != Fs:
            raise ValueError(f"Inconsistent Fs: subject {subject_number} -> {Fs_check} vs {Fs}")

        # preprocess once for all channels
        emg_clean = preprocess_all_channels(emg, Fs)   # (N, C)

        N = emg_clean.shape[0]
        # slide [start, start+win)
        for start in range(0, N - win + 1, step):
            stop = start + win

            stim_win = stimulus[start:stop]
            # label by majority vote with threshold
            lbl, frac = majority_label_in_window(stim_win, majority_thr=majority_threshold)
            if lbl is None:
                continue  # ambiguous window; skip

            if (not include_rest) and (lbl == 0):
                continue

            # pick repetition by majority as well (optional, didn't have to)
            rep_win = repetition[start:stop]
            rep_vals, rep_counts = np.unique(rep_win, return_counts=True)
            rep_id = int(rep_vals[np.argmax(rep_counts)])

            # extract EMG (C, L)
            # emg_clean is (N, C) -> slice -> (L, C) -> transpose
            epoch = emg_clean[start:stop, :].T

            X_list.append(epoch)
            y_list.append(int(lbl))
            subj_list.append(int(subject_number))
            rep_list.append(rep_id)
            t0_list.append(float(time[start]))
            coverage_list.append(float(frac))

    # Stack and save for this window length
    if len(X_list) == 0:
        print(f"No windows kept for T={int(window_s*1000)} ms; skipping save.")
        return None, None, None, None, None, None, None, None

    X = np.stack(X_list, axis=0)                    # (N, C, L)
    y = np.array(y_list, dtype=int)                 # (N,)
    subject_ids = np.array(subj_list, dtype=int)    # (N,)
    rep_ids = np.array(rep_list, dtype=int)         # (N,)
    t0 = np.array(t0_list, dtype=float)             # (N,)
    coverage = np.array(coverage_list, dtype=float) # (N,)

    # Cast to float32 first (halves size, speeds I/O)
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.int16, copy=False)
    subject_ids = subject_ids.astype(np.int16, copy=False)
    rep_ids = rep_ids.astype(np.int8, copy=False)
    t0 = t0.astype(np.float32, copy=False)
    coverage = coverage.astype(np.float32, copy=False)
    return X, y, subject_ids, rep_ids, t0, coverage, gesture_ids_full, n_channels

