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