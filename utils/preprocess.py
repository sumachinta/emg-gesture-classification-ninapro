# Functions to load data and prepreocess EMG signals from the Ninapro database.

from typing import Tuple
import numpy as np
from pathlib import Path
from scipy.io import loadmat

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