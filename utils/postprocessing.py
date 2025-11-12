" Utility functions for post-processing datasets to prepare for training. "

import numpy as np

def exclude_rest_class(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, rep_ids: np.ndarray):
    " Exclude rest class (0) from dataset."
    rest_class = 0
    select_class_mask = (y != rest_class)
    X = X[select_class_mask]
    y = y[select_class_mask]
    subject_ids = subject_ids[select_class_mask]
    rep_ids = rep_ids[select_class_mask]
    return X, y, subject_ids, rep_ids

def split_data_by_subject(subject_ids: np.ndarray, train_percent: float = 0.7, test_percent: float = 0.15, val_percent: float = 0.15):

    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    n_train = int(n_subjects * train_percent)
    n_test = int(n_subjects * test_percent)

    # shuffle subjects
    shuffled_subjects = unique_subjects.copy()
    np.random.shuffle(shuffled_subjects)

    train_subjects = shuffled_subjects[:n_train]
    test_subjects = shuffled_subjects[n_train:n_train+n_test]
    val_subjects = shuffled_subjects[n_train+n_test:]
    print(f"Train subjects: {train_subjects}")
    print(f"Test subjects: {test_subjects}")
    print(f"Validation subjects: {val_subjects}")

    # Set train, test, val indices
    train_indices = np.isin(subject_ids, train_subjects)
    test_indices = np.isin(subject_ids, test_subjects)
    val_indices = np.isin(subject_ids, val_subjects)
    return train_indices, test_indices, val_indices

# Make label maps for gestures
# CrossEntropyLoss in PyTorch expects class labels to be in the range [0, C-1]

def make_label_maps(y: np.ndarray):
    """Create mappings from original labels to indices and back.
    Args:
        y (np.ndarray): Array of original labels.
    Returns:
        label_to_index (dict): Mapping from original label to index.
        index_to_label (dict): Mapping from index to original label."""
    unique_labels = np.unique(y)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    return label_to_index, index_to_label

def encode_labels_zero_based(y: np.ndarray) -> np.ndarray:
    """Encode labels to be zero-based indices.
    Args:
        y (np.ndarray): Array of original labels.
    Returns:
        np.ndarray: Array of encoded labels as zero-based indices."""
    label_to_index, index_to_label = make_label_maps(y)
    return np.array([label_to_index[label] for label in y])
