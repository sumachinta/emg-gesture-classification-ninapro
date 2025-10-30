# NinaPro DB2 EMG Gesture Classification

This repository implements an end-to-end machine learning and deep learning pipeline for decoding **hand and finger gestures from surface EMG (sEMG)** signals using the **NinaPro DB2 dataset**.

The goal of this project is to reproduce and extend classical EMG decoding workflows — from preprocessing and feature extraction to deep neural network classifiers — and benchmark them on gesture recognition accuracy.

---

## 🧠 Project Overview

Electromyography (EMG) captures the electrical activity produced by skeletal muscles.  
In this project, we use multi-channel EMG signals to **classify hand gestures** in the NinaPro DB2 dataset.

We develop two complementary decoding pipelines:

| Pipeline | Approach | Description |
|-----------|-----------|--------------|
| **Classical ML** | Feature-based | RMS, MAV, WL, ZC, SSC + LDA/SVM |
| **Deep Learning** | End-to-end | 1D-CNN / ShallowFBCSPNet / EEGNet-style models |

---

## 📂 Dataset — NinaPro DB2

**Source:** [NinaPro (Non-Invasive Adaptive Prosthetics)](https://ninapro.hevs.ch/)  
**Dataset:** *DB2 – 40 subjects performing 49 hand movements*  
**Sensors:** 12-channel Delsys Trigno sEMG  
**Sampling rate:** 2 kHz  
**Duration:** ~6 seconds per repetition, 6 repetitions per gesture  

**Gesture classes include:**
- Individual finger flexions
- Combined hand grasps (spherical, lateral, cylindrical)
- Wrist and finger movements
- Rest state (T0)

Data files are distributed as `.mat` (MATLAB) files per subject.

---

## ⚙️ Methods

### 1️⃣ Preprocessing
- Band-pass filter (20–450 Hz)
- Notch filter at 50/60 Hz
- Rectification and envelope extraction (low-pass 5–10 Hz)
- Normalization (z-score per channel)
- Epoching into 200–300 ms windows

### 2️⃣ Feature Extraction (Classical)
- **Time-domain:** RMS, MAV, WL, ZC, SSC  
- **Frequency-domain:** Mean and Median Frequency  
- **Time-frequency:** STFT / wavelet features  

### 3️⃣ Machine Learning Models
- Linear Discriminant Analysis (LDA)
- Support Vector Machine (SVM)
- Random Forest

### 4️⃣ Deep Learning Models
- 1D-CNN
- ShallowFBCSPNet (adapted for EMG)
- EEGNet-style compact CNNs

### 5️⃣ Evaluation
- Stratified K-Fold and subject-wise cross-validation  
- Metrics: Accuracy, F1-score, Cohen’s κ  
- Visualization: Confusion matrices and t-SNE embeddings of learned features  

