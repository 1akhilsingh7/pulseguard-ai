"""
=============================================================================
WEARABLE-SIGNAL STRESS-RISK CLASSIFICATION (WESAD REAL DATASET)
Module: preprocessing.py
Description: Data loading, cleaning, feature selection, and subject-independent
             train/test splitting based on genuine WESAD physiological recordings.
Dataset Source: University of Siegen & UCI Machine Learning Repository
Target Audience: 2nd-year B.Tech CSE presentation & demonstration.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Resolve directory paths reliably
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILENAME = os.path.join(BASE_DIR, "wesad_processed_dataset.csv")
RAW_SAMPLES_FILENAME = os.path.join(BASE_DIR, "wesad_raw_samples.pkl")
SCALER_FILENAME = os.path.join(BASE_DIR, "wesad_scaler.pkl")

# List of 28 authentic physiological features extracted from WESAD chest sensors
FEATURE_COLUMNS = [
    # Electrocardiogram (ECG) features
    "ECG_mean", "ECG_std", "ECG_min", "ECG_max", "ECG_median",
    # Cardiac Autonomic Modulation
    "Heart_Rate", "HRV_RMSSD",
    # Electrodermal Activity (EDA / Skin Conductance)
    "EDA_mean", "EDA_std", "EDA_min", "EDA_max", "EDA_median", "EDA_slope",
    # Peripheral Skin Temperature
    "Temp_mean", "Temp_std", "Temp_min", "Temp_max", "Temp_median", "Temp_slope",
    # Respiration Rate & Waveform
    "Resp_mean", "Resp_std", "Resp_min", "Resp_max", "Resp_median", "Respiratory_Rate",
    # Accelerometer (Motion & Kinetic Artifacts)
    "ACC_mean", "ACC_std", "ACC_max"
]

TARGET_COLUMN = "label"
SUBJECT_COLUMN = "subject_id"

# WESAD ground truth class definitions
LABEL_NAMES = {
    0: "Baseline",
    1: "Stress",
    2: "Amusement"
}

# Scientific badge colors
LABEL_COLORS = {
    0: "#10b981",  # Emerald Green for Baseline Homeostasis
    1: "#f43f5e",  # Coral Red for Acute Sympathetic Stress
    2: "#f59e0b"   # Amber Gold for Amusement / Emotional Arousal
}


def load_wesad_dataset(csv_path=DATASET_FILENAME):
    """
    Loads the processed real WESAD feature dataset.
    If missing, runs the processor to extract windows directly from the official Siegen archive.
    """
    if not os.path.exists(csv_path):
        # Fallback check in parent directory
        parent_path = os.path.join(os.path.dirname(BASE_DIR), "wesad_processed_dataset.csv")
        if os.path.exists(parent_path):
            csv_path = parent_path
        else:
            raise FileNotFoundError(
                f"WESAD processed dataset not found at '{csv_path}'. "
                "Please run 'python process_wesad_dataset.py' to extract genuine WESAD windows."
            )
            
    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    
    # 1. Duplicate check
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[PREPROCESSING] Dropped {duplicates} duplicate rows.")
        
    # 2. Missing value verification
    missing = df[FEATURE_COLUMNS].isnull().sum().sum()
    if missing > 0:
        for col in FEATURE_COLUMNS:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        print(f"[PREPROCESSING] Imputed {missing} missing values with column median.")
        
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    print(f"[PREPROCESSING] Loaded {len(df)} genuine WESAD records (from {initial_rows} raw).")
    return df


def prepare_subject_independent_split(df, test_subject="S4", apply_baseline_norm=True):
    """
    Prepares a strict SUBJECT-INDEPENDENT Train/Test split:
    - Training Set: All participants except `test_subject` (e.g., S2 and S3).
    - Test Set: Strictly `test_subject` (e.g., S4).
    
    CRITICAL BIOMEDICAL RULE:
    Zero participant leakage! Overlapping windows from the same individual
    are NEVER mixed between train and test sets.
    
    Optionally applies Subject-Specific Baseline Normalization (Schmidt et al., ICMI 2018):
    z = (x - mean_baseline) / std_baseline
    This accounts for individual basal skin conductance and temperature differences.
    """
    df_clean = df.copy()
    
    if apply_baseline_norm:
        # Standardize features relative to each participant's own resting baseline
        for subj in df_clean[SUBJECT_COLUMN].unique():
            subj_mask = df_clean[SUBJECT_COLUMN] == subj
            base_mask = subj_mask & (df_clean[TARGET_COLUMN] == 0)
            
            base_means = df_clean.loc[base_mask, FEATURE_COLUMNS].mean()
            base_stds = df_clean.loc[base_mask, FEATURE_COLUMNS].std().replace(0, 1.0).fillna(1.0)
            
            df_clean.loc[subj_mask, FEATURE_COLUMNS] = (
                df_clean.loc[subj_mask, FEATURE_COLUMNS] - base_means
            ) / base_stds
            
        df_clean[FEATURE_COLUMNS] = df_clean[FEATURE_COLUMNS].fillna(0.0)
        
    train_mask = df_clean[SUBJECT_COLUMN] != test_subject
    test_mask = df_clean[SUBJECT_COLUMN] == test_subject
    
    train_df = df_clean.loc[train_mask].reset_index(drop=True)
    test_df = df_clean.loc[test_mask].reset_index(drop=True)
    
    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].copy()
    
    # Fit StandardScaler strictly on training subjects
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURE_COLUMNS
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_COLUMNS
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, train_df, test_df


def save_scaler(scaler, filepath=SCALER_FILENAME):
    """Saves fitted StandardScaler to disk using Joblib."""
    joblib.dump(scaler, filepath)
    print(f"[SAVED] Fitted WESAD scaler -> '{filepath}'.")


def load_scaler(filepath=SCALER_FILENAME):
    """Loads fitted StandardScaler from disk."""
    if not os.path.exists(filepath):
        parent_path = os.path.join(os.path.dirname(BASE_DIR), "wesad_scaler.pkl")
        if os.path.exists(parent_path):
            filepath = parent_path
        else:
            raise FileNotFoundError(f"Scaler file not found at '{filepath}'.")
    return joblib.load(filepath)


if __name__ == "__main__":
    print("\n=======================================================")
    print("   GENUINE WESAD DATA PREPROCESSING & VERIFICATION     ")
    print("=======================================================\n")
    
    dataset = load_wesad_dataset()
    print("\n--- Participants Breakdown ---")
    print(dataset[SUBJECT_COLUMN].value_counts())
    
    print("\n--- Target Class Distribution ---")
    for cls_id in [0, 1, 2]:
        cnt = (dataset[TARGET_COLUMN] == cls_id).sum()
        print(f"  Class {cls_id} ({LABEL_NAMES[cls_id]}): {cnt} windows ({cnt / len(dataset) * 100:.1f}%)")
        
    X_tr, X_te, y_tr, y_te, sc, tr_df, te_df = prepare_subject_independent_split(dataset, test_subject="S4")
    print("\n--- Subject-Independent Train/Test Split ---")
    print(f"  Training Windows (S2, S3): {len(X_tr)}")
    print(f"  Testing Windows (S4):      {len(X_te)}")
    print(f"  Feature Count:             {X_tr.shape[1]}")
    save_scaler(sc)
