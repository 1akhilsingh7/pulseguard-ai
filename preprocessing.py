"""
=============================================================================
PULSEGUARD AI — WEARABLE-SIGNAL STRESS CLASSIFICATION
Module: preprocessing.py
Description: Data loading, cleaning, feature selection, subject-specific
             baseline normalization, and train/test splitting based on
             genuine WESAD physiological recordings.
Dataset: WESAD Benchmark (University of Siegen / ACM ICMI 2018)
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
BASELINE_STATS_FILENAME = os.path.join(BASE_DIR, "wesad_baseline_stats.pkl")

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

# User-Facing Healthcare Product Colors:
# Green = Baseline, Red = Stress, Blue = Amusement
LABEL_COLORS = {
    0: "#10b981",  # Vibrant Green for Baseline
    1: "#f43f5e",  # Coral Red for Stress
    2: "#3b82f6"   # Electric Blue for Amusement
}


def load_wesad_dataset(csv_path=DATASET_FILENAME):
    """Loads the processed real WESAD feature dataset."""
    if not os.path.exists(csv_path):
        parent_path = os.path.join(os.path.dirname(BASE_DIR), "wesad_processed_dataset.csv")
        if os.path.exists(parent_path):
            csv_path = parent_path
        else:
            raise FileNotFoundError(
                f"WESAD processed dataset not found at '{csv_path}'. "
                "Please run 'python process_wesad_dataset.py' to extract genuine WESAD windows."
            )
            
    df = pd.read_csv(csv_path)
    
    # Clean duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        
    # Verify missing values
    missing = df[FEATURE_COLUMNS].isnull().sum().sum()
    if missing > 0:
        for col in FEATURE_COLUMNS:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def compute_and_save_baseline_stats(df, filepath=BASELINE_STATS_FILENAME):
    """
    Computes and saves subject-specific resting baseline mean and std for each feature.
    Also computes population baseline statistics as a robust fallback.
    """
    base_df = df[df[TARGET_COLUMN] == 0]
    stats = {
        "subjects": {},
        "population": {
            "mean": base_df[FEATURE_COLUMNS].mean().to_dict(),
            "std": base_df[FEATURE_COLUMNS].std().replace(0, 1.0).fillna(1.0).to_dict()
        }
    }
    for subj in df[SUBJECT_COLUMN].unique():
        s_base = df[(df[SUBJECT_COLUMN] == subj) & (df[TARGET_COLUMN] == 0)][FEATURE_COLUMNS]
        if len(s_base) > 0:
            stats["subjects"][subj] = {
                "mean": s_base.mean().to_dict(),
                "std": s_base.std().replace(0, 1.0).fillna(1.0).to_dict()
            }
            
    joblib.dump(stats, filepath)
    return stats


def load_baseline_stats(filepath=BASELINE_STATS_FILENAME, df_fallback=None):
    """Loads baseline normalization statistics."""
    if not os.path.exists(filepath):
        parent_path = os.path.join(os.path.dirname(BASE_DIR), "wesad_baseline_stats.pkl")
        if os.path.exists(parent_path):
            filepath = parent_path
        else:
            if df_fallback is None:
                df_fallback = load_wesad_dataset()
            return compute_and_save_baseline_stats(df_fallback, filepath)
    return joblib.load(filepath)


def normalize_sample(features_input, baseline_stats, subject_id=None):
    """
    Normalizes raw physiological features using the subject's resting baseline
    (or population baseline if subject is unknown or new).
    z = (x - mean_baseline) / std_baseline
    """
    if isinstance(features_input, pd.Series):
        feat_dict = features_input.to_dict()
    elif isinstance(features_input, dict):
        feat_dict = features_input
    elif isinstance(features_input, pd.DataFrame):
        feat_dict = features_input.iloc[0].to_dict()
    else:
        raise ValueError("Unsupported input format for features_input.")
        
    # Select baseline parameters
    if subject_id and subject_id in baseline_stats.get("subjects", {}):
        ref_mean = baseline_stats["subjects"][subject_id]["mean"]
        ref_std = baseline_stats["subjects"][subject_id]["std"]
    else:
        ref_mean = baseline_stats["population"]["mean"]
        ref_std = baseline_stats["population"]["std"]
        
    norm_row = {}
    for col in FEATURE_COLUMNS:
        val = float(feat_dict.get(col, 0.0))
        m = float(ref_mean.get(col, 0.0))
        s = float(ref_std.get(col, 1.0))
        if s == 0:
            s = 1.0
        norm_row[col] = (val - m) / s
        
    return pd.DataFrame([norm_row], columns=FEATURE_COLUMNS)


def prepare_subject_independent_split(df, test_subject="S4", apply_baseline_norm=True):
    """
    Prepares a strict SUBJECT-INDEPENDENT Train/Test split:
    - Training Set: All participants except `test_subject` (S2 and S3).
    - Test Set: Strictly `test_subject` (S4).
    """
    df_clean = df.copy()
    
    if apply_baseline_norm:
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


def load_scaler(filepath=SCALER_FILENAME):
    """Loads fitted StandardScaler from disk."""
    if not os.path.exists(filepath):
        parent_path = os.path.join(os.path.dirname(BASE_DIR), "wesad_scaler.pkl")
        if os.path.exists(parent_path):
            filepath = parent_path
        else:
            raise FileNotFoundError(f"Scaler file not found at '{filepath}'.")
    return joblib.load(filepath)
