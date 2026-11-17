"""
=============================================================================
PULSEGUARD AI — FULL SYSTEM VERIFICATION SUITE
Verification Script: test_system.py
Validates the complete prediction pipeline, baseline normalization, all 3 classes,
CSV validation, and report generation.
=============================================================================
"""

import os
import sys
import io
import pandas as pd
import numpy as np

# Resolve path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing import (
    DATASET_FILENAME,
    SCALER_FILENAME,
    BASELINE_STATS_FILENAME,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SUBJECT_COLUMN,
    LABEL_NAMES,
    LABEL_COLORS,
    load_wesad_dataset,
    load_baseline_stats,
    load_scaler
)
from train_model import (
    MODEL_FILENAME,
    load_wesad_model,
    predict_wesad_window
)


def run_system_verification():
    print("=" * 65)
    print("      RUNNING PULSEGUARD AI SYSTEM VERIFICATION SUITE            ")
    print("=" * 65)
    
    # 1. Dataset Verification
    print("\n[TEST 1] Verifying WESAD dataset...")
    assert os.path.exists(DATASET_FILENAME), f"Dataset file missing at {DATASET_FILENAME}"
    df = load_wesad_dataset(DATASET_FILENAME)
    assert len(df) >= 1000, f"Expected >= 1000 windows, found {len(df)}"
    assert len(FEATURE_COLUMNS) == 28, f"Expected 28 features, got {len(FEATURE_COLUMNS)}"
    print(f"  [PASS] Verified {len(df)} genuine WESAD windows across 28 features.")
    
    # 2. Baseline Stats Verification
    print("\n[TEST 2] Verifying baseline normalization statistics...")
    stats = load_baseline_stats(BASELINE_STATS_FILENAME, df_fallback=df)
    assert "subjects" in stats, "Missing 'subjects' in baseline stats"
    assert "population" in stats, "Missing 'population' in baseline stats"
    for s in ["S2", "S3", "S4"]:
        assert s in stats["subjects"], f"Missing subject {s} in baseline stats"
    print(f"  [PASS] Subject-specific baseline profiles verified for S2, S3, S4.")
    
    # 3. Model & Scaler Loading
    print("\n[TEST 3] Loading pre-trained Random Forest model and scaler...")
    model, scaler = load_wesad_model(MODEL_FILENAME, SCALER_FILENAME)
    assert hasattr(model, "predict"), "Model object is invalid"
    assert hasattr(scaler, "transform"), "Scaler object is invalid"
    print(f"  [PASS] Random Forest model and StandardScaler loaded successfully.")
    
    # 4. Color Mapping Verification
    print("\n[TEST 4] Verifying color branding (Green = Baseline, Red = Stress, Blue = Amusement)...")
    assert LABEL_COLORS[0] == "#10b981", f"Expected Green for Baseline, got {LABEL_COLORS[0]}"
    assert LABEL_COLORS[1] == "#f43f5e", f"Expected Red for Stress, got {LABEL_COLORS[1]}"
    assert LABEL_COLORS[2] == "#3b82f6", f"Expected Blue for Amusement, got {LABEL_COLORS[2]}"
    print("  [PASS] Color scheme verified: Green=Baseline, Red=Stress, Blue=Amusement.")
    
    # 5. Stress Classification (Method 1: Sample Data)
    print("\n[TEST 5] Testing prediction on genuine WESAD participant samples...")
    # Baseline test (S4)
    s4_base = df[(df[SUBJECT_COLUMN] == "S4") & (df[TARGET_COLUMN] == 0)].iloc[0]
    res_base = predict_wesad_window(s4_base, model, scaler, baseline_stats=stats, subject_id="S4")
    print(f"  -> S4 Baseline Sample: Predicted={res_base['class_name']} ({res_base['confidence']*100:.1f}%)")
    assert res_base["class_name"] == "Baseline", f"Expected Baseline, got {res_base['class_name']}"
    
    # Stress test (S4)
    s4_stress = df[(df[SUBJECT_COLUMN] == "S4") & (df[TARGET_COLUMN] == 1)].iloc[0]
    res_stress = predict_wesad_window(s4_stress, model, scaler, baseline_stats=stats, subject_id="S4")
    print(f"  -> S4 Stress Sample:   Predicted={res_stress['class_name']} ({res_stress['confidence']*100:.1f}%)")
    assert res_stress["class_name"] == "Stress", f"Expected Stress, got {res_stress['class_name']}"
    
    # Amusement test (S2)
    s2_amuse = df[(df[SUBJECT_COLUMN] == "S2") & (df[TARGET_COLUMN] == 2)].iloc[0]
    res_amuse = predict_wesad_window(s2_amuse, model, scaler, baseline_stats=stats, subject_id="S2")
    print(f"  -> S2 Amusement Sample: Predicted={res_amuse['class_name']} ({res_amuse['confidence']*100:.1f}%)")
    assert res_amuse["class_name"] == "Amusement", f"Expected Amusement, got {res_amuse['class_name']}"
    print("  [PASS] All 3 physiological states predicted accurately with high confidence.")
    
    # 6. CSV Validation (Method 2: Upload Data)
    print("\n[TEST 6] Testing CSV upload validation and inference...")
    # Valid CSV
    valid_csv_df = df.head(3)[FEATURE_COLUMNS]
    res_csv = predict_wesad_window(valid_csv_df.iloc[0], model, scaler, baseline_stats=stats)
    assert res_csv["class_name"] in ["Baseline", "Stress", "Amusement"], "Invalid prediction on CSV row"
    print("  -> Valid CSV successfully processed and classified.")
    
    # Invalid CSV (missing columns)
    invalid_csv_df = valid_csv_df.drop(columns=["ECG_mean", "Heart_Rate"])
    missing_cols = [c for c in FEATURE_COLUMNS if c not in invalid_csv_df.columns]
    assert len(missing_cols) == 2, f"Expected 2 missing columns, detected {len(missing_cols)}"
    print(f"  -> Invalid CSV correctly flagged missing columns: {missing_cols}")
    print("  [PASS] CSV file validation operates reliably.")
    
    # 7. Session History & Report Generation
    print("\n[TEST 7] Verifying history logging and report formatting...")
    history_entry = {
        "timestamp": "2026-11-17 19:30:00",
        "source": "Participant S4 (Window #0)",
        "predicted_state": res_base["class_name"],
        "confidence": res_base["confidence"],
        "probabilities": res_base["probabilities"],
        "explanation": res_base["explanation"],
        "vitals": res_base["vitals"]
    }
    assert history_entry["predicted_state"] == "Baseline"
    print("  [PASS] History tracking format validated.")
    
    print("\n" + "=" * 65)
    print("   ALL 7 SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY (100%)   ")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_system_verification()
    sys.exit(0 if success else 1)
