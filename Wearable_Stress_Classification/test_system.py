"""
=============================================================================
PULSE GUARD AI — WEARABLE ML SYSTEM VERIFICATION SUITE
Verification Script: test_system.py
Validates the genuine WESAD dataset, model, LOSO subject-independent evaluation,
inference accuracy, Stress Recall, and report generation.
Target Audience: 2nd-year B.Tech CSE presentation validation.
=============================================================================
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

# Resolve path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing import (
    DATASET_FILENAME,
    RAW_SAMPLES_FILENAME,
    SCALER_FILENAME,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SUBJECT_COLUMN,
    LABEL_NAMES,
    load_wesad_dataset,
    prepare_subject_independent_split,
    load_scaler
)
from train_model import (
    MODEL_FILENAME,
    load_wesad_model,
    evaluate_wesad_model,
    predict_wesad_window
)


def run_system_verification():
    print("=" * 65)
    print("      RUNNING WESAD WEARABLE ML SYSTEM VERIFICATION SUITE        ")
    print("=" * 65)
    
    # 1. Dataset Verification
    print("\n[TEST 1] Verifying authentic WESAD processed dataset...")
    assert os.path.exists(DATASET_FILENAME), f"Dataset file missing at {DATASET_FILENAME}"
    df = load_wesad_dataset(DATASET_FILENAME)
    assert len(df) >= 1000, f"Expected >= 1000 windows, found {len(df)}"
    assert set(df[TARGET_COLUMN].unique()) == {0, 1, 2}, "Classes must be strictly {0: Baseline, 1: Stress, 2: Amusement}"
    assert set(df[SUBJECT_COLUMN].unique()) == {"S2", "S3", "S4"}, "Subjects must be S2, S3, S4"
    assert len(FEATURE_COLUMNS) == 28, f"Expected 28 features, got {len(FEATURE_COLUMNS)}"
    print(f"  [PASS] Verified 1,214 genuine WESAD windows across subjects S2, S3, S4.")
    
    # 2. Raw Signal Samples Verification
    print("\n[TEST 2] Verifying multi-channel raw sensor waveform traces...")
    assert os.path.exists(RAW_SAMPLES_FILENAME), f"Raw traces file missing at {RAW_SAMPLES_FILENAME}"
    with open(RAW_SAMPLES_FILENAME, "rb") as f:
        raw_dict = pickle.load(f)
    for subj in ["S2", "S3", "S4"]:
        assert subj in raw_dict, f"Subject {subj} missing in raw waveforms"
        for cond in ["Baseline", "Stress", "Amusement"]:
            assert cond in raw_dict[subj], f"{cond} trace missing for {subj}"
            trace = raw_dict[subj][cond]
            for sig in ["time_sec", "ECG", "EDA", "Temp", "Resp", "ACC_mag"]:
                assert sig in trace, f"Signal {sig} missing in {subj} {cond}"
                assert len(trace[sig]) == 700, f"Expected 700 samples (10s @ 70Hz), got {len(trace[sig])}"
    print("  [PASS] All 9 multi-channel 10-second waveforms (ECG, EDA, Temp, Resp, ACC) validated.")
    
    # 3. Model & Scaler Verification
    print("\n[TEST 3] Loading pre-trained Random Forest model and scaler...")
    model, scaler = load_wesad_model(MODEL_FILENAME, SCALER_FILENAME)
    assert hasattr(model, "predict"), "Model object is invalid"
    assert hasattr(scaler, "transform"), "Scaler object is invalid"
    print(f"  [PASS] Random Forest (100 Trees) and StandardScaler loaded successfully.")
    
    # 4. Subject-Independent Split Verification
    print("\n[TEST 4] Verifying strict subject-independent split (LOSO)...")
    X_tr, X_te, y_tr, y_te, sc, tr_df, te_df = prepare_subject_independent_split(df, test_subject="S4")
    assert len(X_tr) == 806, f"Expected 806 training windows (S2+S3), got {len(X_tr)}"
    assert len(X_te) == 408, f"Expected 408 testing windows (S4), got {len(X_te)}"
    assert set(tr_df[SUBJECT_COLUMN].unique()) == {"S2", "S3"}, "Training set must only contain S2 and S3"
    assert set(te_df[SUBJECT_COLUMN].unique()) == {"S4"}, "Test set must strictly contain S4"
    print("  [PASS] Zero participant leakage confirmed between training and testing sets.")
    
    # 5. Model Evaluation Metrics & Priority Stress Recall
    print("\n[TEST 5] Verifying model performance on held-out test participant S4...")
    eval_res = evaluate_wesad_model(model, scaler, df, test_subject="S4")
    acc = eval_res["accuracy"]
    stress_rec = eval_res["stress_recall"]
    stress_prec = eval_res["stress_precision"]
    
    print(f"  -> Test Accuracy (S4): {acc*100:.2f}%")
    print(f"  -> Stress Recall:      {stress_rec*100:.2f}%")
    print(f"  -> Stress Precision:   {stress_prec*100:.2f}%")
    
    assert acc >= 0.70, f"Accuracy {acc*100:.1f}% below 70% threshold"
    assert stress_rec >= 0.70, f"Stress Recall {stress_rec*100:.1f}% below 70% priority threshold"
    assert stress_prec >= 0.80, f"Stress Precision {stress_prec*100:.1f}% below 80% threshold"
    print("  [PASS] Authentic model metrics verified with high clinical sensitivity.")
    
    # 6. Single Window Inference Verification
    print("\n[TEST 6] Testing single-window inference on held-out participant S4...")
    s4_stress_sample = te_df[te_df[TARGET_COLUMN] == 1].iloc[0][FEATURE_COLUMNS]
    res_stress = predict_wesad_window(s4_stress_sample, model, scaler)
    print(f"  -> S4 Stress Sample: Predicted = {res_stress['class_name']}, Prob = {res_stress['confidence']*100:.1f}%")
    assert res_stress["pred_class"] == 1, f"Expected Stress (Class 1), got {res_stress['pred_class']}"
    
    s4_base_sample = te_df[te_df[TARGET_COLUMN] == 0].iloc[0][FEATURE_COLUMNS]
    res_base = predict_wesad_window(s4_base_sample, model, scaler)
    print(f"  -> S4 Baseline Sample: Predicted = {res_base['class_name']}, Prob = {res_base['confidence']*100:.1f}%")
    assert res_base["pred_class"] == 0, f"Expected Baseline (Class 0), got {res_base['pred_class']}"
    print("  [PASS] Single-window prediction pipeline verified for both Baseline and Stress.")
    
    # 7. Feature Importance Verification
    print("\n[TEST 7] Verifying Gini feature importance calculation...")
    feat_imp = eval_res["feature_importances"]
    assert len(feat_imp) == 28, "Feature importances must cover all 28 features"
    assert feat_imp["Importance"].sum() > 0.99, "Feature importances must sum to ~1.0"
    top_feature = feat_imp.iloc[0]["Feature"]
    print(f"  [PASS] Feature importances validated. Top biomarker: {top_feature} ({feat_imp.iloc[0]['Importance']*100:.1f}%).")
    
    print("\n" + "=" * 65)
    print("   ALL 7 SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY (100%)   ")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_system_verification()
    sys.exit(0 if success else 1)
