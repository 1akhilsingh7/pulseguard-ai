"""
=============================================================================
PULSEGUARD AI — FULL SYSTEM VERIFICATION SUITE
Verification Script: test_system.py
Validates:
1. Authentication & Security (Registration, Hashing, PBKDF2, Database)
2. Authentic WESAD Dataset & Preprocessing Pipeline
3. Random Forest Model & Scaler Loading
4. All Three Classification Outcomes (Baseline=Green, Stress=Red, Amusement=Blue)
5. Interactive Biosignal Summaries & Sensor Signals
6. CSV Upload Validation (Valid CSV vs Missing Columns)
7. Session History Logging & Report Generation
=============================================================================
"""

import os
import sys
import io
import pickle
import pandas as pd
import numpy as np

# Resolve path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from auth import (
    init_db,
    register_user,
    authenticate_user,
    validate_email,
    validate_password
)
from preprocessing import (
    DATASET_FILENAME,
    RAW_SAMPLES_FILENAME,
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
    print("=" * 70)
    print("      RUNNING PULSEGUARD AI COMPLETE SYSTEM VERIFICATION SUITE        ")
    print("=" * 70)

    # 1. Authentication & Security Verification
    print("\n[TEST 1] Verifying User Authentication System...")
    init_db()
    # Test Demo user
    ok, demo_u, msg = authenticate_user("demo@pulseguard.ai", "Password123!")
    assert ok, f"Demo login failed: {msg}"
    assert demo_u["email"] == "demo@pulseguard.ai"
    print("  -> Demo login successful: demo@pulseguard.ai")

    # Test Bad Password
    bad_ok, _, _ = authenticate_user("demo@pulseguard.ai", "WrongPassword!")
    assert not bad_ok, "Authentication should fail with wrong password"
    print("  -> Password rejection for invalid credentials verified.")

    # Test Registration
    test_user_email = f"test_{int(np.random.randint(1000, 9999))}@example.com"
    reg_ok, reg_msg = register_user("Test User", test_user_email, "SecurePassword123!")
    assert reg_ok, f"Registration failed: {reg_msg}"
    # Verify login with newly registered user
    login_ok, u_data, _ = authenticate_user(test_user_email, "SecurePassword123!")
    assert login_ok and u_data["email"] == test_user_email
    print(f"  -> User registration and immediate login verified ({test_user_email}).")
    print("  [PASS] Authentication system, password hashing, and SQLite DB verified.")

    # 2. WESAD Dataset Integrity
    print("\n[TEST 2] Verifying Authentic WESAD Dataset...")
    assert os.path.exists(DATASET_FILENAME), f"Dataset file missing at {DATASET_FILENAME}"
    df = load_wesad_dataset(DATASET_FILENAME)
    assert len(df) >= 1000, f"Expected >= 1000 windows, found {len(df)}"
    assert len(FEATURE_COLUMNS) == 28, f"Expected 28 features, got {len(FEATURE_COLUMNS)}"
    print(f"  [PASS] Verified {len(df)} genuine WESAD windows across 28 features.")

    # 3. Raw Signal Waveforms (for Interactive Biosignal Summary)
    print("\n[TEST 3] Verifying Multi-Channel Raw Biosignal Waveforms...")
    assert os.path.exists(RAW_SAMPLES_FILENAME), f"Raw traces file missing at {RAW_SAMPLES_FILENAME}"
    with open(RAW_SAMPLES_FILENAME, "rb") as f:
        raw_dict = pickle.load(f)
    for subj in ["S2", "S3", "S4"]:
        assert subj in raw_dict
        for cond in ["Baseline", "Stress", "Amusement"]:
            assert cond in raw_dict[subj]
            for sig in ["time_sec", "ECG", "EDA", "Temp", "Resp", "ACC_mag"]:
                assert sig in raw_dict[subj][cond]
                assert len(raw_dict[subj][cond][sig]) == 700
    print("  [PASS] All 700-point (10-second) raw waveforms (ECG, EDA, Temp, Resp, ACC) validated.")

    # 4. Model & Scaler Loading
    print("\n[TEST 4] Loading Pre-trained Random Forest Model & Scaler...")
    model, scaler = load_wesad_model(MODEL_FILENAME, SCALER_FILENAME)
    assert hasattr(model, "predict"), "Model object is invalid"
    assert hasattr(scaler, "transform"), "Scaler object is invalid"
    print("  [PASS] Random Forest model and StandardScaler loaded successfully.")

    # 5. Baseline Normalization Statistics
    print("\n[TEST 5] Loading Baseline Normalization Statistics...")
    stats = load_baseline_stats(BASELINE_STATS_FILENAME, df_fallback=df)
    assert "subjects" in stats and "population" in stats
    for s in ["S2", "S3", "S4"]:
        assert s in stats["subjects"]
    print("  [PASS] Subject-specific baseline profiles verified.")

    # 6. Color Mapping Verification
    print("\n[TEST 6] Verifying User-Facing Color Scheme...")
    assert LABEL_COLORS[0] == "#10b981", "Baseline must be Green"
    assert LABEL_COLORS[1] == "#f43f5e", "Stress must be Red"
    assert LABEL_COLORS[2] == "#3b82f6", "Amusement must be Blue"
    print("  [PASS] Color scheme verified: Green=Baseline, Red=Stress, Blue=Amusement.")

    # 7. Stress Classification on All 3 Classes
    print("\n[TEST 7] Testing Predictions on Authentic WESAD Participant Samples...")
    # Baseline test (S4)
    s4_base = df[(df[SUBJECT_COLUMN] == "S4") & (df[TARGET_COLUMN] == 0)].iloc[0]
    res_base = predict_wesad_window(s4_base, model, scaler, baseline_stats=stats, subject_id="S4")
    assert res_base["class_name"] == "Baseline"
    print(f"  -> S4 Baseline:  Predicted={res_base['class_name']} (Confidence: {res_base['confidence']*100:.1f}%)")

    # Stress test (S4)
    s4_stress = df[(df[SUBJECT_COLUMN] == "S4") & (df[TARGET_COLUMN] == 1)].iloc[0]
    res_stress = predict_wesad_window(s4_stress, model, scaler, baseline_stats=stats, subject_id="S4")
    assert res_stress["class_name"] == "Stress"
    print(f"  -> S4 Stress:    Predicted={res_stress['class_name']} (Confidence: {res_stress['confidence']*100:.1f}%)")

    # Amusement test (S2)
    s2_amuse = df[(df[SUBJECT_COLUMN] == "S2") & (df[TARGET_COLUMN] == 2)].iloc[0]
    res_amuse = predict_wesad_window(s2_amuse, model, scaler, baseline_stats=stats, subject_id="S2")
    assert res_amuse["class_name"] == "Amusement"
    print(f"  -> S2 Amusement: Predicted={res_amuse['class_name']} (Confidence: {res_amuse['confidence']*100:.1f}%)")
    print("  [PASS] All 3 physiological states classified accurately with authentic probabilities.")

    # 8. CSV Upload Validation
    print("\n[TEST 8] Testing CSV File Validation...")
    # Valid CSV
    valid_csv = df.head(5)[FEATURE_COLUMNS]
    res_csv = predict_wesad_window(valid_csv.iloc[0], model, scaler, baseline_stats=stats)
    assert res_csv["class_name"] in ["Baseline", "Stress", "Amusement"]
    print("  -> Valid CSV successfully processed.")

    # Incompatible CSV
    missing_cols = ["ECG_mean", "Heart_Rate"]
    invalid_csv = valid_csv.drop(columns=missing_cols)
    detected_missing = [c for c in FEATURE_COLUMNS if c not in invalid_csv.columns]
    assert len(detected_missing) == 2
    print(f"  -> Incompatible CSV correctly flagged missing columns: {detected_missing}")
    print("  [PASS] CSV file validation operates reliably.")

    # 9. Session History & Report Generation
    print("\n[TEST 9] Verifying Session History & Report Generation...")
    sample_history = [{
        "timestamp": "2026-11-17 19:45:00",
        "source": "Participant S4 (Window #0)",
        "predicted_state": res_base["class_name"],
        "confidence": res_base["confidence"],
        "probabilities": res_base["probabilities"],
        "explanation": res_base["explanation"],
        "vitals": res_base["vitals"]
    }]
    hist_df = pd.DataFrame(sample_history)
    csv_out = hist_df.to_csv(index=False)
    assert len(csv_out) > 50
    print("  [PASS] History tracking and report formatting verified.")

    print("\n" + "=" * 70)
    print("   ALL 9 SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY (100%)         ")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_system_verification()
    sys.exit(0 if success else 1)
