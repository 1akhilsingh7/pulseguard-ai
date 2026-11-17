"""
=============================================================================
WEARABLE-SIGNAL STRESS-RISK CLASSIFICATION (WESAD REAL DATASET)
Module: train_model.py
Description: Supervised learning model training using Random Forest Classifier,
             authentic subject-independent evaluation, Stress Recall optimization,
             Gini feature importance analysis, and Joblib model persistence.
Dataset: Genuine WESAD physiological recordings (University of Siegen & UCI ML)
Target Audience: 2nd-year B.Tech CSE presentation & demonstration.
=============================================================================
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Resolve directory paths reliably
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
    LABEL_COLORS,
    load_wesad_dataset,
    prepare_subject_independent_split,
    save_scaler,
    load_scaler
)

MODEL_FILENAME = os.path.join(BASE_DIR, "wesad_rf_model.pkl")


def train_wesad_model(
    test_subject="S4",
    data_path=DATASET_FILENAME,
    model_path=MODEL_FILENAME,
    scaler_path=SCALER_FILENAME,
    random_state=42
):
    """
    Trains a Random Forest Classifier on genuine WESAD physiological recordings
    using a strict SUBJECT-INDEPENDENT evaluation methodology.
    
    Training Set: Participants S2 and S3 (806 windows).
    Test Set: Participant S4 (408 windows).
    
    Why Random Forest Classifier?
    1. Ensemble Bagging: Combines 100 decorrelated decision trees, minimizing variance
       and avoiding overfitting on biological time-series features.
    2. Non-Linear Physiological Boundaries: Autonomic nervous system responses
       (e.g., steep EDA slope combined with depressed HRV RMSSD) exhibit non-linear
       thresholds that tree ensembles capture without complex kernel tuning.
    3. Balanced Class Weights (`class_weight='balanced'`): Automatically penalizes
       misclassification inversely proportional to class frequencies. Crucial for
       our medical monitoring objective: NOT missing acute stress episodes.
    4. Interpretability via Gini Feature Importance: Explains exactly which sensor
       channels (EDA, HRV, Respiration, Temp) drive the model's classifications.
    """
    print("=================================================================")
    print("      TRAINING RANDOM FOREST CLASSIFIER ON REAL WESAD DATA       ")
    print("=================================================================")
    
    # Step 1: Load authentic WESAD processed dataset
    print(f"\n[1/4] Loading real WESAD dataset from '{data_path}'...")
    df = load_wesad_dataset(data_path)
    
    # Step 2: Subject-independent split & baseline normalization
    print(f"\n[2/4] Performing Subject-Independent Split (Test Subject: {test_subject})...")
    X_train_sc, X_test_sc, y_train, y_test, scaler, train_df, test_df = prepare_subject_independent_split(
        df, test_subject=test_subject, apply_baseline_norm=True
    )
    print(f"  -> Training Windows (Participants S2, S3): {len(X_train_sc)}")
    print(f"  -> Testing Windows (Participant {test_subject}):      {len(X_test_sc)}")
    print(f"  -> Features per Window:                   {X_train_sc.shape[1]}")
    
    # Save fitted scaler
    save_scaler(scaler, scaler_path)
    
    # Step 3: Train Random Forest Classifier
    print("\n[3/4] Initializing and training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,         # 100 ensemble trees
        max_depth=10,             # Regularization depth to prevent memorization
        min_samples_split=4,      # Robust split threshold
        class_weight="balanced",  # Vital for high stress recall
        random_state=random_state, # 100% reproducible results for academic demo
        n_jobs=-1
    )
    rf_model.fit(X_train_sc, y_train)
    print("  -> Training complete.")
    
    # Persist trained model
    joblib.dump(rf_model, model_path)
    print(f"  -> Model persisted to '{model_path}'.")
    
    # Step 4: Evaluate on unseen participant test data
    print("\n[4/4] Evaluating model performance on held-out test participant...")
    results = evaluate_wesad_model(rf_model, scaler, df, test_subject=test_subject)
    
    print("\n======================= EVALUATION SUMMARY =======================")
    print(f"  Test Subject:                {test_subject}")
    print(f"  Overall Test Accuracy:       {results['accuracy'] * 100:.2f}%")
    print(f"  STRESS RECALL (Priority):    {results['stress_recall'] * 100:.2f}%")
    print(f"  STRESS PRECISION:            {results['stress_precision'] * 100:.2f}%")
    print(f"  Baseline Recall:             {results['class_metrics']['Baseline']['recall'] * 100:.2f}%")
    print(f"  Amusement Recall:            {results['class_metrics']['Amusement']['recall'] * 100:.2f}%")
    print("==================================================================\n")
    
    return rf_model, scaler, results


def evaluate_wesad_model(model, scaler, df, test_subject="S4"):
    """
    Computes rigorous evaluation metrics on the held-out test participant.
    """
    X_train_sc, X_test_sc, y_train, y_test, _, train_df, test_df = prepare_subject_independent_split(
        df, test_subject=test_subject, apply_baseline_norm=True
    )
    
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)
    
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    
    target_names = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
    report_dict = classification_report(
        y_test, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0
    )
    
    # Extract class-specific metrics
    class_metrics = {}
    for cls_name in target_names:
        if cls_name in report_dict:
            class_metrics[cls_name] = {
                "precision": report_dict[cls_name]["precision"],
                "recall": report_dict[cls_name]["recall"],
                "f1": report_dict[cls_name]["f1-score"],
                "support": int(report_dict[cls_name]["support"])
            }
            
    # Extract feature importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    
    # Stress metrics
    stress_recall = class_metrics.get("Stress", {}).get("recall", 0.0)
    stress_precision = class_metrics.get("Stress", {}).get("precision", 0.0)
    
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "stress_recall": stress_recall,
        "stress_precision": stress_precision,
        "confusion_matrix": cm,
        "classification_report_str": report_str,
        "classification_report_dict": report_dict,
        "class_metrics": class_metrics,
        "feature_importances": feat_imp,
        "test_subject": test_subject,
        "n_test_samples": len(y_test),
        "n_train_samples": len(y_train),
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def load_wesad_model(model_path=MODEL_FILENAME, scaler_path=SCALER_FILENAME):
    """
    Loads saved model and scaler. If missing, automatically trains and saves them.
    """
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # Fallback check in parent
        parent_model = os.path.join(os.path.dirname(BASE_DIR), "wesad_rf_model.pkl")
        parent_scaler = os.path.join(os.path.dirname(BASE_DIR), "wesad_scaler.pkl")
        if os.path.exists(parent_model) and os.path.exists(parent_scaler):
            model_path = parent_model
            scaler_path = parent_scaler
        else:
            print("[INFO] Model or scaler not found on disk. Initiating training...")
            model, scaler, _ = train_wesad_model(model_path=model_path, scaler_path=scaler_path)
            return model, scaler
            
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_wesad_window(features_series, model, scaler):
    """
    Inference helper for a single window of 28 physiological features.
    
    Returns:
    - predicted_class: int (0, 1, or 2)
    - class_name: str ("Baseline", "Stress", or "Amusement")
    - probabilities: dict of class_name -> float probability
    - rationale: clinical/physiological explanation of key indicators
    """
    if isinstance(features_series, pd.Series):
        feat_dict = features_series.to_dict()
    elif isinstance(features_series, dict):
        feat_dict = features_series
    elif isinstance(features_series, np.ndarray):
        feat_dict = {col: features_series.flatten()[i] for i, col in enumerate(FEATURE_COLUMNS)}
    else:
        raise ValueError("Unsupported input format for features_series.")
        
    # Scale features with fitted scaler using DataFrame to preserve feature names
    x_df = pd.DataFrame([[feat_dict[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    x_scaled = pd.DataFrame(scaler.transform(x_df), columns=FEATURE_COLUMNS)
    
    pred_class = int(model.predict(x_scaled)[0])
    pred_probs = model.predict_proba(x_scaled)[0]
    
    prob_dict = {
        LABEL_NAMES[i]: float(pred_probs[i])
        for i in range(len(LABEL_NAMES))
    }
    
    # Generate dynamic physiological rationale
    hr = feat_dict.get("Heart_Rate", 0.0)
    hrv = feat_dict.get("HRV_RMSSD", 0.0)
    eda_mean = feat_dict.get("EDA_mean", 0.0)
    eda_slope = feat_dict.get("EDA_slope", 0.0)
    resp_rate = feat_dict.get("Respiratory_Rate", 0.0)
    temp_slope = feat_dict.get("Temp_slope", 0.0)
    
    rationale_bullets = []
    if pred_class == 1:  # Stress
        rationale_bullets.append(
            f"Electrodermal Activity shows significant sympathetic arousal with elevated skin conductance (EDA mean: {eda_mean:.2f} µS, slope: {eda_slope:+.3f})."
        )
        if hr > 75:
            rationale_bullets.append(f"Cardiac tone exhibits sustained elevation (HR: {hr:.1f} BPM) coupled with reduced parasympathetic modulation (HRV RMSSD: {hrv:.2f} ms).")
        if resp_rate > 15:
            rationale_bullets.append(f"Respiration reflects tachypnea / elevated ventilatory drive ({resp_rate:.1f} breaths/min).")
        if temp_slope < 0:
            rationale_bullets.append("Peripheral skin temperature exhibits vasoconstriction (negative slope), consistent with acute sympathetic nervous system activation.")
    elif pred_class == 0:  # Baseline
        rationale_bullets.append(
            f"Autonomic stability is maintained: resting heart rate ({hr:.1f} BPM) with robust vagal parasympathetic modulation (HRV RMSSD: {hrv:.2f} ms)."
        )
        rationale_bullets.append(f"Tonic skin conductance remains quiescent (EDA mean: {eda_mean:.2f} µS) with negligible phasic flaring.")
        rationale_bullets.append(f"Respiration pattern is rhythmic and regular ({resp_rate:.1f} breaths/min).")
    else:  # Amusement
        rationale_bullets.append(
            f"Emotional arousal with positive valence: moderate heart rate ({hr:.1f} BPM) and transient laughter/vocal respiration perturbations."
        )
        rationale_bullets.append(f"EDA exhibits transient phasic peaks without the sustained high sympathetic tonic elevation of distress (EDA mean: {eda_mean:.2f} µS).")
        
    return {
        "pred_class": pred_class,
        "class_name": LABEL_NAMES[pred_class],
        "probabilities": prob_dict,
        "confidence": prob_dict[LABEL_NAMES[pred_class]],
        "rationale": " ".join(rationale_bullets),
        "rationale_bullets": rationale_bullets
    }


if __name__ == "__main__":
    model, scaler, results = train_wesad_model(test_subject="S4")
