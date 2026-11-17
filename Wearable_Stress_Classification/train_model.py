"""
=============================================================================
PULSEGUARD AI — WEARABLE-SIGNAL STRESS CLASSIFICATION
Module: train_model.py
Description: Supervised learning model training using Random Forest Classifier,
             authentic subject-independent evaluation, Stress Recall optimization,
             Gini feature importance analysis, and Joblib model persistence.
Dataset: Genuine WESAD physiological recordings (University of Siegen & UCI ML)
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
    BASELINE_STATS_FILENAME,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SUBJECT_COLUMN,
    LABEL_NAMES,
    LABEL_COLORS,
    load_wesad_dataset,
    prepare_subject_independent_split,
    save_scaler,
    load_scaler,
    load_baseline_stats,
    compute_and_save_baseline_stats,
    normalize_sample
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
    """
    print("=================================================================")
    print("      TRAINING RANDOM FOREST CLASSIFIER ON REAL WESAD DATA       ")
    print("=================================================================")
    
    # 1. Load dataset
    df = load_wesad_dataset(data_path)
    
    # Compute and save baseline normalization statistics
    compute_and_save_baseline_stats(df, BASELINE_STATS_FILENAME)
    
    # 2. Subject-independent split & baseline normalization
    X_train_sc, X_test_sc, y_train, y_test, scaler, train_df, test_df = prepare_subject_independent_split(
        df, test_subject=test_subject, apply_baseline_norm=True
    )
    
    save_scaler(scaler, scaler_path)
    
    # 3. Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train_sc, y_train)
    
    joblib.dump(rf_model, model_path)
    print(f"  -> Model saved to '{model_path}'.")
    
    # 4. Evaluate on unseen participant test data
    results = evaluate_wesad_model(rf_model, scaler, df, test_subject=test_subject)
    return rf_model, scaler, results


def evaluate_wesad_model(model, scaler, df, test_subject="S4"):
    """Computes evaluation metrics on held-out test participant."""
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
    
    class_metrics = {}
    for cls_name in target_names:
        if cls_name in report_dict:
            class_metrics[cls_name] = {
                "precision": report_dict[cls_name]["precision"],
                "recall": report_dict[cls_name]["recall"],
                "f1": report_dict[cls_name]["f1-score"],
                "support": int(report_dict[cls_name]["support"])
            }
            
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    
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
    """Loads saved model and scaler. If missing, automatically trains and saves them."""
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        parent_model = os.path.join(os.path.dirname(BASE_DIR), "wesad_rf_model.pkl")
        parent_scaler = os.path.join(os.path.dirname(BASE_DIR), "wesad_scaler.pkl")
        if os.path.exists(parent_model) and os.path.exists(parent_scaler):
            model_path = parent_model
            scaler_path = parent_scaler
        else:
            model, scaler, _ = train_wesad_model(model_path=model_path, scaler_path=scaler_path)
            return model, scaler
            
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_wesad_window(features_series, model, scaler, baseline_stats=None, subject_id=None, is_normalized=False):
    """
    Inference helper for a single window of 28 physiological features.
    
    Parameters:
    - features_series: dict, pd.Series, or pd.DataFrame containing feature columns
    - model: trained RandomForestClassifier
    - scaler: fitted StandardScaler
    - baseline_stats: precomputed baseline statistics for z-score normalization
    - subject_id: optional subject identifier (e.g., 'S2', 'S3', 'S4')
    - is_normalized: True if input features are already baseline-normalized z-scores
    """
    if isinstance(features_series, pd.Series):
        feat_dict = features_series.to_dict()
    elif isinstance(features_series, dict):
        feat_dict = features_series
    elif isinstance(features_series, pd.DataFrame):
        feat_dict = features_series.iloc[0].to_dict()
    elif isinstance(features_series, np.ndarray):
        feat_dict = {col: features_series.flatten()[i] for i, col in enumerate(FEATURE_COLUMNS)}
    else:
        raise ValueError("Unsupported input format for features_series.")
        
    # Baseline normalization if needed
    if not is_normalized and baseline_stats is not None:
        x_df = normalize_sample(feat_dict, baseline_stats, subject_id=subject_id)
    else:
        x_df = pd.DataFrame([[feat_dict.get(c, 0.0) for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        
    x_scaled = pd.DataFrame(scaler.transform(x_df), columns=FEATURE_COLUMNS)
    
    pred_class = int(model.predict(x_scaled)[0])
    pred_probs = model.predict_proba(x_scaled)[0]
    
    prob_dict = {
        LABEL_NAMES[i]: float(pred_probs[i])
        for i in range(len(LABEL_NAMES))
    }
    
    # Generate simple, clear, user-friendly physiological explanation
    hr = float(feat_dict.get("Heart_Rate", 0.0))
    hrv = float(feat_dict.get("HRV_RMSSD", 0.0))
    eda_mean = float(feat_dict.get("EDA_mean", 0.0))
    resp_rate = float(feat_dict.get("Respiratory_Rate", 0.0))
    
    if pred_class == 1:  # Stress
        explanation = (
            f"Acute sympathetic nervous system arousal detected. Key physiological markers indicate elevated "
            f"skin conductance (EDA: {eda_mean:.2f} µS) and heightened cardiac rhythm (HR: {hr:.1f} BPM) "
            f"with suppressed autonomic variability (HRV RMSSD: {hrv:.1f} ms), characteristic of an active stress response."
        )
    elif pred_class == 0:  # Baseline
        explanation = (
            f"Homeostatic baseline equilibrium detected. Cardiac patterns exhibit healthy vagal tone "
            f"(HR: {hr:.1f} BPM, HRV RMSSD: {hrv:.1f} ms), skin conductance is quiescent ({eda_mean:.2f} µS), "
            f"and respiratory rhythm is calm and regular ({resp_rate:.1f} br/min)."
        )
    else:  # Amusement
        explanation = (
            f"Positive emotional arousal detected. Physiological signals show moderate sympathetic engagement "
            f"(HR: {hr:.1f} BPM, EDA: {eda_mean:.2f} µS) combined with characteristic respiratory modulation, "
            f"reflecting positive emotional valence rather than distress."
        )
        
    return {
        "pred_class": pred_class,
        "class_name": LABEL_NAMES[pred_class],
        "probabilities": prob_dict,
        "confidence": prob_dict[LABEL_NAMES[pred_class]],
        "explanation": explanation,
        "vitals": {
            "Heart_Rate": hr,
            "HRV_RMSSD": hrv,
            "EDA_mean": eda_mean,
            "Respiratory_Rate": resp_rate
        }
    }
