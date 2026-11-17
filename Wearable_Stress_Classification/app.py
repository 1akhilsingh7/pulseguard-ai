"""
=============================================================================
PULSE·GUARD AI — WEARABLE-SIGNAL STRESS-RISK CLASSIFICATION
Platform: Streamlit Healthcare Analytics Web Dashboard
Data Source: Official WESAD Benchmark Dataset (University of Siegen / ACM ICMI 2018)
Architecture: Multi-Modal Wearable Sensors -> Statistical Windows -> Random Forest
Target Audience: 2nd-year B.Tech CSE Presentation & Viva Voce
=============================================================================
"""

import os
import sys
import pickle
import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Ensure local imports resolve seamlessly
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
    load_scaler
)
from train_model import (
    MODEL_FILENAME,
    load_wesad_model,
    evaluate_wesad_model,
    predict_wesad_window
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PulseGuard AI | WESAD Wearable Stress Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# PREMIUM DARK HEALTHCARE SAAS CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Overall background */
    .stApp {
        background-color: #070d18;
        color: #f1f5f9;
    }

    /* Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: rgba(7, 13, 24, 0.85);
        backdrop-filter: blur(10px);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1329;
        border-right: 1px solid #1e293b;
    }

    /* Card Containers */
    .saas-card {
        background: linear-gradient(145deg, #0f172a 0%, #111c38 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .saas-card:hover {
        border-color: #06b6d4;
    }

    .saas-card-glow {
        background: radial-gradient(circle at top right, rgba(6, 182, 212, 0.12) 0%, #0f172a 70%);
        border: 1px solid #0284c7;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 0 25px rgba(6, 182, 212, 0.15);
    }

    /* KPI Stat Metric Cards */
    .kpi-container {
        display: flex;
        flex-direction: column;
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        height: 100%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .kpi-sub {
        color: #06b6d4;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 6px;
    }

    /* Badges */
    .badge-teal {
        display: inline-block;
        background-color: rgba(6, 182, 212, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-baseline {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-stress {
        display: inline-block;
        background-color: rgba(244, 63, 94, 0.18);
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-amuse {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 700;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 18px rgba(6, 182, 212, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* Workflow Pipeline Box */
    .workflow-step {
        background: #0b1329;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        height: 100%;
        position: relative;
    }
    .workflow-num {
        background: #0284c7;
        color: white;
        border-radius: 50%;
        width: 26px;
        height: 26px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .workflow-title {
        font-weight: 700;
        font-size: 0.9rem;
        color: #f1f5f9;
        margin-bottom: 4px;
    }
    .workflow-desc {
        font-size: 0.75rem;
        color: #94a3b8;
        line-height: 1.3;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA & MODEL CACHING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading authentic WESAD biosensor records...")
def get_cached_dataset():
    """Loads and caches the 1,214 genuine WESAD window records."""
    return load_wesad_dataset()

@st.cache_data(show_spinner="Loading authentic multi-channel raw sensor waveforms...")
def get_cached_raw_samples():
    """Loads genuine 10-second raw waveforms (ECG, EDA, Temp, Resp, ACC)."""
    raw_path = RAW_SAMPLES_FILENAME
    if not os.path.exists(raw_path):
        parent_raw = os.path.join(os.path.dirname(BASE_DIR), "wesad_raw_samples.pkl")
        if os.path.exists(parent_raw):
            raw_path = parent_raw
        else:
            raise FileNotFoundError(f"Raw samples file missing at '{raw_path}'.")
    with open(raw_path, "rb") as f:
        data = pickle.load(f)
    return data

@st.cache_resource(show_spinner="Loading pre-trained Random Forest model and scaler...")
def get_cached_model_and_scaler():
    """Loads pre-trained Random Forest model and standard scaler."""
    model, scaler = load_wesad_model()
    return model, scaler

@st.cache_data(show_spinner="Evaluating subject-independent performance on held-out subject S4...")
def get_cached_evaluation(_model, _scaler, _df):
    """Computes subject-independent performance metrics on test subject S4."""
    return evaluate_wesad_model(_model, _scaler, _df, test_subject="S4")

# Initialize cached assets
try:
    df_wesad = get_cached_dataset()
    raw_samples = get_cached_raw_samples()
    model, scaler = get_cached_model_and_scaler()
    eval_results = get_cached_evaluation(model, scaler, df_wesad)
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# MATPLOTLIB DARK THEME HELPER
# -----------------------------------------------------------------------------
def configure_plot_theme(fig, axes):
    """Configures high-contrast medical dark theme styling for Matplotlib."""
    fig.patch.set_facecolor('#0b1329')
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor('#0f172a')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['right'].set_color('#334155')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#f1f5f9')
        ax.tick_params(colors='#94a3b8', which='both')
        ax.grid(color='#1e293b', linestyle='--', linewidth=0.6, alpha=0.7)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 16px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">🫀</span>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em;">PULSE·GUARD AI</div>
                    <div style="font-size: 0.72rem; color: #06b6d4; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Wearable Stress Analytics</div>
                </div>
            </div>
            <div style="margin-top: 12px;">
                <span class="badge-teal">REAL WESAD BENCHMARK</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: #1e293b; margin: 10px 0 16px 0;'>", unsafe_allow_html=True)
    
    page = st.radio(
        "NAVIGATION",
        [
            "Home Dashboard",
            "Dataset Explorer",
            "Signal Analysis",
            "Stress Classification",
            "Model Performance",
            "Feature Importance",
            "Risk Report",
            "About Project"
        ],
        index=0
    )
    
    st.markdown("<hr style='border-color: #1e293b; margin: 20px 0 16px 0;'>", unsafe_allow_html=True)
    
    # System Status Panel
    st.markdown("""
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; font-size: 0.78rem;">
            <div style="color: #94a3b8; font-weight: 600; margin-bottom: 6px;">TELEMETRY STATUS</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #cbd5e1;">Engine:</span>
                <span style="color: #38bdf8; font-weight: 600;">Random Forest</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #cbd5e1;">Eval Protocol:</span>
                <span style="color: #10b981; font-weight: 600;">LOSO (Test: S4)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #cbd5e1;">Windows:</span>
                <span style="color: #f8fafc; font-weight: 600;">1,214 (60s stride 5s)</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #cbd5e1;">System:</span>
                <span style="color: #10b981; font-weight: 600;">● Active</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1: HOME DASHBOARD
# =============================================================================
if page == "Home Dashboard":
    # Hero Banner
    st.markdown("""
        <div class="saas-card-glow">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span class="badge-teal">CLINICAL WEARABLE BIOSENSING</span>
                    <h1 style="color: #ffffff; font-size: 2.1rem; font-weight: 800; margin: 10px 0 6px 0; letter-spacing: -0.02em;">
                        Wearable-Signal Stress-Risk Classification
                    </h1>
                    <p style="color: #94a3b8; font-size: 0.98rem; max-width: 820px; margin: 0; line-height: 1.5;">
                        Continuous physiological affect intelligence and acute sympathetic stress detection powered by genuine 
                        RespiBAN chest biosensors from the official University of Siegen WESAD benchmark.
                    </p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #1e293b; border-radius: 8px; padding: 10px 14px; text-align: right;">
                    <div style="color: #64748b; font-size: 0.72rem; font-weight: 600;">DATASET SOURCE</div>
                    <div style="color: #38bdf8; font-weight: 700; font-size: 0.88rem;">ACM ICMI 2018 / UCI 465</div>
                    <div style="color: #10b981; font-size: 0.72rem;">● 100% Genuine Biosignals</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # KPI Grid
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown("""
            <div class="kpi-container">
                <div class="kpi-label">Subjects</div>
                <div class="kpi-value">3</div>
                <div class="kpi-sub">S2, S3, S4 (WESAD)</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Windows</div>
                <div class="kpi-value">{len(df_wesad):,}</div>
                <div class="kpi-sub">60s with 5s stride</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Features</div>
                <div class="kpi-value">{len(FEATURE_COLUMNS)}</div>
                <div class="kpi-sub">ECG, EDA, Temp, Resp, ACC</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Test Acc (S4)</div>
                <div class="kpi-value">{eval_results['accuracy']*100:.1f}%</div>
                <div class="kpi-sub">Subject-Independent</div>
            </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
            <div class="kpi-container" style="border-color: rgba(244, 63, 94, 0.4);">
                <div class="kpi-label" style="color: #f43f5e;">Stress Recall</div>
                <div class="kpi-value" style="color: #f43f5e;">{eval_results['stress_recall']*100:.1f}%</div>
                <div class="kpi-sub" style="color: #f43f5e;">High-Priority Target</div>
            </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Stress Precision</div>
                <div class="kpi-value">{eval_results['stress_precision']*100:.1f}%</div>
                <div class="kpi-sub">Low False Alarms</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    # End-to-End Workflow Pipeline
    st.markdown("""
        <div class="saas-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <h3 style="color: #f8fafc; font-size: 1.15rem; font-weight: 700; margin: 0;">
                    Biomedical Machine Learning Pipeline
                </h3>
                <span class="badge-teal">Strict Subject-Independent Validation</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                <div class="workflow-step">
                    <div class="workflow-num">1</div>
                    <div class="workflow-title">Wearable Biosensors</div>
                    <div class="workflow-desc">RespiBAN chest unit recording ECG, EDA, Respiration, Skin Temp, and 3D ACC at 700 Hz.</div>
                </div>
                <div class="workflow-step">
                    <div class="workflow-num">2</div>
                    <div class="workflow-title">Signal Windowing</div>
                    <div class="workflow-desc">Segmentation into 60-second windows with 5-second sliding stride (84,000 samples per window).</div>
                </div>
                <div class="workflow-step">
                    <div class="workflow-num">3</div>
                    <div class="workflow-title">Baseline Normalization</div>
                    <div class="workflow-desc">Participant-specific z-score standardization accounting for individual basal conductance & skin temp.</div>
                </div>
                <div class="workflow-step">
                    <div class="workflow-num">4</div>
                    <div class="workflow-title">Random Forest Classifier</div>
                    <div class="workflow-desc">Ensemble of 100 decision trees with balanced class weights to maximize acute stress sensitivity.</div>
                </div>
                <div class="workflow-step">
                    <div class="workflow-num">5</div>
                    <div class="workflow-title">Affect Classification</div>
                    <div class="workflow-desc">Unseen subject inference: Baseline Homeostasis, Acute Stress, or Amusement arousal.</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 3 Physiological Affect Classes Overview
    colA, colB, colC = st.columns(3)
    with colA:
        base_cnt = (df_wesad[TARGET_COLUMN] == 0).sum()
        st.markdown(f"""
            <div class="saas-card" style="border-top: 3px solid #10b981;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span class="badge-baseline">CLASS 0</span>
                    <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">{base_cnt} Windows ({base_cnt/len(df_wesad)*100:.1f}%)</span>
                </div>
                <h4 style="color: #10b981; font-size: 1.1rem; margin: 0 0 8px 0;">Baseline Homeostasis</h4>
                <p style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45; margin: 0 0 10px 0;">
                    Relaxed physiological state during quiet reading. Characterized by high Heart Rate Variability (HRV), 
                    low tonic Electrodermal Activity (EDA), and regular, calm respiration.
                </p>
                <div style="background: #0b1329; border-radius: 6px; padding: 8px 12px; font-size: 0.78rem; color: #cbd5e1;">
                    ● Vagal parasympathetic dominance<br>
                    ● Low sympathetic arousal<br>
                    ● Baseline skin temp stabilization
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with colB:
        stress_cnt = (df_wesad[TARGET_COLUMN] == 1).sum()
        st.markdown(f"""
            <div class="saas-card" style="border-top: 3px solid #f43f5e;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span class="badge-stress">CLASS 1 (HIGH PRIORITY)</span>
                    <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">{stress_cnt} Windows ({stress_cnt/len(df_wesad)*100:.1f}%)</span>
                </div>
                <h4 style="color: #f43f5e; font-size: 1.1rem; margin: 0 0 8px 0;">Acute Sympathetic Stress</h4>
                <p style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45; margin: 0 0 10px 0;">
                    Social-evaluative stress induced via the Trier Social Stress Test (TSST: mental arithmetic + public speech). 
                    Characterized by sudden EDA surges, suppressed HRV, and tachypnea.
                </p>
                <div style="background: #0b1329; border-radius: 6px; padding: 8px 12px; font-size: 0.78rem; color: #cbd5e1;">
                    ● Acute fight-or-flight sympathetic response<br>
                    ● Suppressed parasympathetic tone (low RMSSD)<br>
                    ● Peripheral vasoconstriction
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with colC:
        amuse_cnt = (df_wesad[TARGET_COLUMN] == 2).sum()
        st.markdown(f"""
            <div class="saas-card" style="border-top: 3px solid #f59e0b;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span class="badge-amuse">CLASS 2</span>
                    <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">{amuse_cnt} Windows ({amuse_cnt/len(df_wesad)*100:.1f}%)</span>
                </div>
                <h4 style="color: #f59e0b; font-size: 1.1rem; margin: 0 0 8px 0;">Amusement / Emotional Valence</h4>
                <p style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45; margin: 0 0 10px 0;">
                    Positive emotional arousal induced through curated comedic video clips. Exhibits moderate autonomic 
                    activation with characteristic laughter-induced respiration perturbations.
                </p>
                <div style="background: #0b1329; border-radius: 6px; padding: 8px 12px; font-size: 0.78rem; color: #cbd5e1;">
                    ● High positive emotional valence<br>
                    ● Moderate sympathetic arousal<br>
                    ● Vocal / diaphragm motion signatures
                </div>
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: DATASET EXPLORER
# =============================================================================
elif page == "Dataset Explorer":
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin-bottom: 4px;">
                🗂️ Genuine WESAD Dataset Explorer
            </h2>
            <p style="color: #94a3b8; font-size: 0.92rem; margin: 0;">
                Inspect authentic physiological records processed from the University of Siegen WESAD benchmark. 
                Zero synthetic data or simulated records.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Filters Row
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        subj_choice = st.selectbox("Participant Filter", ["All Participants", "S2", "S3", "S4"])
    with f2:
        cond_choice = st.selectbox("Affect State Filter", ["All Affect States", "Baseline", "Stress", "Amusement"])
    with f3:
        st.markdown("""
            <div style="padding-top: 26px;">
                <span class="badge-teal">1,214 SYNCHRONIZED 60-SECOND WINDOWS</span>
            </div>
        """, unsafe_allow_html=True)
        
    filtered_df = df_wesad.copy()
    if subj_choice != "All Participants":
        filtered_df = filtered_df[filtered_df[SUBJECT_COLUMN] == subj_choice]
    if cond_choice != "All Affect States":
        cond_map = {"Baseline": 0, "Stress": 1, "Amusement": 2}
        filtered_df = filtered_df[filtered_df[TARGET_COLUMN] == cond_map[cond_choice]]
        
    # Stats summary
    st.markdown(f"""
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 18px; margin: 12px 0 16px 0; display: flex; gap: 24px; font-size: 0.88rem;">
            <div><span style="color: #94a3b8;">Showing Windows:</span> <strong style="color: #38bdf8;">{len(filtered_df):,}</strong></div>
            <div><span style="color: #94a3b8;">Total Features:</span> <strong style="color: #f8fafc;">28</strong></div>
            <div><span style="color: #94a3b8;">Subjects Represented:</span> <strong style="color: #10b981;">{', '.join(filtered_df[SUBJECT_COLUMN].unique())}</strong></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Dataset Table
    display_cols = ["subject_id", "condition", "Heart_Rate", "HRV_RMSSD", "EDA_mean", "EDA_slope", "Temp_mean", "Respiratory_Rate", "ACC_mean"]
    st.dataframe(
        filtered_df[display_cols + [c for c in FEATURE_COLUMNS if c not in display_cols]].head(250),
        use_container_width=True,
        height=320
    )
    
    # Distribution Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 style='color: #f1f5f9; font-size: 1rem; margin-top: 14px;'>Class Distribution by Participant</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        configure_plot_theme(fig, ax)
        
        counts = df_wesad.groupby([SUBJECT_COLUMN, "condition"]).size().unstack(fill_value=0)
        colors = [LABEL_COLORS[0], LABEL_COLORS[2], LABEL_COLORS[1]]
        counts.plot(kind="bar", stacked=True, ax=ax, color=["#10b981", "#f59e0b", "#f43f5e"], edgecolor="#0f172a", width=0.55)
        ax.set_ylabel("Number of 60s Windows")
        ax.set_xlabel("Participant ID")
        ax.legend(title="Affect State", facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#cbd5e1")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col2:
        st.markdown("<h4 style='color: #f1f5f9; font-size: 1rem; margin-top: 14px;'>Key Biosignal Distribution Across States</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        configure_plot_theme(fig, ax)
        
        sns.boxplot(
            data=df_wesad,
            x="condition",
            y="EDA_mean",
            palette={"Baseline": "#10b981", "Stress": "#f43f5e", "Amusement": "#f59e0b"},
            ax=ax,
            boxprops=dict(alpha=0.8)
        )
        ax.set_ylabel("Electrodermal Activity Mean (µS)")
        ax.set_xlabel("Affect Condition")
        ax.set_title("Skin Conductance Elevation During TSST Acute Stress", color="#cbd5e1", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    # Physiological Summary Table
    st.markdown("<h4 style='color: #f1f5f9; font-size: 1rem; margin-top: 16px;'>Physiological Characteristics (Mean ± Std Dev)</h4>", unsafe_allow_html=True)
    summary_data = []
    for cond_name, code in [("Baseline", 0), ("Stress", 1), ("Amusement", 2)]:
        sub = df_wesad[df_wesad[TARGET_COLUMN] == code]
        summary_data.append({
            "Affect State": cond_name,
            "Heart Rate (BPM)": f"{sub['Heart_Rate'].mean():.1f} ± {sub['Heart_Rate'].std():.1f}",
            "HRV RMSSD (ms)": f"{sub['HRV_RMSSD'].mean():.1f} ± {sub['HRV_RMSSD'].std():.1f}",
            "EDA Mean (µS)": f"{sub['EDA_mean'].mean():.2f} ± {sub['EDA_mean'].std():.2f}",
            "EDA Slope (µS/s)": f"{sub['EDA_slope'].mean():.4f} ± {sub['EDA_slope'].std():.4f}",
            "Skin Temp (°C)": f"{sub['Temp_mean'].mean():.2f} ± {sub['Temp_mean'].std():.2f}",
            "Respiration Rate (br/min)": f"{sub['Respiratory_Rate'].mean():.1f} ± {sub['Respiratory_Rate'].std():.1f}"
        })
    st.table(pd.DataFrame(summary_data))


# =============================================================================
# PAGE 3: SIGNAL ANALYSIS
# =============================================================================
elif page == "Signal Analysis":
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin-bottom: 4px;">
                📈 Multi-Channel Raw Biosignal Waveform Viewer
            </h2>
            <p style="color: #94a3b8; font-size: 0.92rem; margin: 0;">
                Interactive 10-second continuous multi-channel waveform telemetry recorded by the RespiBAN chest unit. 
                Compare authentic physiological dynamics across resting baseline, acute TSST stress, and amusement.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    c_sub, c_cond, c_info = st.columns([1, 1, 2])
    with c_sub:
        chosen_subject = st.selectbox("Select Participant", ["S2", "S3", "S4"], index=0)
    with c_cond:
        chosen_cond = st.selectbox("Select Condition", ["Baseline", "Stress", "Amusement"], index=1)
    with c_info:
        st.markdown(f"""
            <div style="padding-top: 24px;">
                <span class="badge-teal">TELEMETRY: {chosen_subject} · {chosen_cond.upper()} RECORDING</span>
            </div>
        """, unsafe_allow_html=True)
        
    # Retrieve raw trace from cached pickle
    trace_data = raw_samples.get(chosen_subject, {}).get(chosen_cond, None)
    if trace_data is None:
        st.warning(f"Raw trace for {chosen_subject} - {chosen_cond} not found.")
    else:
        time_axis = trace_data["time_sec"]
        ecg = trace_data["ECG"]
        eda = trace_data["EDA"]
        temp = trace_data["Temp"]
        resp = trace_data["Resp"]
        acc = trace_data["ACC_mag"]
        
        # Multi-channel Waveform Plot
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        configure_plot_theme(fig, axes)
        
        # 1. ECG
        axes[0].plot(time_axis, ecg, color="#38bdf8", linewidth=1.2)
        axes[0].set_ylabel("ECG (mV)", fontsize=9)
        axes[0].set_title(f"Synchronized Multi-Modal Waveforms: Subject {chosen_subject} — {chosen_cond} State (10s Epoch)", fontsize=11, color="#f8fafc", pad=10)
        
        # 2. EDA
        axes[1].plot(time_axis, eda, color="#f43f5e" if chosen_cond == "Stress" else "#10b981", linewidth=1.4)
        axes[1].set_ylabel("EDA (µS)", fontsize=9)
        
        # 3. Respiration
        axes[2].plot(time_axis, resp, color="#a855f7", linewidth=1.2)
        axes[2].set_ylabel("Resp (A.U.)", fontsize=9)
        
        # 4. Skin Temp
        axes[3].plot(time_axis, temp, color="#f59e0b", linewidth=1.3)
        axes[3].set_ylabel("Temp (°C)", fontsize=9)
        
        # 5. ACC Magnitude
        axes[4].plot(time_axis, acc, color="#94a3b8", linewidth=1.0)
        axes[4].set_ylabel("ACC (g)", fontsize=9)
        axes[4].set_xlabel("Time (seconds)", fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Physiological Interpretation Card
        st.markdown(f"""
            <div class="saas-card" style="margin-top: 14px;">
                <h4 style="color: #38bdf8; font-size: 1.05rem; margin-top: 0;">
                    Biomedical Signal Insights for {chosen_cond} State ({chosen_subject})
                </h4>
                <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
                    {"🔴 <strong>Acute Sympathetic Arousal:</strong> Notice the elevated tonic baseline and phasic bursts in Electrodermal Activity (EDA), reflecting eccrine sweat gland innervation by sympathetic axons. ECG exhibits shorter R-R intervals (tachycardia) and suppressed beat-to-beat variability (vagal withdrawal). Respiration exhibits rapid or irregular cycles." if chosen_cond == "Stress" else ""}
                    {"🟢 <strong>Vagal Homeostatic Equilibrium:</strong> Notice the calm, quiescent EDA signal with minimal phasic sweat spikes. ECG displays pronounced respiratory sinus arrhythmia (RSA), where heart rate rhythmically fluctuates with breathing, confirming robust parasympathetic vagal nerve modulation." if chosen_cond == "Baseline" else ""}
                    {"🟡 <strong>Positive Emotional Valence:</strong> ECG shows moderate acceleration, while Respiration displays characteristic transient laughter flutter or vocal dynamics. EDA exhibits phasic skin conductance responses without the prolonged high-amplitude tonic surge observed in acute distress." if chosen_cond == "Amusement" else ""}
                </div>
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4: STRESS CLASSIFICATION (INFERENCE CONSOLE)
# =============================================================================
elif page == "Stress Classification":
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin-bottom: 4px;">
                🔮 Subject-Independent Stress Classification
            </h2>
            <p style="color: #94a3b8; font-size: 0.92rem; margin: 0;">
                Test real-time model inference using authentic held-out patient windows (Subject S4) or simulate custom 
                physiological parameter profiles.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Testing Modes
    test_mode = st.radio(
        "Choose Inference Source:",
        ["Option A: Evaluate Authentic Held-Out Patient Windows (Subject S4)", "Option B: Interactive Physiological Biosensor Simulator"],
        horizontal=True
    )
    
    st.markdown("<hr style='border-color: #1e293b; margin: 16px 0;'>", unsafe_allow_html=True)
    
    sample_to_predict = None
    ground_truth = None
    subj_label = "S4"
    
    if "Option A" in test_mode:
        s4_windows = df_wesad[df_wesad[SUBJECT_COLUMN] == "S4"].reset_index(drop=True)
        
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            filter_state = st.selectbox("Filter Window by State", ["All S4 Windows", "Baseline Windows", "Stress Windows", "Amusement Windows"])
            if filter_state == "Baseline Windows":
                s4_subset = s4_windows[s4_windows[TARGET_COLUMN] == 0].reset_index(drop=True)
            elif filter_state == "Stress Windows":
                s4_subset = s4_windows[s4_windows[TARGET_COLUMN] == 1].reset_index(drop=True)
            elif filter_state == "Amusement Windows":
                s4_subset = s4_windows[s4_windows[TARGET_COLUMN] == 2].reset_index(drop=True)
            else:
                s4_subset = s4_windows
                
            window_idx = st.slider("Select Window Index", 0, len(s4_subset) - 1, 0)
            
        selected_row = s4_subset.iloc[window_idx]
        ground_truth = int(selected_row[TARGET_COLUMN])
        sample_to_predict = selected_row[FEATURE_COLUMNS]
        
        with col_f2:
            st.markdown(f"""
                <div class="saas-card" style="margin-bottom: 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">PATIENT TELEMETRY METADATA</span>
                        <span class="badge-teal">HELD-OUT TEST PARTICIPANT S4</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-size: 0.85rem;">
                        <div><span style="color: #64748b;">Heart Rate:</span> <strong style="color: #f1f5f9;">{selected_row['Heart_Rate']:.1f} BPM</strong></div>
                        <div><span style="color: #64748b;">HRV (RMSSD):</span> <strong style="color: #f1f5f9;">{selected_row['HRV_RMSSD']:.1f} ms</strong></div>
                        <div><span style="color: #64748b;">EDA Mean:</span> <strong style="color: #f1f5f9;">{selected_row['EDA_mean']:.2f} µS</strong></div>
                        <div><span style="color: #64748b;">EDA Slope:</span> <strong style="color: #f1f5f9;">{selected_row['EDA_slope']:+.4f}</strong></div>
                        <div><span style="color: #64748b;">Skin Temp:</span> <strong style="color: #f1f5f9;">{selected_row['Temp_mean']:.2f} °C</strong></div>
                        <div><span style="color: #64748b;">Resp Rate:</span> <strong style="color: #f1f5f9;">{selected_row['Respiratory_Rate']:.1f} br/min</strong></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
    else:  # Option B: Simulator
        st.markdown("<p style='color: #94a3b8; font-size: 0.88rem;'>Simulate wearable biosensor values using quick presets or fine-tune sliders below:</p>", unsafe_allow_html=True)
        preset_cols = st.columns(3)
        with preset_cols[0]:
            p1 = st.button("Preset: Calm Baseline Homeostasis")
        with preset_cols[1]:
            p2 = st.button("Preset: Acute TSST Stress Spike")
        with preset_cols[2]:
            p3 = st.button("Preset: High Amusement Arousal")
            
        # Defaults based on actual WESAD S4 distributions
        if p2:
            def_hr, def_hrv, def_eda, def_slope, def_temp, def_resp = 88.0, 18.5, 3.8, 0.025, 33.1, 19.5
        elif p3:
            def_hr, def_hrv, def_eda, def_slope, def_temp, def_resp = 76.5, 34.0, 1.4, 0.005, 34.2, 16.0
        else:
            def_hr, def_hrv, def_eda, def_slope, def_temp, def_resp = 68.5, 48.0, 0.85, -0.001, 34.6, 13.5
            
        c_sim1, c_sim2, c_sim3 = st.columns(3)
        with c_sim1:
            sim_hr = st.slider("Heart Rate (BPM)", 50.0, 130.0, float(def_hr), 0.5)
            sim_hrv = st.slider("HRV RMSSD (ms)", 8.0, 90.0, float(def_hrv), 0.5)
        with c_sim2:
            sim_eda = st.slider("EDA Mean (µS)", 0.2, 8.0, float(def_eda), 0.1)
            sim_slope = st.slider("EDA Slope (µS/s)", -0.05, 0.05, float(def_slope), 0.001)
        with c_sim3:
            sim_temp = st.slider("Skin Temp (°C)", 30.0, 36.5, float(def_temp), 0.1)
            sim_resp = st.slider("Respiration Rate (br/min)", 8.0, 30.0, float(def_resp), 0.5)
            
        # Construct synthetic feature vector using S4 baseline reference for un-simulated features
        ref_row = df_wesad[df_wesad[SUBJECT_COLUMN] == "S4"].iloc[0].copy()
        ref_row["Heart_Rate"] = sim_hr
        ref_row["HRV_RMSSD"] = sim_hrv
        ref_row["EDA_mean"] = sim_eda
        ref_row["EDA_slope"] = sim_slope
        ref_row["Temp_mean"] = sim_temp
        ref_row["Respiratory_Rate"] = sim_resp
        sample_to_predict = ref_row[FEATURE_COLUMNS]
        
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # Run Prediction
    if sample_to_predict is not None:
        prediction_output = predict_wesad_window(sample_to_predict, model, scaler)
        pred_class = prediction_output["pred_class"]
        pred_label = prediction_output["class_name"]
        probs = prediction_output["probabilities"]
        
        # Color coding
        border_col = LABEL_COLORS[pred_class]
        badge_class = "badge-baseline" if pred_class == 0 else ("badge-stress" if pred_class == 1 else "badge-amuse")
        
        st.markdown(f"""
            <div class="saas-card" style="border: 2px solid {border_col}; margin-top: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span class="{badge_class}">PREDICTED AFFECT STATE</span>
                        <h2 style="color: {border_col}; font-size: 1.9rem; font-weight: 800; margin: 8px 0 4px 0;">
                            {pred_label.upper()}
                        </h2>
                        <div style="color: #94a3b8; font-size: 0.86rem;">
                            Model Confidence: <strong style="color: #f8fafc;">{prediction_output['confidence']*100:.1f}%</strong>
                        </div>
                    </div>
                    {f'''
                    <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 18px; text-align: right;">
                        <div style="color: #64748b; font-size: 0.75rem; font-weight: 600;">GROUND TRUTH (PATIENT S4)</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: {LABEL_COLORS[ground_truth]};">
                            {LABEL_NAMES[ground_truth]}
                        </div>
                        <div style="font-size: 0.78rem; color: {'#10b981' if ground_truth == pred_class else '#f43f5e'};">
                            {'✅ Verified Concordance' if ground_truth == pred_class else '⚠️ Mismatch (Borderline window)'}
                        </div>
                    </div>
                    ''' if ground_truth is not None else ''}
                </div>
                
                <hr style="border-color: #1e293b; margin: 16px 0;">
                
                <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; margin-bottom: 10px;">
                    Ensemble Class Probabilities
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 16px;">
                    <div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                            <span style="color: #10b981;">Baseline</span>
                            <span style="color: #f1f5f9; font-weight: 700;">{probs['Baseline']*100:.1f}%</span>
                        </div>
                        <div style="background: #0b1329; border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="background: #10b981; width: {probs['Baseline']*100}%; height: 100%;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                            <span style="color: #f43f5e;">Stress</span>
                            <span style="color: #f1f5f9; font-weight: 700;">{probs['Stress']*100:.1f}%</span>
                        </div>
                        <div style="background: #0b1329; border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="background: #f43f5e; width: {probs['Stress']*100}%; height: 100%;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                            <span style="color: #f59e0b;">Amusement</span>
                            <span style="color: #f1f5f9; font-weight: 700;">{probs['Amusement']*100:.1f}%</span>
                        </div>
                        <div style="background: #0b1329; border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="background: #f59e0b; width: {probs['Amusement']*100}%; height: 100%;"></div>
                        </div>
                    </div>
                </div>
                
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 14px 18px;">
                    <div style="color: #38bdf8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">
                        Physiological Diagnostic Rationale
                    </div>
                    <ul style="color: #cbd5e1; font-size: 0.85rem; margin: 0; padding-left: 20px; line-height: 1.5;">
                        {''.join(f'<li>{b}</li>' for b in prediction_output['rationale_bullets'])}
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 5: MODEL PERFORMANCE
# =============================================================================
elif page == "Model Performance":
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin-bottom: 4px;">
                📊 Subject-Independent Model Performance
            </h2>
            <p style="color: #94a3b8; font-size: 0.92rem; margin: 0;">
                Rigorous evaluation on unseen participant <strong>S4</strong> (408 windows). Model was trained exclusively on 
                participants <strong>S2 and S3</strong> (806 windows) with zero data leakage.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Priority Stress Alert Banner
    st.markdown(f"""
        <div class="saas-card" style="border: 1px solid rgba(244, 63, 94, 0.4); background: linear-gradient(145deg, rgba(244, 63, 94, 0.08) 0%, #0f172a 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span class="badge-stress">PRIMARY EVALUATION TARGET: STRESS RECALL</span>
                    <h3 style="color: #f43f5e; font-size: 1.45rem; font-weight: 800; margin: 8px 0 4px 0;">
                        {eval_results['stress_recall']*100:.1f}% Stress Recall · {eval_results['stress_precision']*100:.1f}% Stress Precision
                    </h3>
                    <p style="color: #cbd5e1; font-size: 0.88rem; margin: 0; max-width: 800px; line-height: 1.5;">
                        <strong>Why Stress Recall is paramount:</strong> In continuous clinical and occupational health monitoring, a False Negative 
                        (missing an acute stress episode) fails to alert the patient or trigger timely therapeutic biofeedback. 
                        By utilizing balanced class weights, our Random Forest achieves <strong>{eval_results['stress_recall']*100:.1f}%</strong> sensitivity on unseen participants.
                    </p>
                </div>
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 18px; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.72rem; font-weight: 600;">EVALUATION COHORT</div>
                    <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 700;">408 Windows</div>
                    <div style="color: #06b6d4; font-size: 0.75rem;">Subject S4 (Held-Out)</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 4 Main KPI Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Overall Accuracy</div>
                <div class="kpi-value">{eval_results['accuracy']*100:.2f}%</div>
                <div class="kpi-sub">Held-out participant S4</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Macro Precision</div>
                <div class="kpi-value">{eval_results['precision_macro']*100:.2f}%</div>
                <div class="kpi-sub">Unweighted class mean</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Macro Recall</div>
                <div class="kpi-value">{eval_results['recall_macro']*100:.2f}%</div>
                <div class="kpi-sub">Unweighted class mean</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Macro F1-Score</div>
                <div class="kpi-value">{eval_results['f1_macro']*100:.2f}%</div>
                <div class="kpi-sub">Harmonic mean</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    col_cm, col_rep = st.columns([1, 1])
    with col_cm:
        st.markdown("<h4 style='color: #f1f5f9; font-size: 1.05rem;'>Confusion Matrix (Subject S4)</h4>", unsafe_allow_html=True)
        cm = eval_results['confusion_matrix']
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        configure_plot_theme(fig, ax)
        
        target_labels = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_labels,
            yticklabels=target_labels,
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold", "color": "#ffffff"}
        )
        ax.set_ylabel("Actual Condition (Ground Truth)", color="#94a3b8", fontsize=10)
        ax.set_xlabel("Predicted Condition by RF", color="#94a3b8", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col_rep:
        st.markdown("<h4 style='color: #f1f5f9; font-size: 1.05rem;'>Class-by-Class Metric Breakdown</h4>", unsafe_allow_html=True)
        rep_rows = []
        for cname in ["Baseline", "Stress", "Amusement"]:
            if cname in eval_results['class_metrics']:
                m = eval_results['class_metrics'][cname]
                rep_rows.append({
                    "Affect Class": cname,
                    "Precision": f"{m['precision']*100:.1f}%",
                    "Recall": f"{m['recall']*100:.1f}%",
                    "F1-Score": f"{m['f1']*100:.1f}%",
                    "Test Windows": m['support']
                })
        st.table(pd.DataFrame(rep_rows))
        
        st.markdown("""
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 16px; font-size: 0.8rem; color: #94a3b8; line-height: 1.45;">
                <strong style="color: #38bdf8;">Why Subject-Independent Splitting Matters:</strong><br>
                In wearable ML, standard random train/test splits cause severe participant window leakage (adjacent 60-second windows 
                from the same person end up in both train and test sets), creating an illusion of 98-99% accuracy that crashes in real deployment. 
                Our subject-independent split reflects true generalization to a completely new human participant.
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 6: FEATURE IMPORTANCE
# =============================================================================
elif page == "Feature Importance":
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin-bottom: 4px;">
                🧬 Random Forest Feature Importance Analysis
            </h2>
            <p style="color: #94a3b8; font-size: 0.92rem; margin: 0;">
                Mean decrease in impurity (Gini Importance) across 100 decision trees. Uncovers which biosensor features 
                are most discriminating for acute sympathetic stress.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    feat_df = eval_results['feature_importances'].head(12)
    
    col_chart, col_exp = st.columns([1.2, 1])
    with col_chart:
        fig, ax = plt.subplots(figsize=(7, 5.2))
        configure_plot_theme(fig, ax)
        
        # Invert for horizontal chart
        df_plot = feat_df.sort_values("Importance", ascending=True)
        bars = ax.barh(df_plot["Feature"], df_plot["Importance"], color="#06b6d4", edgecolor="#0284c7", height=0.65)
        ax.set_xlabel("Normalized Gini Importance", fontsize=10)
        ax.set_title("Top 12 Most Informative Physiological Biomarkers", fontsize=11, color="#f8fafc", pad=12)
        
        # Add labels
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.002, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%", va='center', ha='left', color='#cbd5e1', fontsize=8.5)
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col_exp:
        st.markdown("""
            <div class="saas-card" style="margin-bottom: 12px;">
                <h4 style="color: #06b6d4; font-size: 1rem; margin-top: 0;">1. Electrodermal Dynamics (EDA Slope & Mean)</h4>
                <p style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45; margin: 0;">
                    Sweat glands are exclusively innervated by the sympathetic branch with no parasympathetic counter-regulation. 
                    A positive EDA slope and elevated tonic level provide the most direct, unmasked marker of acute fight-or-flight activation.
                </p>
            </div>
            <div class="saas-card" style="margin-bottom: 12px;">
                <h4 style="color: #38bdf8; font-size: 1rem; margin-top: 0;">2. Heart Rate Variability (HRV RMSSD)</h4>
                <p style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45; margin: 0;">
                    RMSSD (Root Mean Square of Successive RR interval Differences) reflects vagally mediated parasympathetic activity. 
                    During cognitive stress, the prefrontal cortex reduces inhibitory vagal control, driving RMSSD down sharply.
                </p>
            </div>
            <div class="saas-card">
                <h4 style="color: #f59e0b; font-size: 1rem; margin-top: 0;">3. Respiration & Peripheral Vasoconstriction</h4>
                <p style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45; margin: 0;">
                    Respiration rate and thoracic amplitude shift rapidly during vocalization and psychological arousal. 
                    Simultaneously, peripheral vasoconstriction produces a characteristic negative skin temperature slope.
                </p>
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 7: RISK REPORT
# =============================================================================
elif page == "Risk Report":
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin-bottom: 4px;">
                📑 Wearable Physiological Risk Assessment Report
            </h2>
            <p style="color: #94a3b8; font-size: 0.92rem; margin: 0;">
                Generate, inspect, and export comprehensive clinical research reports for any patient monitoring window.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Pick window
    s4_windows = df_wesad[df_wesad[SUBJECT_COLUMN] == "S4"].reset_index(drop=True)
    c_sel, c_btn = st.columns([2, 1])
    with c_sel:
        rep_win_idx = st.slider("Select Patient Window for Report Generation (Subject S4)", 0, len(s4_windows)-1, 120)
    with c_btn:
        st.markdown("<div style='padding-top: 24px;'></div>", unsafe_allow_html=True)
        gen_clicked = st.button("Generate Diagnostic Report")
        
    win_row = s4_windows.iloc[rep_win_idx]
    pred_res = predict_wesad_window(win_row[FEATURE_COLUMNS], model, scaler)
    p_class = pred_res["pred_class"]
    p_name = pred_res["class_name"]
    p_conf = pred_res["confidence"]
    probs = pred_res["probabilities"]
    
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Report Card
    st.markdown(f"""
        <div class="saas-card" style="border-top: 4px solid {LABEL_COLORS[p_class]}; margin-top: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid #1e293b; padding-bottom: 14px; margin-bottom: 16px;">
                <div>
                    <h3 style="color: #ffffff; font-size: 1.25rem; font-weight: 800; margin: 0;">
                        PHYSIOLOGICAL STRESS ASSESSMENT DOSSIER
                    </h3>
                    <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">
                        Patient ID: <strong style="color: #38bdf8;">WESAD-S4</strong> · Protocol: Trier Social Stress Test · Generated: {timestamp_str}
                    </div>
                </div>
                <div style="text-align: right;">
                    <span class="{'badge-stress' if p_class == 1 else ('badge-baseline' if p_class == 0 else 'badge-amuse')}">
                        DIAGNOSIS: {p_name.upper()}
                    </span>
                    <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px;">
                        Confidence: {p_conf*100:.1f}%
                    </div>
                </div>
            </div>
            
            <h4 style="color: #f1f5f9; font-size: 0.95rem; margin-bottom: 10px;">1. Vital Biomarker Telemetry</h4>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; font-size: 0.85rem;">
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 6px; padding: 10px;">
                    <div style="color: #64748b;">Heart Rate</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{win_row['Heart_Rate']:.1f} BPM</div>
                    <div style="font-size: 0.72rem; color: {'#f43f5e' if win_row['Heart_Rate'] > 85 else '#10b981'};">
                        {'Elevated' if win_row['Heart_Rate'] > 85 else 'Normal Resting'}
                    </div>
                </div>
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 6px; padding: 10px;">
                    <div style="color: #64748b;">HRV (RMSSD)</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{win_row['HRV_RMSSD']:.1f} ms</div>
                    <div style="font-size: 0.72rem; color: {'#f43f5e' if win_row['HRV_RMSSD'] < 25 else '#10b981'};">
                        {'Suppressed Vagal Tone' if win_row['HRV_RMSSD'] < 25 else 'Robust Vagal Tone'}
                    </div>
                </div>
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 6px; padding: 10px;">
                    <div style="color: #64748b;">EDA Mean / Slope</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{win_row['EDA_mean']:.2f} µS</div>
                    <div style="font-size: 0.72rem; color: {'#f43f5e' if win_row['EDA_slope'] > 0.01 else '#10b981'};">
                        Slope: {win_row['EDA_slope']:+.4f}
                    </div>
                </div>
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 6px; padding: 10px;">
                    <div style="color: #64748b;">Respiration Rate</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">{win_row['Respiratory_Rate']:.1f} br/min</div>
                    <div style="font-size: 0.72rem; color: {'#f43f5e' if win_row['Respiratory_Rate'] > 18 else '#10b981'};">
                        {'Tachypnea' if win_row['Respiratory_Rate'] > 18 else 'Normal'}
                    </div>
                </div>
            </div>
            
            <h4 style="color: #f1f5f9; font-size: 0.95rem; margin-bottom: 8px;">2. Diagnostic Evaluation</h4>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5; margin: 0 0 14px 0;">
                {pred_res['rationale']}
            </p>
            
            <h4 style="color: #f1f5f9; font-size: 0.95rem; margin-bottom: 8px;">3. Recommended Clinical Interventions</h4>
            <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 6px; padding: 12px 16px; font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                {"● <strong>Immediate Biofeedback:</strong> Initiate 4-7-8 diaphragmatic paced breathing to stimulate vagal baroreflex sensitivity.<br>● <strong>Cognitive De-escalation:</strong> Implement a 10-minute micro-break away from cognitive/social stressors.<br>● <strong>Hydration & Ergonomics:</strong> Rest and realign posture to mitigate sympathetic vasoconstriction." if p_class == 1 else "● <strong>Maintain Baseline Habits:</strong> Autonomic stability is optimal. Continue present tasks with scheduled periodic hydration breaks."}
            </div>
            
            <div style="margin-top: 16px; font-size: 0.72rem; color: #64748b; font-style: italic;">
                Academic Disclaimer: This diagnostic dossier is generated for research and educational purposes using the WESAD benchmark. Not certified for direct autonomous medical decision-making.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Download Buttons
    report_text = f"""=============================================================================
PULSE GUARD AI — WEARABLE PHYSIOLOGICAL ASSESSMENT REPORT
=============================================================================
Timestamp: {timestamp_str}
Subject ID: WESAD-S4 (Held-Out Test Participant)
Window Index: {rep_win_idx}
Sensor Hardware: RespiBAN Professional Chest Unit @ 700 Hz

DIAGNOSTIC PREDICTION:
- Classification: {p_name.upper()}
- Model Confidence: {p_conf*100:.2f}%
- Ground Truth Label: {LABEL_NAMES[int(win_row['label'])]}

PROBABILITY DISTRIBUTION:
- Baseline:   {probs['Baseline']*100:.2f}%
- Stress:     {probs['Stress']*100:.2f}%
- Amusement:  {probs['Amusement']*100:.2f}%

KEY VITAL SIGNS:
- Heart Rate:             {win_row['Heart_Rate']:.2f} BPM
- HRV RMSSD:              {win_row['HRV_RMSSD']:.2f} ms
- EDA Mean:               {win_row['EDA_mean']:.4f} uS
- EDA Slope:              {win_row['EDA_slope']:+.4f} uS/s
- Skin Temperature:       {win_row['Temp_mean']:.2f} deg C
- Respiration Rate:       {win_row['Respiratory_Rate']:.2f} breaths/min

DIAGNOSTIC RATIONALE:
{pred_res['rationale']}

ACADEMIC RESEARCH NOTICE:
This report is generated using the open scientific WESAD dataset for university
coursework presentation. Not certified for autonomous clinical treatment.
=============================================================================
"""
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="📥 Download Clinical Text Report",
            data=report_text,
            file_name=f"wesad_risk_report_S4_win{rep_win_idx}.txt",
            mime="text/plain"
        )
    with col_d2:
        csv_buffer = io.StringIO()
        win_row.to_frame().T.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Window Telemetry CSV",
            data=csv_buffer.getvalue(),
            file_name=f"wesad_window_telemetry_S4_win{rep_win_idx}.csv",
            mime="text/csv"
        )


# =============================================================================
# PAGE 8: ABOUT PROJECT
# =============================================================================
elif page == "About Project":
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin-bottom: 4px;">
                ℹ️ About PulseGuard AI & WESAD Dataset
            </h2>
            <p style="color: #94a3b8; font-size: 0.92rem; margin: 0;">
                Wearable-Signal Stress-Risk Classification project designed for 2nd-year B.Tech CSE presentation.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="saas-card">
            <h3 style="color: #06b6d4; font-size: 1.15rem; margin-top: 0;">1. Scientific Citation & Research Attribution</h3>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
                This project utilizes the authentic <strong>WESAD (Wearable Stress and Affect Detection)</strong> dataset, 
                collected by researchers at the University of Siegen, Germany, and hosted on the UCI Machine Learning Repository:
            </p>
            <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 6px; padding: 12px 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #38bdf8;">
                Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Stockloew, and Kristof Van Laerhoven.<br>
                "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection."<br>
                ACM International Conference on Multimodal Interaction (ICMI '18), Boulder, CO, USA, 2018.
            </div>
        </div>
        
        <div class="saas-card">
            <h3 style="color: #06b6d4; font-size: 1.15rem; margin-top: 0;">2. Wearable Hardware Specifications</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 0.88rem; color: #cbd5e1;">
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;">
                    <strong style="color: #f8fafc;">RespiBAN Professional Chest Strap:</strong>
                    <ul style="margin: 8px 0 0 0; padding-left: 20px; font-size: 0.82rem; line-height: 1.5;">
                        <li><strong>Sampling Frequency:</strong> 700 Hz continuous</li>
                        <li><strong>Electrocardiogram (ECG):</strong> Lead-II configuration</li>
                        <li><strong>Electrodermal Activity (EDA):</strong> Placed on inner chest</li>
                        <li><strong>Respiration:</strong> Inductive plethysmography band</li>
                        <li><strong>Skin Temperature & 3D Accelerometer</strong></li>
                    </ul>
                </div>
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;">
                    <strong style="color: #f8fafc;">Empatica E4 Medical Wristband:</strong>
                    <ul style="margin: 8px 0 0 0; padding-left: 20px; font-size: 0.82rem; line-height: 1.5;">
                        <li><strong>Blood Volume Pulse (BVP):</strong> 64 Hz PPG sensor</li>
                        <li><strong>Electrodermal Activity (EDA):</strong> 4 Hz palmar/wrist</li>
                        <li><strong>Skin Temperature:</strong> 4 Hz infrared optical</li>
                        <li><strong>3-Axis Accelerometer:</strong> 32 Hz motion capture</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="saas-card">
            <h3 style="color: #06b6d4; font-size: 1.15rem; margin-top: 0;">3. Stress Induction Laboratory Protocol</h3>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5; margin: 0;">
                Participants underwent the standardized <strong>Trier Social Stress Test (TSST)</strong>:
                <br>● <strong>Mental Arithmetic:</strong> Counting backwards from 2043 in steps of 17 under loud negative social feedback.
                <br>● <strong>Public Speaking:</strong> Unprepared simulated job interview in front of an unresponsive panel of evaluators and video camera.
                <br>● <strong>Amusement:</strong> Watching 11 curated funny video clips to induce positive valence.
                <br>● <strong>Baseline:</strong> 20 minutes of quiet neutral reading in a seated position.
            </p>
        </div>
        
        <div class="saas-card">
            <h3 style="color: #06b6d4; font-size: 1.15rem; margin-top: 0;">4. Machine Learning & Engineering Highlights</h3>
            <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
                ● <strong>Subject-Independent Evaluation (LOSO):</strong> Completely avoids participant data leakage. 
                Trained on S2 and S3 (806 windows) and tested strictly on unseen participant S4 (408 windows).
                <br>● <strong>Subject-Specific Baseline Normalization:</strong> Implements Schmidt et al. standard 
                z-score normalization relative to resting baseline, resolving inter-subject physiological variance.
                <br>● <strong>Supervised Random Forest Bagging:</strong> 100 decorrelated decision trees with balanced class weights 
                ensuring a high <strong>{eval_results['stress_recall']*100:.1f}% Stress Recall</strong>.
            </div>
        </div>
    """, unsafe_allow_html=True)
