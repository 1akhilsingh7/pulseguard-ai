"""
=============================================================================
PULSEGUARD AI — WEARABLE STRESS ANALYTICS
Tagline: Understand Your Body. Monitor Your Stress.
Platform: Streamlit Healthcare Technology Web Product
=============================================================================
"""

import os
import sys
import io
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st

# Ensure local imports resolve seamlessly
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
    predict_wesad_window,
    evaluate_wesad_model
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PulseGuard AI | Understand Your Body. Monitor Your Stress.",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "HOME"

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

# -----------------------------------------------------------------------------
# PREMIUM HEALTHCARE SAAS CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .stApp {
        background-color: #070d18;
        color: #f1f5f9;
    }

    header[data-testid="stHeader"] {
        background-color: rgba(7, 13, 24, 0.85);
        backdrop-filter: blur(10px);
    }

    section[data-testid="stSidebar"] {
        background-color: #0b1329;
        border-right: 1px solid #1e293b;
    }

    /* Product Cards */
    .saas-card {
        background: linear-gradient(145deg, #0f172a 0%, #111c38 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .saas-card:hover {
        border-color: #0284c7;
    }

    .saas-hero-card {
        background: radial-gradient(circle at 85% 20%, rgba(6, 182, 212, 0.15) 0%, #0f172a 65%);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 40px 32px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
    }

    /* Hero Button */
    .hero-btn {
        display: inline-block;
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 14px 32px;
        border-radius: 8px;
        text-decoration: none;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.35);
        transition: all 0.2s ease;
        border: none;
        cursor: pointer;
    }
    .hero-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(6, 182, 212, 0.55);
    }

    /* Feature Pillars */
    .pillar-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 22px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .pillar-icon {
        font-size: 1.8rem;
        margin-bottom: 12px;
    }
    .pillar-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .pillar-desc {
        color: #94a3b8;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* Badges */
    .badge-baseline {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.16);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-stress {
        display: inline-block;
        background-color: rgba(244, 63, 94, 0.18);
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-amuse {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.18);
        color: #38bdf8;
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    /* Primary Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Telemetry grid box */
    .telemetry-metric {
        background: #0b1329;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 14px;
    }
    .telemetry-title {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .telemetry-val {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL & DATA CACHING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Initializing biosensor telemetry engine...")
def load_app_dataset():
    return load_wesad_dataset()

@st.cache_resource(show_spinner="Loading pre-trained AI stress classification model...")
def load_app_model_and_scaler():
    model, scaler = load_wesad_model()
    return model, scaler

@st.cache_data(show_spinner="Loading physiological normalization baselines...")
def load_app_baseline_stats():
    return load_baseline_stats()

try:
    df_data = load_app_dataset()
    model, scaler = load_app_model_and_scaler()
    baseline_stats = load_app_baseline_stats()
except Exception as e:
    st.error(f"System Initialization Error: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="padding: 10px 0 16px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">🫀</span>
                <div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em;">PULSEGUARD AI</div>
                    <div style="font-size: 0.72rem; color: #06b6d4; font-weight: 600;">INTELLIGENT WEARABLE ANALYTICS</div>
                </div>
            </div>
            <div style="color: #94a3b8; font-size: 0.78rem; font-style: italic; margin-top: 8px;">
                "Understand Your Body. Monitor Your Stress."
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #1e293b; margin: 8px 0 16px 0;'>", unsafe_allow_html=True)

    nav_options = ["HOME", "STRESS CHECK", "MY RESULTS"]
    try:
        current_idx = nav_options.index(st.session_state.current_page)
    except ValueError:
        current_idx = 0

    selected_nav = st.radio(
        "NAVIGATION",
        nav_options,
        index=current_idx,
        label_visibility="collapsed"
    )

    if selected_nav != st.session_state.current_page:
        st.session_state.current_page = selected_nav
        st.rerun()

    st.markdown("<hr style='border-color: #1e293b; margin: 24px 0 16px 0;'>", unsafe_allow_html=True)

    # Status indicator
    st.markdown("""
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; font-size: 0.78rem;">
            <div style="color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 6px;">Engine Status</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #cbd5e1;">Random Forest:</span>
                <span style="color: #10b981; font-weight: 600;">● Online</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #cbd5e1;">WESAD Records:</span>
                <span style="color: #38bdf8; font-weight: 600;">1,214 Ready</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Technical Specifications (Optional Collapsible)
    with st.expander("Technical Model Specs"):
        st.caption("""
            **Architecture:** Random Forest Ensemble (100 Trees)  
            **Class Weighting:** Balanced  
            **Features:** 28 biosensor features (ECG, EDA, Resp, Temp, ACC)  
            **Validation:** Subject-Independent (LOSO)  
            **Stress Precision:** 87.9%  
            **Stress Recall:** 73.1%
        """)


# =============================================================================
# PAGE 1: HOME
# =============================================================================
if st.session_state.current_page == "HOME":
    # Hero Section
    st.markdown("""
        <div class="saas-hero-card">
            <span style="display: inline-block; background-color: rgba(6, 182, 212, 0.15); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 20px; padding: 4px 12px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">
                INTELLIGENT PHYSIOLOGICAL MONITORING
            </span>
            <h1 style="color: #ffffff; font-size: 2.5rem; font-weight: 800; line-height: 1.15; margin: 0 0 14px 0; letter-spacing: -0.02em;">
                Your Body Has a Story.<br><span style="color: #06b6d4;">Understand It.</span>
            </h1>
            <p style="color: #94a3b8; font-size: 1.05rem; line-height: 1.6; max-width: 680px; margin: 0 0 24px 0;">
                Analyze physiological signals and discover patterns associated with stress using intelligent wearable data analysis. 
                Identify sympathetic nervous system activation and regain balance with actionable insights.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("⚡ CHECK STRESS NOW", use_container_width=True):
            st.session_state.current_page = "STRESS CHECK"
            st.rerun()

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # How It Works (3 Simple Steps)
    st.markdown("<h3 style='color: #f8fafc; font-size: 1.3rem; font-weight: 700; margin-bottom: 16px;'>How It Works</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
            <div class="pillar-card">
                <div class="pillar-icon">📡</div>
                <div class="pillar-title">1. Capture Biosignals</div>
                <div class="pillar-desc">
                    Wearable sensors continuously monitor vital physiological channels including ECG, Electrodermal Activity (EDA), and respiration.
                </div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="pillar-card">
                <div class="pillar-icon">🧠</div>
                <div class="pillar-title">2. Intelligent Analysis</div>
                <div class="pillar-desc">
                    Our machine learning model analyzes 28 multi-modal features, standardizing against personal resting baselines to detect subtle shifts in autonomic tone.
                </div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="pillar-card">
                <div class="pillar-icon">📊</div>
                <div class="pillar-title">3. Instant Insights</div>
                <div class="pillar-desc">
                    Receive immediate, transparent classification into Baseline, Stress, or Amusement states with confidence metrics and physiological explanations.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # Supported Physiological Signals & Service Overview
    st.markdown("<h3 style='color: #f8fafc; font-size: 1.3rem; font-weight: 700; margin-bottom: 16px;'>Supported Physiological Signals</h3>", unsafe_allow_html=True)
    sig1, sig2, sig3, sig4 = st.columns(4)
    with sig1:
        st.markdown("""
            <div class="telemetry-metric">
                <div class="telemetry-title">Cardiac Dynamics</div>
                <div style="font-weight: 700; color: #f8fafc; margin-top: 4px;">ECG & HRV (RMSSD)</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Tracks autonomic vagal nerve withdrawal and pulse changes.</div>
            </div>
        """, unsafe_allow_html=True)
    with sig2:
        st.markdown("""
            <div class="telemetry-metric">
                <div class="telemetry-title">Electrodermal Activity</div>
                <div style="font-weight: 700; color: #f8fafc; margin-top: 4px;">EDA Tonic & Phasic</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Measures eccrine sweat micro-bursts driven by sympathetic nerve axons.</div>
            </div>
        """, unsafe_allow_html=True)
    with sig3:
        st.markdown("""
            <div class="telemetry-metric">
                <div class="telemetry-title">Respiration Rhythm</div>
                <div style="font-weight: 700; color: #f8fafc; margin-top: 4px;">Breathing Frequency</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Detects hyperventilation, shallow thoracic breathing, and laughter.</div>
            </div>
        """, unsafe_allow_html=True)
    with sig4:
        st.markdown("""
            <div class="telemetry-metric">
                <div class="telemetry-title">Peripheral Biometrics</div>
                <div style="font-weight: 700; color: #f8fafc; margin-top: 4px;">Skin Temp & Motion</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Monitors stress-induced vasoconstriction and filters kinetic artifacts.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="saas-card">
            <h4 style="color: #38bdf8; font-size: 1.05rem; margin: 0 0 8px 0;">About the Service</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; margin: 0;">
                PulseGuard AI translates raw wearable biosignals into clear, real-time stress intelligence. 
                Whether you are evaluating wearable telemetry from clinical cohorts or uploading your own sensor logs, 
                our system delivers continuous visibility into your body's stress response.
            </p>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: STRESS CHECK
# =============================================================================
elif st.session_state.current_page == "STRESS CHECK":
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin-bottom: 4px;">
                🫀 Stress Classification Console
            </h2>
            <p style="color: #94a3b8; font-size: 0.92rem; margin: 0;">
                Select a real participant sample window or upload custom wearable data to analyze stress in real time.
            </p>
        </div>
    """, unsafe_allow_html=True)

    tab_sample, tab_upload = st.tabs(["METHOD 1: SAMPLE DATA", "METHOD 2: UPLOAD DATA"])

    sample_to_analyze = None
    subject_to_analyze = "S4"
    source_identifier = ""

    # -------------------------------------------------------------------------
    # TAB 1: SAMPLE DATA
    # -------------------------------------------------------------------------
    with tab_sample:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        col_s1, col_s2 = st.columns([1, 2])

        with col_s1:
            subj_choice = st.selectbox("Select Participant", ["S4 (Held-Out Test)", "S2 (Training Cohort)", "S3 (Training Cohort)"], index=0)
            subj_id = "S4" if "S4" in subj_choice else ("S2" if "S2" in subj_choice else "S3")

            subj_windows = df_data[df_data[SUBJECT_COLUMN] == subj_id].reset_index(drop=True)
            filter_cat = st.selectbox("Filter Windows by Recorded State", ["All Windows", "Baseline Windows", "Stress Windows", "Amusement Windows"])

            if filter_cat == "Baseline Windows":
                filtered_windows = subj_windows[subj_windows[TARGET_COLUMN] == 0].reset_index(drop=True)
            elif filter_cat == "Stress Windows":
                filtered_windows = subj_windows[subj_windows[TARGET_COLUMN] == 1].reset_index(drop=True)
            elif filter_cat == "Amusement Windows":
                filtered_windows = subj_windows[subj_windows[TARGET_COLUMN] == 2].reset_index(drop=True)
            else:
                filtered_windows = subj_windows

            selected_idx = st.slider("Select Window Index", 0, len(filtered_windows) - 1, 0)
            chosen_row = filtered_windows.iloc[selected_idx]

        with col_s2:
            st.markdown(f"""
                <div class="saas-card" style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="color: #94a3b8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">
                            Telemetric Biosignal Summary ({subj_id} · Window #{selected_idx})
                        </span>
                        <span style="font-size: 0.75rem; color: #38bdf8; font-weight: 600;">
                            WESAD 60-Second Epoch
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                        <div class="telemetry-metric">
                            <div class="telemetry-title">Heart Rate</div>
                            <div class="telemetry-val">{chosen_row['Heart_Rate']:.1f} <span style="font-size: 0.8rem; color: #94a3b8;">BPM</span></div>
                        </div>
                        <div class="telemetry-metric">
                            <div class="telemetry-title">HRV (RMSSD)</div>
                            <div class="telemetry-val">{chosen_row['HRV_RMSSD']:.1f} <span style="font-size: 0.8rem; color: #94a3b8;">ms</span></div>
                        </div>
                        <div class="telemetry-metric">
                            <div class="telemetry-title">EDA Mean</div>
                            <div class="telemetry-val">{chosen_row['EDA_mean']:.2f} <span style="font-size: 0.8rem; color: #94a3b8;">µS</span></div>
                        </div>
                        <div class="telemetry-metric">
                            <div class="telemetry-title">EDA Slope</div>
                            <div class="telemetry-val">{chosen_row['EDA_slope']:+.4f}</div>
                        </div>
                        <div class="telemetry-metric">
                            <div class="telemetry-title">Skin Temp</div>
                            <div class="telemetry-val">{chosen_row['Temp_mean']:.2f} <span style="font-size: 0.8rem; color: #94a3b8;">°C</span></div>
                        </div>
                        <div class="telemetry-metric">
                            <div class="telemetry-title">Breathing Rate</div>
                            <div class="telemetry-val">{chosen_row['Respiratory_Rate']:.1f} <span style="font-size: 0.8rem; color: #94a3b8;">br/min</span></div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        if st.button("🚀 ANALYZE STRESS", key="btn_analyze_sample"):
            sample_to_analyze = chosen_row[FEATURE_COLUMNS]
            subject_to_analyze = subj_id
            source_identifier = f"Participant {subj_id} (Window #{selected_idx})"

    # -------------------------------------------------------------------------
    # TAB 2: UPLOAD DATA
    # -------------------------------------------------------------------------
    with tab_upload:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("""
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                Upload a CSV file containing physiological features extracted from wearable sensors. 
                Ensure your file contains the 28 required feature columns.
            </p>
        """, unsafe_allow_html=True)

        # Downloadable Template Button
        template_df = df_data.head(3)[FEATURE_COLUMNS]
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Compatible CSV Template",
            data=template_csv,
            file_name="pulseguard_sample_template.csv",
            mime="text/csv",
            help="Download a pre-formatted CSV template containing the 28 required physiological features."
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Biosensor CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                missing_cols = [c for c in FEATURE_COLUMNS if c not in uploaded_df.columns]

                if missing_cols:
                    st.error(
                        f"❌ Incompatible Data Format: Missing {len(missing_cols)} required columns: "
                        f"{', '.join(missing_cols[:5])}{'...' if len(missing_cols) > 5 else ''}. "
                        "Please download the compatible CSV template above."
                    )
                else:
                    st.success(f"✅ File successfully validated: {len(uploaded_df)} valid window records found.")
                    st.dataframe(uploaded_df[FEATURE_COLUMNS[:7]].head(5), use_container_width=True)

                    up_row_idx = 0
                    if len(uploaded_df) > 1:
                        up_row_idx = st.slider("Select Record Row to Analyze", 0, len(uploaded_df) - 1, 0, key="slider_up_idx")

                    if st.button("🚀 ANALYZE STRESS", key="btn_analyze_upload"):
                        sample_to_analyze = uploaded_df.iloc[up_row_idx][FEATURE_COLUMNS]
                        subject_to_analyze = uploaded_df.iloc[up_row_idx].get("subject_id", "Unknown")
                        source_identifier = f"Uploaded File ({uploaded_file.name}, Row #{up_row_idx})"

            except Exception as ex:
                st.error(f"Error parsing uploaded CSV: {ex}")

    # -------------------------------------------------------------------------
    # PREDICTION EXECUTION & RESULTS
    # -------------------------------------------------------------------------
    if sample_to_analyze is not None:
        # Animated loading indicator
        with st.spinner("Analyzing physiological biosignals across 28 multi-modal channels..."):
            time.sleep(0.7)
            try:
                pred_result = predict_wesad_window(
                    sample_to_analyze,
                    model=model,
                    scaler=scaler,
                    baseline_stats=baseline_stats,
                    subject_id=subject_to_analyze
                )
            except Exception as err:
                st.error(f"Prediction Engine Error: {err}")
                pred_result = None

        if pred_result is not None:
            st.session_state.latest_prediction = pred_result
            # Save to session history
            st.session_state.analysis_history.append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": source_identifier,
                "predicted_state": pred_result["class_name"],
                "confidence": pred_result["confidence"],
                "probabilities": pred_result["probabilities"],
                "explanation": pred_result["explanation"],
                "vitals": pred_result["vitals"]
            })

    # Render Latest Prediction Result
    if st.session_state.latest_prediction is not None:
        res = st.session_state.latest_prediction
        pred_cls = res["pred_class"]
        pred_name = res["class_name"]
        conf = res["confidence"]
        probs = res["probabilities"]

        # Color coding: Green = Baseline, Red = Stress, Blue = Amusement
        state_color = LABEL_COLORS[pred_cls]
        badge_cls = "badge-baseline" if pred_cls == 0 else ("badge-stress" if pred_cls == 1 else "badge-amuse")

        st.markdown("<hr style='border-color: #1e293b; margin: 24px 0 20px 0;'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="saas-card" style="border: 2px solid {state_color}; background: linear-gradient(145deg, #0f172a 0%, #111c38 100%);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 16px;">
                    <div>
                        <span class="{badge_cls}">PREDICTED PHYSIOLOGICAL STATE</span>
                        <h2 style="color: {state_color}; font-size: 2.2rem; font-weight: 800; margin: 8px 0 2px 0; letter-spacing: -0.01em;">
                            {pred_name.upper()}
                        </h2>
                        <div style="color: #94a3b8; font-size: 0.9rem;">
                            Model Confidence: <strong style="color: #f8fafc;">{conf * 100:.1f}%</strong>
                        </div>
                    </div>
                    <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 20px; text-align: center;">
                        <div style="color: #64748b; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Result Indicator</div>
                        <div style="font-size: 1.5rem; margin-top: 2px;">
                            {'🟢' if pred_cls == 0 else ('🔴' if pred_cls == 1 else '🔵')}
                        </div>
                    </div>
                </div>

                <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.04em;">
                    Prediction Probabilities
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 18px;">
                    <div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                            <span style="color: #10b981; font-weight: 600;">Baseline</span>
                            <span style="color: #f1f5f9; font-weight: 700;">{probs['Baseline'] * 100:.1f}%</span>
                        </div>
                        <div style="background: #0b1329; border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="background: #10b981; width: {probs['Baseline'] * 100}%; height: 100%;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                            <span style="color: #f43f5e; font-weight: 600;">Stress</span>
                            <span style="color: #f1f5f9; font-weight: 700;">{probs['Stress'] * 100:.1f}%</span>
                        </div>
                        <div style="background: #0b1329; border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="background: #f43f5e; width: {probs['Stress'] * 100}%; height: 100%;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px;">
                            <span style="color: #3b82f6; font-weight: 600;">Amusement</span>
                            <span style="color: #f1f5f9; font-weight: 700;">{probs['Amusement'] * 100:.1f}%</span>
                        </div>
                        <div style="background: #0b1329; border-radius: 4px; height: 8px; overflow: hidden;">
                            <div style="background: #3b82f6; width: {probs['Amusement'] * 100}%; height: 100%;"></div>
                        </div>
                    </div>
                </div>

                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 16px 20px;">
                    <div style="color: #38bdf8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">
                        Physiological Analysis
                    </div>
                    <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.55; margin: 0;">
                        {res['explanation']}
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3: MY RESULTS
# =============================================================================
elif st.session_state.current_page == "MY RESULTS":
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin-bottom: 4px;">
                📑 Analysis History & Records
            </h2>
            <p style="color: #94a3b8; font-size: 0.92rem; margin: 0;">
                Review all stress classifications performed during your current session and export assessment reports.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analysis_history:
        st.markdown("""
            <div class="saas-card" style="text-align: center; padding: 48px 24px;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">📋</div>
                <h3 style="color: #f8fafc; font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">
                    No Stress Analyses Recorded Yet
                </h3>
                <p style="color: #94a3b8; font-size: 0.92rem; max-width: 480px; margin: 0 auto 20px auto;">
                    Run your first stress check using sample participant data or an uploaded CSV to see results here.
                </p>
            </div>
        """, unsafe_allow_html=True)

        col_cta, _ = st.columns([1, 2])
        with col_cta:
            if st.button("⚡ GO TO STRESS CHECK", use_container_width=True):
                st.session_state.current_page = "STRESS CHECK"
                st.rerun()

    else:
        # History Table & Controls
        h_df_rows = []
        for i, h in enumerate(reversed(st.session_state.analysis_history)):
            v = h["vitals"]
            h_df_rows.append({
                "Date & Time": h["timestamp"],
                "Data Source": h["source"],
                "Predicted State": h["predicted_state"],
                "Confidence": f"{h['confidence'] * 100:.1f}%",
                "Heart Rate (BPM)": f"{v.get('Heart_Rate', 0.0):.1f}",
                "HRV RMSSD (ms)": f"{v.get('HRV_RMSSD', 0.0):.1f}",
                "EDA Mean (µS)": f"{v.get('EDA_mean', 0.0):.2f}",
                "Breathing Rate": f"{v.get('Respiratory_Rate', 0.0):.1f}"
            })

        history_df = pd.DataFrame(h_df_rows)

        # Overview Stats
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f"""
                <div class="telemetry-metric">
                    <div class="telemetry-title">Total Analyses</div>
                    <div class="telemetry-val">{len(st.session_state.analysis_history)}</div>
                </div>
            """, unsafe_allow_html=True)
        with s2:
            latest = st.session_state.analysis_history[-1]
            st.markdown(f"""
                <div class="telemetry-metric">
                    <div class="telemetry-title">Latest Diagnosis</div>
                    <div class="telemetry-val" style="color: {LABEL_COLORS[0 if latest['predicted_state']=='Baseline' else (1 if latest['predicted_state']=='Stress' else 2)]};">
                        {latest['predicted_state']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with s3:
            stress_count = sum(1 for h in st.session_state.analysis_history if h['predicted_state'] == 'Stress')
            st.markdown(f"""
                <div class="telemetry-metric">
                    <div class="telemetry-title">Stress Episodes Detected</div>
                    <div class="telemetry-val" style="color: #f43f5e;">{stress_count}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.dataframe(history_df, use_container_width=True)

        # Action Buttons: Export & Clear
        col_exp1, col_exp2, col_clr = st.columns([1, 1, 1])

        with col_exp1:
            csv_data = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_data,
                file_name=f"pulseguard_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_exp2:
            # Formatted text report
            txt_report = "=================================================================\n"
            txt_report += "PULSEGUARD AI — CLINICAL STRESS ASSESSMENT DOSSIER\n"
            txt_report += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            txt_report += "=================================================================\n\n"
            for h in reversed(st.session_state.analysis_history):
                txt_report += f"Timestamp:       {h['timestamp']}\n"
                txt_report += f"Source:          {h['source']}\n"
                txt_report += f"Diagnosis:       {h['predicted_state'].upper()} (Confidence: {h['confidence']*100:.1f}%)\n"
                txt_report += f"Vitals:          HR: {h['vitals'].get('Heart_Rate', 0.0):.1f} BPM | HRV: {h['vitals'].get('HRV_RMSSD', 0.0):.1f} ms | EDA: {h['vitals'].get('EDA_mean', 0.0):.2f} uS\n"
                txt_report += f"Analysis:        {h['explanation']}\n"
                txt_report += "-----------------------------------------------------------------\n"
            st.download_button(
                label="📄 Download Report (TXT)",
                data=txt_report,
                file_name=f"pulseguard_dossier_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col_clr:
            if st.button("🗑️ Clear Analysis History", use_container_width=True):
                st.session_state.analysis_history = []
                st.session_state.latest_prediction = None
                st.rerun()
