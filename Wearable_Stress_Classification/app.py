"""
=============================================================================
PULSEGUARD AI — WEARABLE BIOSIGNAL STRESS ANALYTICS
Tagline: Understand Your Body. Monitor Your Stress.
Platform: Streamlit Healthcare Technology Web Application
=============================================================================
"""

import os
import sys
import io
import time
import textwrap
import datetime
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure local imports resolve seamlessly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from auth import (
    init_db,
    register_user,
    authenticate_user
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

# Initialize database
init_db()

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
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "current_page" not in st.session_state:
    st.session_state.current_page = "HOME"

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

if "selected_sensor_channel" not in st.session_state:
    st.session_state.selected_sensor_channel = "Heart Rate"

# -----------------------------------------------------------------------------
# HEALTHCARE SAAS CSS
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
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }

    .saas-hero-card {
        background: radial-gradient(circle at 85% 20%, rgba(6, 182, 212, 0.15) 0%, #0f172a 65%);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 36px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
    }

    .pillar-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .pillar-icon {
        font-size: 1.8rem;
        margin-bottom: 10px;
    }
    .pillar-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .pillar-desc {
        color: #94a3b8;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    /* Interactive Sensor Tile */
    .sensor-tile {
        background: #0b1329;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: border-color 0.2s ease, transform 0.1s ease;
    }
    .sensor-tile:hover {
        border-color: #06b6d4;
        transform: translateY(-1px);
    }
    .sensor-tile-active {
        background: #111c38;
        border: 1px solid #06b6d4 !important;
        box-shadow: 0 0 12px rgba(6, 182, 212, 0.2);
    }

    /* Badges */
    .badge-baseline {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.16);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 6px;
        padding: 5px 12px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-stress {
        display: inline-block;
        background-color: rgba(244, 63, 94, 0.18);
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-radius: 6px;
        padding: 5px 12px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-amuse {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.18);
        color: #38bdf8;
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 6px;
        padding: 5px 12px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    /* Primary Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Footer styling */
    .saas-footer {
        margin-top: 50px;
        border-top: 1px solid #1e293b;
        padding: 24px 0 16px 0;
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA & CACHING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Initializing biosensor telemetry engine...")
def load_app_dataset():
    return load_wesad_dataset()

@st.cache_resource(show_spinner="Loading pre-trained AI stress classification model...")
def load_app_model_and_scaler():
    return load_wesad_model()

@st.cache_data(show_spinner="Loading physiological normalization baselines...")
def load_app_baseline_stats():
    return load_baseline_stats()

@st.cache_data(show_spinner="Loading multi-channel raw sensor waveforms...")
def load_app_raw_samples():
    raw_path = RAW_SAMPLES_FILENAME
    if not os.path.exists(raw_path):
        parent_raw = os.path.join(os.path.dirname(BASE_DIR), "wesad_raw_samples.pkl")
        if os.path.exists(parent_raw):
            raw_path = parent_raw
        else:
            return None
    with open(raw_path, "rb") as f:
        return pickle.load(f)

try:
    df_data = load_app_dataset()
    model, scaler = load_app_model_and_scaler()
    baseline_stats = load_app_baseline_stats()
    raw_samples = load_app_raw_samples()
except Exception as e:
    st.error(f"System Initialization Error: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# FOOTER COMPONENT
# -----------------------------------------------------------------------------
def render_footer():
    st.markdown(textwrap.dedent("""
        <div class="saas-footer">
            <div style="font-weight: 700; color: #94a3b8; margin-bottom: 4px;">
                © 2026 PulseGuard AI. All Rights Reserved.
            </div>
            <div style="font-style: italic; color: #64748b; margin-bottom: 10px;">
                "Understand Your Body. Monitor Your Stress."
            </div>
            <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 8px;">
                <strong>Dataset Attribution:</strong> Developed using the open-access WESAD (Wearable Stress and Affect Detection) 
                multimodal benchmark dataset created by researchers at the University of Siegen (Schmidt et al., ACM ICMI 2018 / UCI ID 465). 
                PulseGuard AI does not claim proprietary ownership of original WESAD recordings.
            </div>
            <div style="font-size: 0.72rem; color: #475569;">
                Medical Disclaimer: PulseGuard AI is a research and educational platform. Its predictions do not constitute autonomous medical advice or diagnosis.
            </div>
        </div>
    """), unsafe_allow_html=True)


# =============================================================================
# AUTHENTICATION WALL (LOGIN & REGISTRATION)
# =============================================================================
if not st.session_state.is_authenticated:
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    col_l1, col_l2, col_l3 = st.columns([1, 1.8, 1])
    with col_l2:
        st.markdown(textwrap.dedent("""
            <div style="text-align: center; margin-bottom: 24px;">
                <div style="display: inline-flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                    <span style="font-size: 2.2rem;">🫀</span>
                    <span style="font-size: 1.9rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">PULSEGUARD AI</span>
                </div>
                <div style="color: #06b6d4; font-size: 0.95rem; font-weight: 600;">
                    "Understand Your Body. Monitor Your Stress."
                </div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 6px;">
                    Sign in to access your intelligent wearable biosignal analytics dashboard
                </div>
            </div>
        """), unsafe_allow_html=True)

        auth_tab1, auth_tab2 = st.tabs(["🔐 Sign In", "📝 Create Account"])

        with auth_tab1:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            with st.form("login_form"):
                login_email = st.text_input("Email Address", placeholder="name@example.com")
                login_password = st.text_input("Password", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("Sign In to PulseGuard AI", use_container_width=True)

                if submit_login:
                    ok, user_data, msg = authenticate_user(login_email, login_password)
                    if ok:
                        st.session_state.is_authenticated = True
                        st.session_state.user_email = user_data["email"]
                        st.session_state.user_name = user_data["full_name"]
                        st.success("Authentication successful! Welcome to PulseGuard AI.")
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

            st.markdown(textwrap.dedent("""
                <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px 16px; margin-top: 16px; font-size: 0.82rem;">
                    <strong style="color: #38bdf8;">💡 Instant Demo Credentials:</strong><br>
                    <span style="color: #cbd5e1;">Email:</span> <code style="color: #38bdf8;">demo@pulseguard.ai</code><br>
                    <span style="color: #cbd5e1;">Password:</span> <code style="color: #38bdf8;">Password123!</code>
                </div>
            """), unsafe_allow_html=True)

            with st.expander("Forgot your password?"):
                st.caption(
                    "For security, password recovery is linked to your verified organization or clinical contact. "
                    "Please contact support@pulseguard.ai or use the pre-configured demo credentials above to sign in."
                )

        with auth_tab2:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            with st.form("register_form"):
                reg_name = st.text_input("Full Name", placeholder="e.g. Jane Doe")
                reg_email = st.text_input("Email Address", placeholder="name@example.com")
                reg_pwd = st.text_input("Create Password", type="password", placeholder="At least 6 characters")
                reg_pwd_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
                submit_reg = st.form_submit_button("Create Account", use_container_width=True)

                if submit_reg:
                    if reg_pwd != reg_pwd_confirm:
                        st.error("Passwords do not match. Please verify your password entry.")
                    else:
                        reg_ok, reg_msg = register_user(reg_name, reg_email, reg_pwd)
                        if reg_ok:
                            st.success(f"✅ {reg_msg}")
                            st.info("Please switch to the 'Sign In' tab to log in with your new credentials.")
                        else:
                            st.error(f"❌ {reg_msg}")

    render_footer()
    st.stop()


# =============================================================================
# AUTHENTICATED APPLICATION
# =============================================================================

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(textwrap.dedent(f"""
        <div style="padding: 10px 0 16px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">🫀</span>
                <div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em;">PULSEGUARD AI</div>
                    <div style="font-size: 0.72rem; color: #06b6d4; font-weight: 600;">INTELLIGENT WEARABLE ANALYTICS</div>
                </div>
            </div>
            <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 8px 12px; margin-top: 12px; font-size: 0.78rem;">
                <span style="color: #64748b;">Logged in as:</span><br>
                <strong style="color: #f1f5f9;">{st.session_state.user_name}</strong><br>
                <span style="color: #38bdf8; font-size: 0.72rem;">{st.session_state.user_email}</span>
            </div>
        </div>
    """), unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #1e293b; margin: 4px 0 14px 0;'>", unsafe_allow_html=True)

    nav_options = ["HOME", "STRESS CHECK", "MY RESULTS", "ABOUT"]
    try:
        current_idx = nav_options.index(st.session_state.current_page)
    except ValueError:
        current_idx = 0

    selected_nav = st.radio(
        "MAIN NAVIGATION",
        nav_options,
        index=current_idx,
        label_visibility="collapsed"
    )

    if selected_nav != st.session_state.current_page:
        st.session_state.current_page = selected_nav
        st.rerun()

    st.markdown("<hr style='border-color: #1e293b; margin: 20px 0 14px 0;'>", unsafe_allow_html=True)

    # Logout Button
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.is_authenticated = False
        st.session_state.user_email = ""
        st.session_state.user_name = ""
        st.session_state.current_page = "HOME"
        st.rerun()

    # System Status
    st.markdown(textwrap.dedent("""
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; font-size: 0.78rem; margin-top: 14px;">
            <div style="color: #64748b; font-weight: 600; text-transform: uppercase; margin-bottom: 6px;">Engine Telemetry</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #cbd5e1;">Random Forest:</span>
                <span style="color: #10b981; font-weight: 600;">● Online</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #cbd5e1;">WESAD Records:</span>
                <span style="color: #38bdf8; font-weight: 600;">1,214 Ready</span>
            </div>
        </div>
    """), unsafe_allow_html=True)


# =============================================================================
# PAGE 1: HOME
# =============================================================================
if st.session_state.current_page == "HOME":
    st.markdown(textwrap.dedent("""
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
    """), unsafe_allow_html=True)

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
        st.markdown(textwrap.dedent("""
            <div class="pillar-card">
                <div class="pillar-icon">📡</div>
                <div class="pillar-title">1. Capture Biosignals</div>
                <div class="pillar-desc">
                    Wearable sensors continuously monitor vital physiological channels including ECG, Electrodermal Activity (EDA), and respiration.
                </div>
            </div>
        """), unsafe_allow_html=True)
    with c2:
        st.markdown(textwrap.dedent("""
            <div class="pillar-card">
                <div class="pillar-icon">🧠</div>
                <div class="pillar-title">2. Intelligent Analysis</div>
                <div class="pillar-desc">
                    Our machine learning model analyzes 28 multi-modal features, standardizing against personal resting baselines to detect subtle shifts in autonomic tone.
                </div>
            </div>
        """), unsafe_allow_html=True)
    with c3:
        st.markdown(textwrap.dedent("""
            <div class="pillar-card">
                <div class="pillar-icon">📊</div>
                <div class="pillar-title">3. Instant Insights</div>
                <div class="pillar-desc">
                    Receive immediate, transparent classification into Baseline, Stress, or Amusement states with confidence metrics and physiological explanations.
                </div>
            </div>
        """), unsafe_allow_html=True)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # Supported Physiological Signals & Service Overview
    st.markdown("<h3 style='color: #f8fafc; font-size: 1.3rem; font-weight: 700; margin-bottom: 16px;'>Supported Physiological Signals</h3>", unsafe_allow_html=True)
    sig1, sig2, sig3, sig4 = st.columns(4)
    with sig1:
        st.markdown(textwrap.dedent("""
            <div class="pillar-card">
                <div style="color: #38bdf8; font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">🫀 ECG & HRV (RMSSD)</div>
                <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.45;">Tracks autonomic vagal nerve withdrawal, heart rate fluctuations, and beat-to-beat variability.</div>
            </div>
        """), unsafe_allow_html=True)
    with sig2:
        st.markdown(textwrap.dedent("""
            <div class="pillar-card">
                <div style="color: #10b981; font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">⚡ EDA Tonic & Phasic</div>
                <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.45;">Measures skin conductance and sweat gland micro-bursts driven exclusively by sympathetic nerve axons.</div>
            </div>
        """), unsafe_allow_html=True)
    with sig3:
        st.markdown(textwrap.dedent("""
            <div class="pillar-card">
                <div style="color: #a855f7; font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">🫁 Respiration Dynamics</div>
                <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.45;">Monitors chest expansion, breathing frequency, shallow ventilatory drive, and laughter flutter.</div>
            </div>
        """), unsafe_allow_html=True)
    with sig4:
        st.markdown(textwrap.dedent("""
            <div class="pillar-card">
                <div style="color: #f59e0b; font-weight: 700; font-size: 0.95rem; margin-bottom: 4px;">🌡️ Skin Temp & Motion</div>
                <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.45;">Monitors stress-induced peripheral vasoconstriction and filters motion artifacts with 3D ACC.</div>
            </div>
        """), unsafe_allow_html=True)

    render_footer()


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
    condition_to_analyze = "Baseline"
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
            chosen_cond = chosen_row.get("condition", "Baseline")

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            if st.button("🚀 ANALYZE STRESS", key="btn_analyze_sample", use_container_width=True):
                sample_to_analyze = chosen_row[FEATURE_COLUMNS]
                subject_to_analyze = subj_id
                condition_to_analyze = chosen_cond
                source_identifier = f"Participant {subj_id} (Window #{selected_idx})"

        with col_s2:
            st.markdown(textwrap.dedent(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #f8fafc; font-size: 0.95rem; font-weight: 700;">
                        Telemetric Biosignal Summary ({subj_id} · Window #{selected_idx})
                    </span>
                    <span style="font-size: 0.75rem; color: #38bdf8; font-weight: 600;">
                        ● WESAD 60-Second Window
                    </span>
                </div>
            """), unsafe_allow_html=True)

            # Interactive Sensor Selector (Pills)
            sensor_options = [
                "Heart Rate",
                "Heart Rate Variability",
                "Electrodermal Activity",
                "Skin Temperature",
                "Respiration Rate",
                "Blood Volume Pulse",
                "Accelerometer"
            ]

            active_sensor = st.segmented_control(
                "Click a sensor value below to inspect measurement details and signal graph:",
                sensor_options,
                default=st.session_state.selected_sensor_channel
            )
            if active_sensor is None:
                active_sensor = "Heart Rate"
            st.session_state.selected_sensor_channel = active_sensor

            # Summary Grid
            g1, g2, g3 = st.columns(3)
            with g1:
                st.markdown(textwrap.dedent(f"""
                    <div class="sensor-tile {'sensor-tile-active' if active_sensor == 'Heart Rate' else ''}">
                        <div style="color: #64748b; font-size: 0.72rem; font-weight: 600;">🫀 HEART RATE</div>
                        <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 700;">{chosen_row['Heart_Rate']:.1f} <span style="font-size: 0.75rem; color: #94a3b8;">BPM</span></div>
                        <div style="color: #10b981; font-size: 0.7rem;">Normal: 60-100 BPM</div>
                    </div>
                """), unsafe_allow_html=True)
                st.markdown(textwrap.dedent(f"""
                    <div class="sensor-tile {'sensor-tile-active' if active_sensor == 'Heart Rate Variability' else ''}">
                        <div style="color: #64748b; font-size: 0.72rem; font-weight: 600;">🫀 HRV (RMSSD)</div>
                        <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 700;">{chosen_row['HRV_RMSSD']:.1f} <span style="font-size: 0.75rem; color: #94a3b8;">ms</span></div>
                        <div style="color: #38bdf8; font-size: 0.7rem;">Normal: 20-50+ ms</div>
                    </div>
                """), unsafe_allow_html=True)
            with g2:
                st.markdown(textwrap.dedent(f"""
                    <div class="sensor-tile {'sensor-tile-active' if active_sensor == 'Electrodermal Activity' else ''}">
                        <div style="color: #64748b; font-size: 0.72rem; font-weight: 600;">⚡ EDA CONDUCTANCE</div>
                        <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 700;">{chosen_row['EDA_mean']:.2f} <span style="font-size: 0.75rem; color: #94a3b8;">µS</span></div>
                        <div style="color: #f59e0b; font-size: 0.7rem;">Slope: {chosen_row['EDA_slope']:+.4f}</div>
                    </div>
                """), unsafe_allow_html=True)
                st.markdown(textwrap.dedent(f"""
                    <div class="sensor-tile {'sensor-tile-active' if active_sensor == 'Skin Temperature' else ''}">
                        <div style="color: #64748b; font-size: 0.72rem; font-weight: 600;">🌡️ SKIN TEMPERATURE</div>
                        <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 700;">{chosen_row['Temp_mean']:.2f} <span style="font-size: 0.75rem; color: #94a3b8;">°C</span></div>
                        <div style="color: #10b981; font-size: 0.7rem;">Normal: 32-35.5 °C</div>
                    </div>
                """), unsafe_allow_html=True)
            with g3:
                st.markdown(textwrap.dedent(f"""
                    <div class="sensor-tile {'sensor-tile-active' if active_sensor == 'Respiration Rate' else ''}">
                        <div style="color: #64748b; font-size: 0.72rem; font-weight: 600;">🫁 RESPIRATION RATE</div>
                        <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 700;">{chosen_row['Respiratory_Rate']:.1f} <span style="font-size: 0.75rem; color: #94a3b8;">br/min</span></div>
                        <div style="color: #10b981; font-size: 0.7rem;">Normal: 12-20 br/min</div>
                    </div>
                """), unsafe_allow_html=True)
                st.markdown(textwrap.dedent(f"""
                    <div class="sensor-tile {'sensor-tile-active' if active_sensor == 'Accelerometer' else ''}">
                        <div style="color: #64748b; font-size: 0.72rem; font-weight: 600;">🏃 ACCELEROMETER</div>
                        <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 700;">{chosen_row['ACC_mean']:.2f} <span style="font-size: 0.75rem; color: #94a3b8;">g</span></div>
                        <div style="color: #94a3b8; font-size: 0.7rem;">Motion Artifact: Minimal</div>
                    </div>
                """), unsafe_allow_html=True)

            # Interactive Detail Card with Signal Graph
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            
            # Fetch raw waveform trace if available
            raw_trace = raw_samples.get(subj_id, {}).get(chosen_cond, None) if raw_samples else None

            if active_sensor in ["Heart Rate", "Heart Rate Variability"]:
                st.markdown(textwrap.dedent(f"""
                    <div class="saas-card" style="padding: 16px; margin-bottom: 12px; border-color: #0284c7;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong style="color: #38bdf8; font-size: 0.95rem;">🫀 Cardiac Autonomic Channel ({active_sensor})</strong>
                            <span style="font-size: 0.72rem; background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 4px; font-weight: 700;">
                                Available (Lead-II ECG)
                            </span>
                        </div>
                        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                            <strong>Current Measured Value:</strong> {chosen_row['Heart_Rate']:.1f} BPM | <strong>HRV RMSSD:</strong> {chosen_row['HRV_RMSSD']:.1f} ms<br>
                            <strong>Scientifically Supported Range:</strong> 60 – 100 BPM resting heart rate; RMSSD 20 – 50+ ms. Suppressed RMSSD reflects vagal withdrawal during acute mental stress.<br>
                            <strong>Measurement Type:</strong> Continuous Lead-II Electrocardiogram captured at 700 Hz.
                        </div>
                    </div>
                """), unsafe_allow_html=True)

                if raw_trace and "ECG" in raw_trace:
                    fig, ax = plt.subplots(figsize=(7, 2.0))
                    fig.patch.set_facecolor('#0b1329')
                    ax.set_facecolor('#0f172a')
                    ax.plot(raw_trace["time_sec"], raw_trace["ECG"], color="#38bdf8", linewidth=1.2)
                    ax.set_ylabel("ECG (mV)", color="#94a3b8", fontsize=8)
                    ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=8)
                    ax.tick_params(colors="#94a3b8", labelsize=7)
                    ax.grid(color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.7)
                    ax.set_title(f"10-Second Lead-II ECG Waveform Trace ({subj_id} · {chosen_cond})", color="#f8fafc", fontsize=9, pad=6)
                    for spine in ax.spines.values():
                        spine.set_color("#1e293b")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            elif active_sensor == "Electrodermal Activity":
                st.markdown(textwrap.dedent(f"""
                    <div class="saas-card" style="padding: 16px; margin-bottom: 12px; border-color: #0284c7;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong style="color: #06b6d4; font-size: 0.95rem;">⚡ Electrodermal Activity (Skin Conductance)</strong>
                            <span style="font-size: 0.72rem; background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 4px; font-weight: 700;">
                                Available (Raw Biosensor)
                            </span>
                        </div>
                        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                            <strong>Current Measured Value:</strong> {chosen_row['EDA_mean']:.2f} µS (Slope: {chosen_row['EDA_slope']:+.4f} µS/s)<br>
                            <strong>Scientifically Supported Range:</strong> 0.2 – 10.0 µS. Basal levels vary per individual. Elevated mean and positive slope indicate acute sympathetic arousal.<br>
                            <strong>Measurement Type:</strong> Galvanic skin conductance measured via RespiBAN chest electrodes at 700 Hz.
                        </div>
                    </div>
                """), unsafe_allow_html=True)

                if raw_trace and "EDA" in raw_trace:
                    fig, ax = plt.subplots(figsize=(7, 2.0))
                    fig.patch.set_facecolor('#0b1329')
                    ax.set_facecolor('#0f172a')
                    ax.plot(raw_trace["time_sec"], raw_trace["EDA"], color="#06b6d4", linewidth=1.4)
                    ax.set_ylabel("EDA (µS)", color="#94a3b8", fontsize=8)
                    ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=8)
                    ax.tick_params(colors="#94a3b8", labelsize=7)
                    ax.grid(color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.7)
                    ax.set_title(f"10-Second Continuous EDA Waveform ({subj_id} · {chosen_cond})", color="#f8fafc", fontsize=9, pad=6)
                    for spine in ax.spines.values():
                        spine.set_color("#1e293b")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            elif active_sensor == "Skin Temperature":
                st.markdown(textwrap.dedent(f"""
                    <div class="saas-card" style="padding: 16px; margin-bottom: 12px; border-color: #0284c7;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong style="color: #f59e0b; font-size: 0.95rem;">🌡️ Peripheral Skin Temperature</strong>
                            <span style="font-size: 0.72rem; background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 4px; font-weight: 700;">
                                Available (Raw Biosensor)
                            </span>
                        </div>
                        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                            <strong>Current Measured Value:</strong> {chosen_row['Temp_mean']:.2f} °C (Slope: {chosen_row['Temp_slope']:+.4f})<br>
                            <strong>Scientifically Supported Range:</strong> 32.0 – 35.5 °C. Peripheral vasoconstriction during acute sympathetic fight-or-flight reduces skin temperature.<br>
                            <strong>Measurement Type:</strong> Thermistor thermal sensor sampled continuously at 700 Hz.
                        </div>
                    </div>
                """), unsafe_allow_html=True)

                if raw_trace and "Temp" in raw_trace:
                    fig, ax = plt.subplots(figsize=(7, 2.0))
                    fig.patch.set_facecolor('#0b1329')
                    ax.set_facecolor('#0f172a')
                    ax.plot(raw_trace["time_sec"], raw_trace["Temp"], color="#f59e0b", linewidth=1.4)
                    ax.set_ylabel("Temp (°C)", color="#94a3b8", fontsize=8)
                    ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=8)
                    ax.tick_params(colors="#94a3b8", labelsize=7)
                    ax.grid(color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.7)
                    ax.set_title(f"10-Second Skin Temperature Trace ({subj_id} · {chosen_cond})", color="#f8fafc", fontsize=9, pad=6)
                    for spine in ax.spines.values():
                        spine.set_color("#1e293b")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            elif active_sensor == "Respiration Rate":
                st.markdown(textwrap.dedent(f"""
                    <div class="saas-card" style="padding: 16px; margin-bottom: 12px; border-color: #0284c7;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong style="color: #a855f7; font-size: 0.95rem;">🫁 Respiration Rate & Thoracic Expansion</strong>
                            <span style="font-size: 0.72rem; background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 4px; font-weight: 700;">
                                Available (Inductive Plethysmography)
                            </span>
                        </div>
                        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                            <strong>Current Measured Value:</strong> {chosen_row['Respiratory_Rate']:.1f} breaths/min<br>
                            <strong>Scientifically Supported Range:</strong> 12 – 20 breaths/min. Rapid or shallow breathing often correlates with stress or cognitive load.<br>
                            <strong>Measurement Type:</strong> Inductive chest band measuring thoracic expansion and contraction at 700 Hz.
                        </div>
                    </div>
                """), unsafe_allow_html=True)

                if raw_trace and "Resp" in raw_trace:
                    fig, ax = plt.subplots(figsize=(7, 2.0))
                    fig.patch.set_facecolor('#0b1329')
                    ax.set_facecolor('#0f172a')
                    ax.plot(raw_trace["time_sec"], raw_trace["Resp"], color="#a855f7", linewidth=1.3)
                    ax.set_ylabel("Resp (A.U.)", color="#94a3b8", fontsize=8)
                    ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=8)
                    ax.tick_params(colors="#94a3b8", labelsize=7)
                    ax.grid(color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.7)
                    ax.set_title(f"10-Second Thoracic Respiration Waveform ({subj_id} · {chosen_cond})", color="#f8fafc", fontsize=9, pad=6)
                    for spine in ax.spines.values():
                        spine.set_color("#1e293b")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            elif active_sensor == "Blood Volume Pulse":
                st.markdown(textwrap.dedent("""
                    <div class="saas-card" style="padding: 16px; margin-bottom: 12px; border-color: #0284c7;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong style="color: #f43f5e; font-size: 0.95rem;">🩸 Blood Volume Pulse (BVP)</strong>
                            <span style="font-size: 0.72rem; background: rgba(245, 158, 11, 0.2); color: #f59e0b; padding: 2px 8px; border-radius: 4px; font-weight: 700;">
                                Wrist-Sensor Only (Empatica E4)
                            </span>
                        </div>
                        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                            <strong>Current Measured Value:</strong> N/A for chest unit (Captured on Empatica E4 wristband in WESAD).<br>
                            <strong>Physiological Context:</strong> Optical photoplethysmography (PPG) captures arterial pulsation. In this model, 
                            cardiac features are extracted directly from the higher-fidelity medical-grade Lead-II ECG chest sensor.
                        </div>
                    </div>
                """), unsafe_allow_html=True)

            elif active_sensor == "Accelerometer":
                st.markdown(textwrap.dedent(f"""
                    <div class="saas-card" style="padding: 16px; margin-bottom: 12px; border-color: #0284c7;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <strong style="color: #94a3b8; font-size: 0.95rem;">🏃 3D Accelerometer (Motion & Kinetic Artifacts)</strong>
                            <span style="font-size: 0.72rem; background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 4px; font-weight: 700;">
                                Available (Raw Biosensor)
                            </span>
                        </div>
                        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                            <strong>Current Measured Value:</strong> Mean: {chosen_row['ACC_mean']:.2f} g | Max: {chosen_row['ACC_max']:.2f} g<br>
                            <strong>Scientifically Supported Range:</strong> 0.8 – 1.2 g (Earth gravitational baseline for sedentary recording). Ensures physiological changes are not induced by gross motor movement.<br>
                            <strong>Measurement Type:</strong> 3-Axis accelerometer on chest RespiBAN unit.
                        </div>
                    </div>
                """), unsafe_allow_html=True)

                if raw_trace and "ACC_mag" in raw_trace:
                    fig, ax = plt.subplots(figsize=(7, 2.0))
                    fig.patch.set_facecolor('#0b1329')
                    ax.set_facecolor('#0f172a')
                    ax.plot(raw_trace["time_sec"], raw_trace["ACC_mag"], color="#94a3b8", linewidth=1.2)
                    ax.set_ylabel("ACC (g)", color="#94a3b8", fontsize=8)
                    ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=8)
                    ax.tick_params(colors="#94a3b8", labelsize=7)
                    ax.grid(color="#1e293b", linestyle="--", linewidth=0.5, alpha=0.7)
                    ax.set_title(f"10-Second 3D Accelerometer Trace ({subj_id} · {chosen_cond})", color="#f8fafc", fontsize=9, pad=6)
                    for spine in ax.spines.values():
                        spine.set_color("#1e293b")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

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

                    if st.button("🚀 ANALYZE STRESS", key="btn_analyze_upload", use_container_width=True):
                        sample_to_analyze = uploaded_df.iloc[up_row_idx][FEATURE_COLUMNS]
                        subject_to_analyze = str(uploaded_df.iloc[up_row_idx].get("subject_id", "Unknown"))
                        source_identifier = f"Uploaded File ({uploaded_file.name}, Row #{up_row_idx})"

            except Exception as ex:
                st.error(f"Error parsing uploaded CSV: {ex}")

    # -------------------------------------------------------------------------
    # PREDICTION EXECUTION & RESULTS (FIXED RENDERING)
    # -------------------------------------------------------------------------
    if sample_to_analyze is not None:
        with st.spinner("Analyzing physiological biosignals across 28 multi-modal channels..."):
            time.sleep(0.6)
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

        state_color = LABEL_COLORS[pred_cls]
        badge_cls = "badge-baseline" if pred_cls == 0 else ("badge-stress" if pred_cls == 1 else "badge-amuse")
        indicator_symbol = "🟢" if pred_cls == 0 else ("🔴" if pred_cls == 1 else "🔵")

        st.markdown("<hr style='border-color: #1e293b; margin: 28px 0 20px 0;'>", unsafe_allow_html=True)
        
        # Main Result Card
        st.markdown(textwrap.dedent(f"""
            <div class="saas-card" style="border: 2px solid {state_color};">
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
                        <div style="color: #64748b; font-size: 0.72rem; font-weight: 600; text-transform: uppercase;">Visual Indicator</div>
                        <div style="font-size: 1.6rem; margin-top: 2px;">
                            {indicator_symbol}
                        </div>
                    </div>
                </div>
            </div>
        """), unsafe_allow_html=True)

        # Native Prediction Probabilities Section (100% immune to code-block escaping)
        st.markdown("<h4 style='color: #f8fafc; font-size: 1.05rem; margin-bottom: 12px;'>Prediction Probabilities</h4>", unsafe_allow_html=True)
        prob_c1, prob_c2, prob_c3 = st.columns(3)
        with prob_c1:
            st.markdown(textwrap.dedent(f"""
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px;">
                    <strong style="color: #10b981;">Baseline</strong>
                    <span style="color: #f1f5f9; font-weight: 700;">{probs['Baseline'] * 100:.1f}%</span>
                </div>
            """), unsafe_allow_html=True)
            st.progress(float(probs["Baseline"]))

        with prob_c2:
            st.markdown(textwrap.dedent(f"""
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px;">
                    <strong style="color: #f43f5e;">Stress</strong>
                    <span style="color: #f1f5f9; font-weight: 700;">{probs['Stress'] * 100:.1f}%</span>
                </div>
            """), unsafe_allow_html=True)
            st.progress(float(probs["Stress"]))

        with prob_c3:
            st.markdown(textwrap.dedent(f"""
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px;">
                    <strong style="color: #3b82f6;">Amusement</strong>
                    <span style="color: #f1f5f9; font-weight: 700;">{probs['Amusement'] * 100:.1f}%</span>
                </div>
            """), unsafe_allow_html=True)
            st.progress(float(probs["Amusement"]))

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Physiological Explanation Box
        st.markdown(textwrap.dedent(f"""
            <div class="saas-card" style="padding: 18px 20px;">
                <div style="color: #38bdf8; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">
                    Physiological Analysis
                </div>
                <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6; margin: 0;">
                    {res['explanation']}
                </p>
            </div>
        """), unsafe_allow_html=True)

    render_footer()


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
        st.markdown(textwrap.dedent("""
            <div class="saas-card" style="text-align: center; padding: 48px 24px;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">📋</div>
                <h3 style="color: #f8fafc; font-size: 1.25rem; font-weight: 700; margin-bottom: 8px;">
                    No Stress Analyses Recorded Yet
                </h3>
                <p style="color: #94a3b8; font-size: 0.92rem; max-width: 480px; margin: 0 auto 20px auto;">
                    Run your first stress check using sample participant data or an uploaded CSV to see results here.
                </p>
            </div>
        """), unsafe_allow_html=True)

        col_cta, _ = st.columns([1, 2])
        with col_cta:
            if st.button("⚡ GO TO STRESS CHECK", use_container_width=True):
                st.session_state.current_page = "STRESS CHECK"
                st.rerun()

    else:
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

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(textwrap.dedent(f"""
                <div class="pillar-card">
                    <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">Total Analyses</div>
                    <div style="font-size: 1.7rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">{len(st.session_state.analysis_history)}</div>
                </div>
            """), unsafe_allow_html=True)
        with s2:
            latest = st.session_state.analysis_history[-1]
            latest_col = LABEL_COLORS[0 if latest['predicted_state']=='Baseline' else (1 if latest['predicted_state']=='Stress' else 2)]
            st.markdown(textwrap.dedent(f"""
                <div class="pillar-card">
                    <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">Latest Diagnosis</div>
                    <div style="font-size: 1.7rem; font-weight: 700; color: {latest_col}; margin-top: 4px;">{latest['predicted_state']}</div>
                </div>
            """), unsafe_allow_html=True)
        with s3:
            stress_count = sum(1 for h in st.session_state.analysis_history if h['predicted_state'] == 'Stress')
            st.markdown(textwrap.dedent(f"""
                <div class="pillar-card">
                    <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">Stress Events Detected</div>
                    <div style="font-size: 1.7rem; font-weight: 700; color: #f43f5e; margin-top: 4px;">{stress_count}</div>
                </div>
            """), unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.dataframe(history_df, use_container_width=True)

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

    render_footer()


# =============================================================================
# PAGE 4: ABOUT
# =============================================================================
elif st.session_state.current_page == "ABOUT":
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 800; margin-bottom: 4px;">
                ℹ️ About PulseGuard AI
            </h2>
            <p style="color: #06b6d4; font-size: 0.95rem; font-weight: 600; margin: 0;">
                "Understand Your Body. Monitor Your Stress."
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(textwrap.dedent("""
        <div class="saas-card">
            <h3 style="color: #f8fafc; font-size: 1.15rem; font-weight: 700; margin-top: 0;">About the Platform</h3>
            <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6; margin: 0;">
                <strong>PulseGuard AI</strong> is a wearable biosignal analytics platform designed to analyze physiological signals and 
                identify patterns associated with stress using supervised machine learning. 
                By providing an intuitive, transparent interface for continuous physiological telemetry, PulseGuard AI empowers individuals 
                and researchers to understand their body's autonomic nervous system dynamics.
            </p>
        </div>
    """), unsafe_allow_html=True)

    st.markdown(textwrap.dedent("""
        <div class="saas-card">
            <h3 style="color: #f8fafc; font-size: 1.15rem; font-weight: 700; margin-top: 0;">Our Technology</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 12px;">
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;">
                    <div style="color: #38bdf8; font-weight: 700; margin-bottom: 4px;">📡 Physiological Signal Processing</div>
                    <div style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45;">Signal synchronization, 60-second windowing, and artifact rejection across multiple sensor channels.</div>
                </div>
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;">
                    <div style="color: #38bdf8; font-weight: 700; margin-bottom: 4px;">🫀 ECG & BVP Analysis</div>
                    <div style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45;">Continuous extraction of heart rate and Heart Rate Variability (HRV RMSSD) to monitor autonomic vagal modulation.</div>
                </div>
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;">
                    <div style="color: #06b6d4; font-weight: 700; margin-bottom: 4px;">⚡ Electrodermal Activity (EDA)</div>
                    <div style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45;">Tonic baseline and phasic sweat burst detection reflecting sympathetic fight-or-flight innervation.</div>
                </div>
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;">
                    <div style="color: #f59e0b; font-weight: 700; margin-bottom: 4px;">🌡️ Skin Temperature Analysis</div>
                    <div style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45;">Tracking thermal microvascular fluctuations driven by sympathetic vasoconstriction.</div>
                </div>
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;">
                    <div style="color: #a855f7; font-weight: 700; margin-bottom: 4px;">🫁 Respiration Dynamics</div>
                    <div style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45;">Quantifying breathing frequency and expansion depth to observe emotional and ventilatory shifts.</div>
                </div>
                <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;">
                    <div style="color: #10b981; font-weight: 700; margin-bottom: 4px;">🌲 Random Forest Classification</div>
                    <div style="color: #94a3b8; font-size: 0.84rem; line-height: 1.45;">Supervised ensemble learning with balanced class weighting and subject-specific baseline normalization.</div>
                </div>
            </div>
        </div>
    """), unsafe_allow_html=True)

    st.markdown(textwrap.dedent("""
        <div class="saas-card">
            <h3 style="color: #f8fafc; font-size: 1.15rem; font-weight: 700; margin-top: 0;">Dataset Attribution</h3>
            <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6;">
                This platform is powered by the benchmark <strong>WESAD (Wearable Stress and Affect Detection)</strong> dataset, 
                collected and curated by researchers at the University of Siegen, Germany:
            </p>
            <div style="background: #0b1329; border: 1px solid #1e293b; border-radius: 8px; padding: 14px 16px; margin: 12px 0; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #38bdf8;">
                Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Stockloew, and Kristof Van Laerhoven.<br>
                "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection."<br>
                ACM International Conference on Multimodal Interaction (ICMI '18), Boulder, CO, USA, 2018.
            </div>
            <p style="color: #cbd5e1; font-size: 0.88rem; margin: 0;">
                Official Dataset Repository: <a href="https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection" target="_blank" style="color: #06b6d4; text-decoration: underline;">UCI Machine Learning Repository (Dataset ID 465)</a>
            </p>
        </div>
    """), unsafe_allow_html=True)

    st.markdown(textwrap.dedent("""
        <div class="saas-card">
            <h3 style="color: #f8fafc; font-size: 1.15rem; font-weight: 700; margin-top: 0;">Our Purpose</h3>
            <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6; margin: 0;">
                Our purpose is to help users explore physiological signals and understand patterns associated with different experimental states. 
                By providing transparent data exploration and genuine machine learning inference, PulseGuard AI demonstrates how wearable computing 
                can bridge the gap between autonomic physiology and personal awareness.
            </p>
        </div>
    """), unsafe_allow_html=True)

    st.markdown(textwrap.dedent("""
        <div class="saas-card" style="border-left: 4px solid #f59e0b;">
            <h3 style="color: #f59e0b; font-size: 1.1rem; font-weight: 700; margin-top: 0;">Important Disclaimer</h3>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.55; margin: 0;">
                PulseGuard AI is an educational and research-oriented biosignal analytics platform. Its predictions are based on statistical 
                correlations observed in laboratory experiments and do not constitute medical diagnoses. This software should never replace 
                professional medical advice, clinical consultation, or emergency care.
            </p>
        </div>
    """), unsafe_allow_html=True)

    render_footer()
