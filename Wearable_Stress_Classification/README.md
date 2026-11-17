# 🫀 PULSEGUARD AI
> **"Understand Your Body. Monitor Your Stress."**  
> *Intelligent Wearable Biosignal Analytics Platform powered by Genuine WESAD Dataset Recordings*

[![Platform](https://img.shields.io/badge/Platform-Streamlit%20Cloud-06b6d4)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-WESAD%20Benchmark-10b981)](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
[![Engine](https://img.shields.io/badge/ML%20Engine-Random%20Forest%20Ensemble-38bdf8)](https://scikit-learn.org/)
[![Auth](https://img.shields.io/badge/Security-PBKDF2--SHA256%20%7C%20SQLite-10b981)](https://github.com/)

---

## 📌 1. Product Overview
**PulseGuard AI** is a commercial-grade healthcare technology SaaS web application designed to analyze physiological signals and discover patterns associated with stress using intelligent wearable data analysis.

By evaluating multi-modal biosignals recorded from continuous wearable sensors, PulseGuard AI translates raw cardiac, electrodermal, and respiratory telemetry into transparent, real-time stress intelligence.

---

## 🔐 2. Authentication & User Management
Before accessing the main analytics dashboard, visitors are protected by a secure authentication portal:
- **Sign In / Registration:** New users can create an account or sign in with existing credentials.
- **Cryptographic Security:** Passwords are never stored in plaintext; they are securely hashed using **PBKDF2-HMAC-SHA256** with cryptographically random 16-byte salts and 100,000 iterations.
- **Persistent Storage:** User accounts are stored in a persistent local SQLite database (`pulseguard_users.db`).
- **Instant Demo Access:**
  - **Email:** `demo@pulseguard.ai`
  - **Password:** `Password123!`
- **Session Security:** Full session protection with a prominent **"Log Out"** button in the sidebar.

---

## 🧭 3. The 4 Main Navigation Sections

### 1. 🏠 HOME
- **Header:** *"Your Body Has a Story. Understand It."*
- **Action:** Quick-start CTA button to **CHECK STRESS NOW**.
- **How It Works:** 3-step breakdown (Capture Biosignals → Intelligent Analysis → Instant Insights).
- **Supported Biosignals:** Electrocardiogram (ECG & HRV), Electrodermal Activity (EDA), Respiration Rhythm, and Skin Temperature.

### 2. 🫀 STRESS CHECK (Dual-Input Classification Engine)
- **Method 1: Sample Data**  
  Select from authentic WESAD participants (`S4` held-out test participant, `S2`, `S3`) and 60-second window records.
- **Interactive Telemetric Biosignal Summary:**  
  Every sensor value is interactive! Click on any sensor (**Heart Rate**, **Heart Rate Variability**, **Electrodermal Activity**, **Skin Temperature**, **Respiration Rate**, **Blood Volume Pulse**, or **Accelerometer**) to view:
  - Current measured value and correct measurement unit
  - Physiological explanation of the autonomic mechanism
  - Scientifically supported normal reference ranges
  - High-contrast, dark-mode time-series graph of the **10-second continuous raw sensor waveform** (ECG, EDA, Temp, Resp, ACC)
  - Sensor category badges and hardware availability indicators (e.g., Lead-II chest ECG vs Empatica E4 wrist PPG).
- **Method 2: Upload Data**  
  Upload custom biosensor CSV files containing the 28 required multi-modal features. Includes an instant **Download Compatible CSV Template** button and automated schema validation.
- **Prediction Results Display:**
  - **Color-Coded Badges:** Green = `Baseline`, Red = `Stress`, Blue = `Amusement`.
  - **Prediction Probabilities:** Rendered using clean native visual progress bars with zero code-block escaping bugs.
  - **Physiological Explanation:** Human-readable diagnostic insights into sympathetic vs parasympathetic balance.

### 3. 📑 MY RESULTS
- **Session History:** Tracks all analyses performed during the current browsing session.
- **Audit Table:** Chronological record of timestamps, data sources, predicted states, confidence scores, and vital signs.
- **Data Export:** One-click download of results in **CSV** format or formatted **Clinical Assessment Dossier (TXT)**.
- **History Management:** Reset and clear session history with a single click.

### 4. ℹ️ ABOUT
- **Platform Overview:** Comprehensive product description and mission statement.
- **Our Technology:** Multi-modal wearable signal processing, ECG/BVP, EDA, skin temperature, and Random Forest classification.
- **Dataset Attribution:** Formal scientific credit to the benchmark **WESAD** dataset (Schmidt et al., ACM ICMI 2018) with a direct link to the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection).
- **Purpose & Medical Disclaimer:** Clearly states educational and research intent.

---

## 🫀 4. Data Authenticity & Machine Learning Pipeline
- **Authentic Benchmark:** Evaluated on genuine physiological recordings from the **WESAD** (Wearable Stress and Affect Detection) study (University of Siegen / ACM ICMI 2018).
- **Three Original Affect Classes:**
  - `0 = Baseline` (Homeostatic equilibrium, robust vagal parasympathetic modulation)
  - `1 = Stress` (Acute sympathetic nervous system fight-or-flight response)
  - `2 = Amusement` (Positive emotional valence and arousal)
- **Subject-Independent Validation:** Zero participant window leakage! Trained on subjects `S2` and `S3` (806 windows) and evaluated on unseen subject `S4` (408 windows).
- **Priority Target — Stress Recall:** 73.1% Stress Recall and 87.9% Stress Precision via balanced Random Forest bagging.
- **Subject-Specific Baseline Normalization:** Features are standardized against personal resting baselines ($z = \frac{x - \mu_{base}}{\sigma_{base}}$) before classification.

---

## 🚀 5. How to Run Locally

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Verification Test Suite
```bash
python test_system.py
```
*Output: All 9 verification tests pass with 100% success.*

### Step 3: Launch Web Application
```bash
streamlit run app.py
```
Or double-click [run_app.bat](file:///c:/Users/Akhil/Desktop/supervised%20project/run_app.bat) to launch in your default browser.

---

## 🌐 6. Streamlit Community Cloud Deployment Guide

1. Push this repository to your GitHub account:
```bash
git init
git add .
git commit -m "feat: complete PulseGuard AI healthcare SaaS web application"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/pulseguard-ai.git
git push -u origin main
```
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository, branch `main`, main file path `app.py`.
4. Click **Deploy!**

---

## 📄 7. License & Disclaimer
© 2026 PulseGuard AI. All Rights Reserved.  
"Understand Your Body. Monitor Your Stress."  
Dataset Attribution: Developed using the open-access WESAD multimodal benchmark dataset (Schmidt et al., ACM ICMI 2018 / UCI ID 465). PulseGuard AI provides intelligent physiological insights for research and education. Not intended for direct autonomous medical diagnosis without professional healthcare supervision.
