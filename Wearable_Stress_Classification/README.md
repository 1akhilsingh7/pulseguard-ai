# 🫀 PULSEGUARD AI
> **"Understand Your Body. Monitor Your Stress."**  
> *Intelligent Wearable Biosignal Analytics Platform powered by Genuine WESAD Dataset Recordings*

[![Platform](https://img.shields.io/badge/Platform-Streamlit%20Cloud-06b6d4)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-WESAD%20Benchmark-10b981)](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
[![Engine](https://img.shields.io/badge/ML%20Engine-Random%20Forest%20Ensemble-38bdf8)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-10b981)](https://github.com/)

---

## 📌 1. Product Overview
**PulseGuard AI** is a modern healthcare technology SaaS web application designed to analyze physiological signals and discover patterns associated with stress using intelligent wearable data analysis.

By evaluating multi-modal biosignals recorded from continuous wearable sensors, PulseGuard AI translates raw cardiac, electrodermal, and respiratory telemetry into transparent, real-time stress intelligence.

---

## 🧭 2. Core Product Features (3 Simple Navigation Sections)

### 1. 🏠 HOME
- **Header:** *"Your Body Has a Story. Understand It."*
- **Action:** Quick-start button to **CHECK STRESS NOW**.
- **How It Works:** 3-step breakdown (Capture Biosignals → Intelligent Analysis → Instant Insights).
- **Supported Biosignals:** Electrocardiogram (ECG & HRV), Electrodermal Activity (EDA), Respiration Rhythm, and Skin Temperature.

### 2. 🫀 STRESS CHECK (Dual-Input Classification Engine)
- **Method 1: Sample Data**  
  Select from authentic WESAD participants (`S4` held-out test participant, `S2`, `S3`) and sample 60-second windows. Review real-time vitals and click **ANALYZE STRESS**.
- **Method 2: Upload Data**  
  Upload custom biosensor CSV files containing the 28 required multi-modal features. Includes an instant **Download Compatible CSV Template** button and automated schema validation.
- **Results Display:**
  - **Predicted State:** Green = `Baseline`, Red = `Stress`, Blue = `Amusement`.
  - **Visual Indicator & Probabilities:** Confidence percentage with ensemble probability distribution.
  - **Physiological Explanation:** Human-readable insights into autonomic nervous system balance.

### 3. 📑 MY RESULTS
- **Session History:** Tracks all analyses performed during the current browsing session.
- **Audit Table:** Chronological record of timestamps, data sources, predicted states, confidence scores, and vital signs.
- **Data Export:** One-click download of results in **CSV** format or formatted **Clinical Assessment Dossier (TXT)**.
- **History Management:** Reset and clear session history with a single click.

---

## 🫀 3. Data Authenticity & Machine Learning Pipeline
- **Authentic Benchmark:** Evaluated on genuine physiological recordings from the **WESAD** (Wearable Stress and Affect Detection) study (University of Siegen / ACM ICMI 2018).
- **Three Original Affect Classes:**
  - `0 = Baseline` (Homeostatic equilibrium, robust vagal parasympathetic modulation)
  - `1 = Stress` (Acute sympathetic nervous system fight-or-flight response)
  - `2 = Amusement` (Positive emotional valence and arousal)
- **Subject-Independent Validation:** Zero participant window leakage! Trained on subjects `S2` and `S3` (806 windows) and evaluated on unseen subject `S4` (408 windows).
- **Priority Target — Stress Recall:** 73.1% Stress Recall and 87.9% Stress Precision via balanced Random Forest bagging.
- **Subject-Specific Baseline Normalization:** Features are standardized against personal resting baselines ($z = \frac{x - \mu_{base}}{\sigma_{base}}$) before classification.

---

## 🚀 4. How to Run Locally

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Verification Test Suite
```bash
python test_system.py
```
*Output: All 7 verification tests pass with 100% success.*

### Step 3: Launch Web Application
```bash
streamlit run app.py
```
Or double-click [run_app.bat](file:///c:/Users/Akhil/Desktop/supervised%20project/run_app.bat) to launch in your default browser.

---

## 🌐 5. Streamlit Community Cloud Deployment Guide

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
3. Click **New app**, select your repository, branch `main`, and main file path `app.py`.
4. Click **Deploy!**

---

## 📄 6. Disclaimer
PulseGuard AI provides intelligent physiological insights based on the open-access WESAD scientific dataset. Not certified as a standalone medical diagnostic device.
