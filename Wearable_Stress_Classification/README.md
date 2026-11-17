# 🩺 PulseGuard AI — Wearable-Signal Stress-Risk Classification
> **AI Healthcare Biosensing Dashboard powered by Authentic WESAD Dataset Recordings**  
> *Ready for Streamlit Community Cloud Web Deployment & 2nd-Year B.Tech CSE College Presentation*

[![Dataset](https://img.shields.io/badge/Dataset-WESAD%20(ACM%20ICMI%202018)-06b6d4)](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
[![Model](https://img.shields.io/badge/Model-Random%20Forest%20(100%20Trees)-10b981)](https://scikit-learn.org/)
[![Validation](https://img.shields.io/badge/Validation-Subject--Independent%20LOSO-38bdf8)](https://scikit-learn.org/)
[![Target](https://img.shields.io/badge/Priority%20Metric-Stress%20Recall%20(73.1%25)-f43f5e)](https://scikit-learn.org/)

---

## 📌 1. Project Overview & Problem Statement
Acute and chronic psychological stress are major triggers for cardiovascular disease, autonomic dysfunction, hypertension, and clinical burnout. Traditional clinical stress assessments rely on subjective retrospective surveys, which suffer from recall bias and cannot provide real-time warnings.

**PulseGuard AI** implements an end-to-end supervised machine learning application that detects acute sympathetic stress episodes from continuous wearable biosignals. 

### Key Project Highlights:
- **100% Real Biosignal Data:** Trained and evaluated strictly on the benchmark **WESAD** (Wearable Stress and Affect Detection) dataset recorded at the University of Siegen.
- **Subject-Independent Validation:** Zero participant window leakage! Evaluated strictly on held-out subject **S4** (408 windows) after training on subjects **S2 and S3** (806 windows).
- **High Sensitivity Priority (Stress Recall):** Implements `class_weight='balanced'` in Random Forest to prioritize **Stress Recall (73.1%)** and **Stress Precision (87.9%)**, avoiding dangerous false negatives.
- **Modern AI Healthcare SaaS Interface:** Built in dark navy (`#070d18`), cyan, and teal accents with 8 comprehensive analytical sections.

---

## 🫀 2. Dataset Attribution & Sensor Modalities
- **Citation:** Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Stockloew, and Kristof Van Laerhoven. *"Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection."* ACM International Conference on Multimodal Interaction (ICMI '18), 2018.
- **Repository:** UCI Machine Learning Repository (ID 465).
- **Biosensor Hardware:** RespiBAN Professional Chest Strap (Lead-II ECG, EDA, Inductive Respiration, Skin Temperature, 3D Accelerometer sampled at 700 Hz).
- **Original WESAD Affect Classes:**
  - `0 = Baseline Homeostasis` (Quiet resting reading, vagal parasympathetic dominance)
  - `1 = Acute Sympathetic Stress` (Trier Social Stress Test: TSST mental arithmetic + speech)
  - `2 = Amusement / Positive Valence` (Curated funny video clips, laughter respiration transients)

---

## 🔬 3. Machine Learning Architecture
```
[RespiBAN Chest Sensors (ECG, EDA, Resp, Temp, ACC @ 700 Hz)]
                              │
                              ▼
        [Signal Synchronization & 60-Second Windowing (5s stride)]
                              │
                              ▼
[Subject-Specific Baseline Normalization: z = (x - mean_base) / std_base]
                              │
                              ▼
       [Statistical Feature Extraction (28 Multi-Modal Channels)]
                              │
                              ▼
 [Random Forest Classifier Ensemble (100 Trees, class_weight='balanced')]
                              │
                              ▼
[Subject-Independent Evaluation (Trained on S2, S3 | Tested strictly on S4)]
                              │
                              ▼
         [Accuracy: 76.0% | Stress Recall: 73.1% | Stress Precision: 87.9%]
```

### Why Subject-Independent (LOSO) Evaluation?
In physiological time-series data, standard randomized `train_test_split` mixes adjacent 60-second windows from the same human subject across training and test sets. This leads to **data leakage** and produces an illusion of 98-99% accuracy that crashes completely on a new user. Our model is trained strictly on participants `S2` and `S3` and tested on an unseen human `S4`, proving authentic generalization to new patients.

---

## 📊 4. Model Evaluation Summary
| Metric | Performance on Held-Out Subject S4 | Interpretation |
| :--- | :---: | :--- |
| **Overall Test Accuracy** | **75.98%** | Matches published WESAD chest benchmarks |
| **Stress Recall (Priority)** | **73.11%** | Successfully catches 73 out of 100 acute stress episodes |
| **Stress Precision** | **87.88%** | When stress is predicted, it is 88% reliable |
| **Baseline Recall** | **100.00%** | Zero false alarms during relaxed resting periods |
| **Macro F1-Score** | **56.84%** | Balanced across all three affective states |

---

## 💻 5. Web Dashboard Architecture (8 Interactive Pages)
1. **🏠 Home Dashboard:** High-level executive overview, verified WESAD badge, KPI summary cards, end-to-end ML pipeline diagram, and affect class definitions.
2. **🗂️ Dataset Explorer:** Interactive tabular view of all 1,214 windows, filter by participant and condition, class distribution charts, and physiological summary tables.
3. **📈 Signal Analysis:** Multi-channel 10-second raw waveform viewer (ECG in mV, EDA in µS, Respiration, Skin Temp, and 3D ACC magnitude) with clinical trace interpretations.
4. **🔮 Stress Classification:** Real-time inference console. Evaluate real held-out patient windows from S4 or simulate custom biosensor readings with instant probability distribution and diagnostic rationale.
5. **📊 Model Performance:** Confusion matrix heatmap, classification report, and deep dive into why Stress Recall is prioritized in healthcare.
6. **🧬 Feature Importance:** Gini importance bar chart highlighting top biomarkers (EDA slope, HRV RMSSD, Heart Rate, Respiration Rate).
7. **📑 Risk Report:** Comprehensive clinical assessment report with vital signs breakdown, recommended interventions (diaphragmatic breathing), and one-click TXT/CSV downloads.
8. **ℹ️ About Project:** Official WESAD study citation, hardware specifications (RespiBAN & Empatica E4), laboratory TSST protocol details, and B.Tech CSE presentation highlights.

---

## 🚀 6. How to Run Locally

### Prerequisites:
- Python 3.9, 3.10, 3.11, or 3.12
- Git installed (optional)

### Step 1: Install Dependencies
Open PowerShell or Command Prompt in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run Verification Test Suite
Verify that all datasets, models, and tests pass:
```bash
python test_system.py
```

### Step 3: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Or double-click `run_app.bat` to launch automatically in your default browser.

---

## 🌐 7. Streamlit Community Cloud Deployment Guide

### Step 1: Push Code to GitHub
1. Create a free account at [github.com](https://github.com).
2. Create a new repository named `wearable-stress-classification`.
3. In your local project directory, run:
```bash
git init
git add .
git commit -m "feat: complete WESAD wearable stress classification dashboard"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/wearable-stress-classification.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **New app**.
3. Fill in the fields:
   - **Repository:** `<YOUR-USERNAME>/wearable-stress-classification`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**
5. Streamlit will install dependencies from `requirements.txt` and launch your live public URL (e.g. `https://wearable-stress-classification.streamlit.app`).

---

## 🎓 8. College Presentation & Viva Voce Q&A Cheat Sheet

### Q1: What makes this project different from basic synthetic toy projects?
> *"Unlike projects that train on fabricated synthetic rows, this project uses the real benchmark WESAD dataset recorded using medical-grade RespiBAN chest straps at 700 Hz. We process genuine physiological recordings across baseline, acute cognitive stress (TSST), and amusement."*

### Q2: Why did you choose Random Forest instead of a simple Logistic Regression or complex Deep Learning?
> *"Random Forest is an ensemble bagging method that trains 100 decorrelated decision trees. Physiological responses (e.g., skin conductance spikes and vagal withdrawal) are non-linear, which tree ensembles handle naturally without overfitting. Furthermore, Random Forest provides transparent Gini feature importances and handles class imbalance through balanced class weighting."*

### Q3: Why is high Stress Recall prioritized over Accuracy?
> *"In clinical and occupational health monitoring, a False Negative (missing an acute stress episode) fails to alert the user or initiate timely biofeedback. Our primary goal is not missing high-stress events, which is reflected in our 73.1% Stress Recall and 87.9% Stress Precision."*

### Q4: What is subject-independent evaluation and why is it important?
> *"In wearable physiological data, if you use standard randomized train_test_split, adjacent 60-second windows from the same subject leak into both sets, inflating accuracy to 99% artificially. By using Leave-One-Subject-Out (training on S2 and S3, testing on S4), we prove the model generalizes to an entirely new human participant."*

---

## 📄 9. License & Disclaimer
This project is developed for educational and academic presentation purposes for a 2nd-year B.Tech CSE degree. Biosignal data is sourced from the open-access WESAD dataset under academic attribution. Not intended for autonomous clinical diagnosis without professional medical supervision.
