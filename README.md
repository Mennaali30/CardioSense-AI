# 🫀 CardioSense AI — Heart Disease Prediction

> A portfolio-grade Streamlit web application that predicts heart disease risk
> using an end-to-end machine learning pipeline trained on the UCI Cleveland
> Heart Disease dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🤖 6 ML Models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN |
| 🏆 Auto-selection | Best model chosen by ROC-AUC via 5-fold cross-validation |
| 🔧 Hyperparameter Tuning | GridSearchCV on winning model |
| 📊 Full Analytics | ROC curves, confusion matrix, feature importances, model comparison |
| 🎨 Modern UI | Dark theme, gauge chart, risk badges, metric cards |
| 📋 Feature Glossary | Clinical explanation for all 13 input variables |

---
## 🚀 Live Demo

**Try the application here:**

https://cardiosense-ai-htcbewfaa3rtf64wmfv6df.streamlit.app/

## 🚀 Quick Start

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cardiosense-ai.git
cd cardiosense-ai
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
cardiosense-ai/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🏥 Dataset

**Source:** [UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

| Attribute | Value |
|-----------|-------|
| Samples   | 303   |
| Features  | 13    |
| Target    | Binary (0 = no disease, 1 = disease) |
| Origin    | Cleveland Clinic Foundation |

The dataset is downloaded automatically at runtime via the `ucimlrepo` package.
No manual download required.

---

## 🔬 Input Features

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | 1 = Male, 0 = Female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dL) |
| `fbs` | Fasting blood sugar > 120 mg/dL |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression (exercise vs. rest) |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (fluoroscopy) |
| `thal` | Thalassemia type |

---

## 🤖 ML Pipeline

```
Raw Data (UCI)
     │
     ▼
Data Cleaning (drop NaN, binarize target)
     │
     ▼
Stratified Train / Test Split  (80 / 20)
     │
     ▼
StandardScaler
     │
     ▼
Train 6 Classifiers  →  Evaluate (Accuracy, ROC-AUC, 5-fold CV)
     │
     ▼
Select Best Model (highest ROC-AUC)
     │
     ▼
GridSearchCV Hyperparameter Tuning
     │
     ▼
Final Tuned Model  →  Streamlit Inference
```

---

## 📸 App Tabs

| Tab | Content |
|-----|---------|
| 🔍 Prediction | Sidebar inputs → gauge chart → risk level → confidence bars |
| 📊 Model Analytics | Model comparison, ROC curves, confusion matrix, feature importances |
| 📋 Dataset | Raw data preview, statistics, correlation heatmap, class distribution |
| ℹ️ About | Project description, dataset info, feature glossary |

---

## ⚠️ Disclaimer

This application is developed for **educational and portfolio purposes only**.
It is **not** intended as a medical device or clinical decision support tool.
Always consult a qualified cardiologist for medical decisions.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*CardioSense AI — Built with ❤️ using Streamlit & Scikit-learn*
