# IBM-EDUNET-Project


# 🤖 Binary Classification — Adult Income (UCI)

Predict whether a person earns **> \$50K/year** using the **UCI Adult (Census Income)** dataset.  
Built end-to-end with a clean ML pipeline: **data prep → modeling → evaluation → (optional) deployment**.  
**Acknowledgment:** Completed as part of the **AI/ML Internship — IBM SkillBuild by Edunet Foundation**.



## 📦 Repository

https://github.com/BhaskarMishra05/IBM-EDUNET-Project.git


---

## ❓ Problem Statement

Classify individuals into **`>50K`** or **`<=50K`** income brackets from demographic and work-related attributes.
Objective: achieve balanced **Precision** and **Recall** with solid generalization.

---

## 📂 Dataset

* **Source:** UCI Machine Learning Repository — Adult (Census Income)
* **Link:** [https://archive.ics.uci.edu/dataset/2/adult](https://archive.ics.uci.edu/dataset/2/adult)
* **Rows:** \~48,842
* **Features:** 14 (mix of categorical & numeric)
* **Target:** `income` (`>50K` / `<=50K`)

---

## ⚙️ Setup

### 1) Clone

```bash
git clone https://github.com/BhaskarMishra05/IBM-EDUNET-Project.git
cd IBM-EDUNET-Project
```

### 2) Create & activate a Python venv (recommended)

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
python -V
pip -V
```

**Windows (PowerShell)**

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -V
pip -V
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run

Training (example):

```bash
python -m src.start
```

App (if included):

```bash
python app.py
# open http://127.0.0.1:5000/
```

---

## 🧰 Libraries

* `pandas`, `numpy` — data wrangling
* `scikit-learn`,`LightGBM` — models, preprocessing, metrics
* `matplotlib`, `seaborn` — EDA & plots
* `joblib` — model persistence
* `Flask` — simple deployment (optional)

---

## 🛠️ Approach

**Data Cleaning**

* Handle missing entries such as `'?'` in categorical columns.
* Strip whitespace, standardize categories.

**Feature Engineering**

* One-hot encode categorical features.
* Scale/standardize numeric features where beneficial.
* Drop or reassess `fnlwgt` (often weak signal and confusing).

**Modeling**

* Establish baselines (e.g., Logistic Regression, Decision Tree).
* Tune best candidate(s) and finalize.

**Evaluation**

* Report **Accuracy, Precision, Recall, F1**, and **Confusion Matrix** on a held-out test set.
* Prefer balanced Precision/Recall; avoid overfitting.

---

## 📊 Results

**From latest run (timestamps in IST)**

```
Train Accuracy : 89%
Test  Accuracy : 88%
Baseline Accuracy : 0.8804
Baseline Precision: 0.8763
Baseline Recall   : 0.8804
Baseline F1-score : 0.8769
```

| Metric             | Value  |
| ------------------ | ------ |
| **Train Accuracy** | 0.8943 |
| **Test Accuracy**  | 0.8803 |
| **Precision**      | 0.8763 |
| **Recall**         | 0.8804 |
| **F1 Score**       | 0.8769 |

**Confusion Matrix**

```
[[8836  518]
 [ 943 1914]]
```

---

## 🔍 Findings

* **\~88%** test accuracy with balanced precision/recall ⇒ good generalization.
* Most errors cluster near the income threshold boundary (expected).
* Encodings + basic modeling already deliver strong baselines; room for calibrated thresholds, better feature interactions, and advanced ensembles.

---

## ⚖️ Fairness & Bias (Important)

The Adult dataset includes sensitive attributes (e.g., `sex`, `race`) and legacy category names. Use care:

* Evaluate metrics by subgroup (e.g., demographic parity, equal opportunity).
* Consider **calibrated probabilities** and **threshold tuning** per operational constraints.
* Document and communicate limitations transparently.

---

## 📁 Project Structure (typical)

```text
IBM-EDUNET-Project/
├─ data/                 # raw / processed data (gitignored)
├─ notebooks/            # EDA / experiments
├─ src/
│  ├─ data/              # loaders, preprocessors
│  ├─ models/            # training, inference
│  ├─ utils/             # helpers, metrics, logging
│  └─ start.py           # pipeline entry point
├─ app.py                # Flask app (optional)
├─ templates.py  
├─ requirements.txt
└─ README.md
```

---

## 🧪 Repro Tips

* Fix seeds for numpy/sklearn where applicable.
* Log configs and hyperparameters; save trained artifacts with `joblib`.
* Keep train/test split constant or use stratified CV for comparison.

---

## 👤 Author
**Bhaskar Mishra**  
📧 bhaskarmishra1590@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/bhaskar-mishra-026848221/)


---

## 🙏 Acknowledgment

This project was developed as part of the AI/ML Internship under IBM SkillBuild, supported by the Edunet Foundation.
Special thanks to the mentors and resources provided during the internship program.

