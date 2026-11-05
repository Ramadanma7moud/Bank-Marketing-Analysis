# 🏦 Bank Term Deposit Prediction

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

An end-to-end machine learning project that predicts whether a bank customer will subscribe to a **term deposit** after a marketing campaign. The project covers exploratory data analysis, leakage-safe preprocessing with Scikit-learn Pipelines, model comparison, cross-validation, hyperparameter tuning, final evaluation, business insights, and a **multi-page Streamlit application** for interactive predictions.

---

## 📑 Table of Contents

1. [Business Problem](#-business-problem)
2. [Project Objectives](#-project-objectives)
3. [Dataset Overview](#-dataset-overview)
4. [Exploratory Data Analysis](#-exploratory-data-analysis)
5. [Preprocessing and Pipelines](#-preprocessing-and-pipelines)
6. [Avoiding Data Leakage](#-avoiding-data-leakage)
7. [Modeling Approach](#-modeling-approach)
8. [Results](#-results)
9. [Business Insights and Recommendations](#-business-insights-and-recommendations)
10. [Streamlit Application](#-streamlit-application)
11. [Project Structure](#-project-structure)
12. [Installation and Usage](#-installation-and-usage)
13. [Technologies Used](#-technologies-used)
14. [Limitations and Future Work](#-limitations-and-future-work)
15. [Author](#-author)

---

## 💼 Business Problem

Banks run marketing campaigns to persuade customers to open term deposits. Contacting every customer is costly, and many calls do not convert. If the bank can estimate **which customers are most likely to subscribe**, it can prioritize outreach, reduce wasted effort, and make better use of its marketing budget.

This project frames the problem as a **binary classification task**: given a customer's profile and campaign history, predict whether they will subscribe (`yes`) or not (`no`).

## 🎯 Project Objectives

- Explore the customer data and identify patterns associated with term deposit subscriptions.
- Build a reproducible, leakage-safe preprocessing and modeling workflow using Scikit-learn Pipelines.
- Compare a baseline and several classifiers, then select and tune the best candidate.
- Evaluate the final model on unseen data with metrics that matter for the business (especially recall for subscribers).
- Translate findings into practical marketing recommendations.
- Deploy the trained model in an interactive Streamlit app alongside a data-insights dashboard and a model-performance page.

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| File | `bank.csv` |
| Records | 11,162 customers |
| Columns | 17 (16 predictors + 1 target) |
| Missing values | None |
| Duplicate rows | None |
| **Target variable** | `deposit` (`yes` / `no`) |

**Target distribution** — the classes are close to balanced:

| Class | Count | Share |
|---|---|---|
| `no` | 5,873 | 52.62% |
| `yes` | 5,289 | 47.38% |

**Features**

| Type | Columns |
|---|---|
| Numeric | `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous` |
| Categorical | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome` |

| Feature | Description (as used in the app) |
|---|---|
| `age` | Customer age |
| `job`, `marital`, `education` | Occupation, marital status, education level |
| `default` | Whether the customer has credit in default |
| `balance` | Account balance |
| `housing`, `loan` | Has a housing loan / personal loan |
| `contact` | Contact type (`cellular`, `telephone`, `unknown`) |
| `day`, `month` | Day and month of the last contact |
| `duration` | Duration of the last contact (**excluded from modeling**, see below) |
| `campaign` | Number of contacts during this campaign |
| `pdays` | Days since previous contact (`-1` indicates no previous contact in this dataset) |
| `previous` | Number of contacts before this campaign |
| `poutcome` | Outcome of the previous campaign (`unknown`, `failure`, `other`, `success`) |

---

## 🔎 Exploratory Data Analysis

The analysis is documented in [`Bank_Marketing.ipynb`](Bank_Marketing.ipynb) and includes:

- **Data quality checks:** `info()`, `describe()`, missing-value counts, and duplicate detection (no missing values or duplicates were found).
- **Target analysis:** class distribution count plot and percentages.
- **Univariate analysis:** histograms for all numeric features and count plots for every categorical feature.
- **Outlier inspection:** box plots for `age`, `balance`, `campaign`, and `previous`. Outliers were inspected visually; no rows were removed or capped.
- **Bivariate analysis:** each categorical feature plotted against `deposit`, a subscription-rate cross-tabulation by job, and box plots of `age`, `balance`, `campaign`, and `previous` split by deposit status.

---

## ⚙️ Preprocessing and Pipelines

Features and target:

- `X` = all columns except `deposit` and `duration`
- `y` = `deposit`
- **Split:** 80% train / 20% test (`test_size=0.2`, `random_state=42`) with **stratification** on the target. Resulting shapes: 8,929 training rows and 2,233 test rows, with near-identical class proportions in both sets.

Preprocessing is built from a `ColumnTransformer` that combines two sub-pipelines:

| Branch | Columns | Steps |
|---|---|---|
| Numeric | `age`, `balance`, `day`, `campaign`, `pdays`, `previous` | `SimpleImputer(strategy='median')` → `StandardScaler()` |
| Categorical | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome` | `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')` |

After encoding, the feature space has **50 columns**. The imputers act as a safeguard for future or production data (the training data had no missing values). `handle_unknown='ignore'` lets the pipeline handle unseen categories at prediction time without errors.

The final model is a single Scikit-learn `Pipeline`: `preprocessor` → `RandomForestClassifier`. Saving this one object means the Streamlit app applies exactly the same preprocessing used in training, directly on raw user input.

---

## 🛡️ Avoiding Data Leakage

Several steps in the workflow guard against leakage:

1. **Split before fitting.** The stratified train/test split happens before any transformer is fitted.
2. **Fit on training data only.** The preprocessor is fitted with `fit_transform` on the training set and only applied with `transform` to the test set.
3. **Pipelines inside cross-validation.** `cross_validate` and `RandomizedSearchCV` receive the **full pipeline** (preprocessor + model) and the raw training data, so imputation, scaling, and encoding are re-fitted inside each fold using only that fold's training portion.
4. **Test set held back for final evaluation.** Hyperparameters were selected by cross-validation on the training data; the tuned model was then evaluated once on the held-out test set.
5. **`duration` removed from the features.** Call duration is only known once a call has finished, so it cannot be used to decide who to call. Dropping it keeps the model realistic for pre-call targeting and avoids an overly optimistic score.
6. **Single saved artifact.** Preprocessing and model are stored together in `bank_deposit_model.pkl`, preventing a mismatch between training-time and inference-time transformations.

---

## 🤖 Modeling Approach

Four models were evaluated on the same preprocessed data:

| Model | Configuration |
|---|---|
| Dummy Classifier | `strategy='most_frequent'` (baseline) |
| Logistic Regression | `random_state=42` |
| Decision Tree | `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42` |

**Selection:** Random Forest achieved the highest accuracy and recall for subscribers, so it was taken forward.

**Cross-validation:** Stratified 5-fold (`shuffle=True`, `random_state=42`) on the training set, scored on accuracy and on recall, precision, and F1 for the `yes` class.

**Hyperparameter tuning:** `RandomizedSearchCV` with 20 sampled candidates × 5 folds (100 fits), optimizing and refitting on **recall for the `yes` class**, so the model misses fewer likely subscribers.

| Parameter | Search space |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | None, 10, 20, 30 |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | `'sqrt'`, 0.5 |

**Best parameters found:**

```python
{
    'n_estimators': 100,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 0.5,
    'max_depth': None
}
```

---

## 📈 Results

### Model comparison (hold-out test set, n = 2,233)

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) |
|---|---|---|---|---|
| Dummy Classifier (baseline) | 52.62% | – | – | – |
| Logistic Regression | 69.55% | 0.73 | 0.56 | 0.64 |
| Decision Tree | 63.64% | 0.62 | 0.60 | 0.61 |
| Random Forest (default) | 72.32% | 0.75 | 0.63 | 0.68 |

Logistic Regression was slightly better at identifying non-subscribers (recall for `no` of 0.82 vs. 0.81), but Random Forest was better at finding actual subscribers.

### Cross-validation (5-fold, training set)

| Metric | Random Forest (default) | Random Forest (tuned) |
|---|---|---|
| Accuracy | 0.724 (± 0.002) | 0.728 |
| Recall (Yes) | 0.645 (± 0.018) | 0.649 |
| Precision (Yes) | 0.740 (± 0.011) | 0.745 |
| F1 (Yes) | 0.689 (± 0.006) | 0.693 |

Fold-to-fold variation is small, which points to stable performance. Tuning produced small, consistent improvements across all four metrics.

### Final model: tuned Random Forest on the test set

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| No | 0.71 | 0.81 | 0.76 | 1,175 |
| Yes | 0.75 | 0.63 | 0.69 | 1,058 |
| **Accuracy** | | | **72.46%** | 2,233 |
| Macro avg | 0.73 | 0.72 | 0.72 | 2,233 |

### Confusion matrix

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 949 (TN) | 226 (FP) |
| **Actual Yes** | 389 (FN) | 669 (TP) |

- **669 true positives:** subscribers correctly identified.
- **389 false negatives:** subscribers the model missed, i.e. lost opportunities.
- **226 false positives:** customers flagged as likely subscribers who would not subscribe, i.e. wasted outreach.

---

## 💡 Business Insights and Recommendations

### Insights from the data

Subscription rates by customer group (computed from `bank.csv`; overall rate is 47.38%):

| Segment | Observation |
|---|---|
| Job | **Students** have the highest subscription rate (74.72%), followed by **retired** customers (66.32%). **Blue-collar** customers have the lowest (36.42%), i.e. the highest non-subscription rate at 63.58%. |
| Age group | The youngest (18–25: 71.1%) and oldest (65+: 80.4%) groups subscribe the most; middle-aged groups (36–55) are the lowest at roughly 41%. |
| Account balance | Subscription rate rises with balance: negative balance 34.3%, 0–1,000: 44.9%, above 10,000: 59.2%. |
| Education | Tertiary 54.1%, secondary 44.7%, primary 39.4%. |
| Marital status | Single 54.3%, divorced 48.1%, married 43.4%. |

### Insights from the model

- The tuned Random Forest reaches **72.46% accuracy** versus a 52.62% majority-class baseline, a clear improvement over guessing.
- Recall for subscribers is **63%**, and the model produced **389 false negatives**, which means a meaningful share of real subscribers would be missed.

### Practical recommendations

1. **Prioritize high-propensity segments.** Focused campaigns for students, retired customers, and customers aged 65+ target the groups with the highest observed subscription rates.
2. **Review the strategy for blue-collar customers** instead of applying the same marketing effort to every group.
3. **Use the model as a prioritization aid.** It can rank customers by predicted subscription probability to help allocate outreach, but it should not be the sole basis for large-scale campaign decisions yet.
4. **Focus future model work on recall for the `yes` class** to reduce missed opportunities, and weigh it against the cost of extra outreach caused by false positives.

---

## 🖥️ Streamlit Application

The app is a multi-page Streamlit application that loads the saved pipeline (`bank_deposit_model.pkl`) with `joblib`.

### 1. Prediction (`app.py`)

An interactive form that collects customer details in two columns and returns a prediction.

- **Personal information:** age, job, marital status, education, account balance, credit in default.
- **Banking and campaign information:** housing loan, personal loan, contact type, last contact day and month, number of contacts in this campaign, days since previous contact, number of previous contacts, previous campaign outcome.
- **Output:** a result card stating whether the customer is *Likely* or *Unlikely* to subscribe, the **subscription probability** (from `predict_proba`), and a progress bar showing that probability.

### 2. Data Insights (`Data_Insights.py`)

A dashboard built on `bank.csv`:

- Dataset overview: total customers, number of features, target variable, and an expandable preview of the data.
- Subscription distribution (counts and bar chart).
- Subscription rate (%) by **job**, **age group**, **education**, **marital status**, and **account balance group**, each with a bar chart and a statistics table.

### 3. Model Performance (`Model_Performance.py`)

A summary of the Random Forest's results on the unseen test set: accuracy, precision, recall, and F1 for the `yes` class, plus the confusion matrix. These figures are static values entered in the page and correspond to the final model's evaluation in the notebook.

---

## 📁 Project Structure

Streamlit discovers extra pages from a `pages/` directory next to the main script, so the recommended layout is:

```
bank-term-deposit-prediction/
├── app.py                      # Streamlit entry point: Prediction page
├── pages/
│   ├── Data_Insights.py        # Data Insights page
│   └── Model_Performance.py    # Model Performance page
├── Bank_Marketing.ipynb        # EDA, preprocessing, modeling, tuning, evaluation
├── bank.csv                    # Dataset (11,162 rows × 17 columns)
├── bank_deposit_model.pkl      # Trained pipeline (preprocessor + Random Forest)
├── requirements.txt            # Python dependencies (see below)
└── README.md
```

> Run Streamlit from the project root so `bank.csv` and `bank_deposit_model.pkl` are found by their relative paths.

---

## 🚀 Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/bank-term-deposit-prediction.git
cd bank-term-deposit-prediction
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn==1.9.0 joblib streamlit matplotlib seaborn jupyter
```

> **Important:** `bank_deposit_model.pkl` was saved with **scikit-learn 1.9.0**. Using the same version avoids unpickling warnings or incompatibilities. `matplotlib`, `seaborn`, and `jupyter` are only needed to run the notebook.

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app opens in your browser (by default at `http://localhost:8501`), with the Data Insights and Model Performance pages available from the sidebar.

### 5. Reproduce the analysis (optional)

```bash
jupyter notebook Bank_Marketing.ipynb
```

The notebook reads the dataset from a local Windows path. Before running it, change the `pd.read_csv(...)` path in the first cell to `"bank.csv"`. The final cells re-save the trained model as `bank_deposit_model.pkl`.

---

## 🧰 Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Data manipulation | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Streamlit native charts |
| Machine learning | Scikit-learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `DummyClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `StratifiedKFold`, `cross_validate`, `RandomizedSearchCV`) |
| Model persistence | Joblib |
| Application | Streamlit |
| Development | Jupyter Notebook |

---

## ⚠️ Limitations and Future Work

**Current limitations**

- Overall accuracy (~72%) and recall for subscribers (~63%) leave room for improvement; the model is best used as a decision-support tool.
- The initial comparison of the four models was reported on the hold-out test set; a stricter setup would compare all candidates with cross-validation on the training data and reserve the test set for the final model only.
- The Model Performance page shows hardcoded metric values rather than computing them live.
- Only the Random Forest was cross-validated and tuned; Logistic Regression and Decision Tree were evaluated with default settings.

**Possible improvements**

- Tune and cross-validate all candidate models for a fairer comparison.
- Try additional algorithms (for example gradient boosting) and class-weighting or threshold tuning to raise recall for subscribers.
- Add feature-importance analysis and explainability to the app.
- Add a `requirements.txt` and automated tests, and deploy the app (for example on Streamlit Community Cloud).

---

⭐ If you found this project useful, consider giving the repository a star!
