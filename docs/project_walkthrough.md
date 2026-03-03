# Water Potability Prediction — Complete End-to-End Flow

## 1. Project Overview

This project predicts whether water is **safe to drink (potable)** based on 9 water quality parameters. It uses a **Machine Learning** pipeline built with **MLOps** practices (DVC for pipeline management and version control).

**Dataset**: 3,276 water samples with 9 chemical/physical features and a binary target (Potable vs Not Potable).

---

## 2. Columns Used and Why

All 9 features are standard **water quality indicators** used by the WHO and water treatment authorities:

| #          | Column              | Description                            | Why It Matters                                                                        |
| ---------- | ------------------- | -------------------------------------- | ------------------------------------------------------------------------------------- |
| 1          | **ph**              | Acidity/basicity of water (0-14 scale) | WHO recommends pH 6.5-8.5 for potable water. Extreme pH indicates contamination.      |
| 2          | **Hardness**        | Calcium and magnesium content (mg/L)   | High hardness can cause scale buildup. Affects taste and usability.                   |
| 3          | **Solids**          | Total dissolved solids (TDS) in ppm    | High TDS means more dissolved minerals. WHO limit is 500 ppm for drinking water.      |
| 4          | **Chloramines**     | Chlorine + ammonia compound (ppm)      | Used as disinfectant in water. Safe up to 4 ppm according to EPA.                     |
| 5          | **Sulfate**         | Sulfate ion concentration (mg/L)       | High sulfate causes laxative effects. WHO suggests limit of 250 mg/L.                 |
| 6          | **Conductivity**    | Electrical conductivity (μS/cm)        | Indicates ion concentration. Higher conductivity = more dissolved ions.               |
| 7          | **Organic_carbon**  | Total organic carbon (ppm)             | Measures organic pollutants. High values indicate contamination from organic sources. |
| 8          | **Trihalomethanes** | THM concentration (μg/L)               | Byproducts of chlorine disinfection. Potentially carcinogenic above 80 μg/L.          |
| 9          | **Turbidity**       | Cloudiness/haziness of water (NTU)     | Caused by suspended particles. WHO limit for drinking water is 5 NTU.                 |
| **Target** | **Potability**      | 0 = Not Potable, 1 = Potable           | Binary classification target — this is what we predict.                               |

### Important Notes:

- **No Label Encoding / One-Hot Encoding was needed** because ALL features are already numerical (float64 type)
- **No categorical columns exist** in this dataset — so no encoding techniques were required
- The target column `Potability` is already binary (0 and 1), so no label encoding was needed for it either

---

## 3. End-to-End Flow — What Was Done Step by Step

### Step 1: Project Setup (Cookiecutter Data Science Template)

- Created project structure using **Cookiecutter Data Science** template
- This gives us organized folders: `src/data/`, `src/model/`, `src/visualization/`, `data/`, `models/`, `reports/`
- Initialized **Git** for version control
- Initialized **DVC** for data and model versioning

### Step 2: Data Collection (`src/data/data_collection.py`) — Created FIRST

```
File: src/data/data_collection.py
```

**What it does:**

1. Loads the raw CSV data from `src/data/water_potability.csv`
2. Reads parameters (test_size) from `params.yaml`
3. Splits data into **train (80%)** and **test (20%)** using `train_test_split`
4. Saves train.csv and test.csv to `data/raw/`

**Key Decision**: Used `random_state=42` for reproducible splits.

### Step 3: Data Preprocessing (`src/data/data_prep.py`) — Created SECOND

```
File: src/data/data_prep.py
```

**What it does:**

1. Loads train and test data from `data/raw/`
2. **Fills missing values using MEAN imputation** — some columns (ph, Sulfate, Trihalomethanes) had null values
3. Saves cleaned data to `data/processed/`

**Feature Engineering / Data Prep Details:**

- **Imputation Strategy**: Mean imputation was used (filling NaN with column mean)
- **No Encoding**: All features are already numerical — no label encoding, one-hot encoding, or ordinal encoding was needed
- **No Scaling in Preprocessing**: Scaling is only applied during model comparison for distance-based models (SVM, LR)
- **No Feature Selection**: All 9 features were kept since they are all relevant water quality parameters

### Step 4: Model Building (`src/model/model_building.py`) — Created THIRD

```
File: src/model/model_building.py
```

**What it does:**

1. Loads processed training data
2. Separates features (X) and target (y = Potability)
3. Trains a **Random Forest Classifier** with `n_estimators=100`
4. Saves the trained model as `models/model.pkl` using pickle

### Step 5: Model Evaluation (`src/model/model_eval.py`) — Created FOURTH

```
File: src/model/model_eval.py
```

**What it does:**

1. Loads the saved model and test data
2. Makes predictions on test data
3. Calculates 4 metrics: **Accuracy, Precision, Recall, F1-Score**
4. Saves metrics to `reports/metrics.json`

### Step 6: Model Comparison (`src/model/model_comparison.py`) — Created FIFTH

```
File: src/model/model_comparison.py
```

**What it does:**

1. Trains and compares **3 different models**:
   - **Random Forest Classifier** (tree-based, no scaling needed)
   - **Support Vector Machine / SVM** (distance-based, needs scaling)
   - **Logistic Regression** (gradient-based, needs scaling)
2. Uses `StandardScaler` for SVM and LR (because they are distance-based)
3. Evaluates all 3 models on the same test data
4. Generates a comparison bar chart
5. Saves results to `reports/model_comparison.json`

### Step 7: Visualization (`src/visualization/visualize.py`) — Created SIXTH

```
File: src/visualization/visualize.py
```

**What it does:**

1. **Correlation Heatmap** — relationships between all features
2. **Feature Distribution Plots** — histograms colored by potability class
3. **Box Plots** — outlier detection by class
4. **Pairplot** — scatter plots of top correlated features
5. **Missing Values Chart** — shows which columns had nulls
6. **Target Distribution** — shows class imbalance

---

## 4. Model Comparison Results — Why Random Forest?

| Model               | Accuracy   | Precision  | Recall     | F1-Score   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| **Random Forest**   | 0.6723     | 0.6014     | **0.3525** | **0.4444** |
| SVM                 | **0.6951** | **0.6964** | 0.3197     | 0.4382     |
| Logistic Regression | 0.628      | 0.0        | 0.0        | 0.0        |

### Why Random Forest is the Best Choice:

1. **Best F1-Score (0.4444)** — F1 is the most important metric for imbalanced datasets because it balances precision and recall
2. **Highest Recall (0.3525)** — In water potability, recall matters more because we want to correctly identify potable water (missing a potable sample is costly)
3. **No Feature Scaling Needed** — Random Forest is tree-based and doesn't require feature normalization, making it more robust
4. **Handles Non-Linear Relationships** — RF can capture complex interactions between water features
5. **Robust to Outliers** — Water quality data often has outliers; RF handles them well
6. **SVM has marginally better accuracy** but lower recall, meaning it misses more potable water samples
7. **Logistic Regression completely fails** — it predicts everything as "Not Potable" (0 precision/recall), showing the data relationships are non-linear

---

## 5. Where MLOps is Used (DVC — Data Version Control)

### What is DVC?

DVC (Data Version Control) is an MLOps tool that manages:

- Data file versioning (like Git, but for large files)
- ML pipeline definition and reproducibility
- Experiment tracking through parameters and metrics

### Exactly Where DVC is Used in This Project:

| File                   | MLOps Purpose                                                                                                                                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dvc.yaml`             | **Pipeline Definition** — Defines the 6 stages of the ML pipeline (data_collection → preprocessing → model_building → model_eval → model_comparison → visualization) with their dependencies, parameters, and outputs |
| `dvc.lock`             | **Pipeline Lock File** — Stores MD5 hashes of every input/output/parameter so DVC knows what has changed and what needs re-running                                                                                    |
| `params.yaml`          | **Parameter Management** — Stores hyperparameters (test_size=0.20, n_estimators=100) centrally. Change a param, run `dvc repro`, and only affected stages re-run                                                      |
| `.dvc/`                | **DVC Configuration** — Internal DVC settings and cache                                                                                                                                                               |
| `.dvcignore`           | **DVC Ignore** — Tells DVC which files to skip (like .gitignore)                                                                                                                                                      |
| `reports/metrics.json` | **Metrics Tracking** — DVC tracks this as a metrics file, enabling comparison across experiments with `dvc metrics show` and `dvc metrics diff`                                                                       |

### DVC Pipeline Flow:

```
data_collection → pre_preprocessing → model_building → model_eval
                                  ↓
                           model_comparison

         visualization (independent — uses raw data)
```

### Key DVC Commands:

```bash
dvc repro              # Reproduce the entire pipeline
dvc metrics show       # Show current metrics
dvc metrics diff       # Compare metrics across commits
dvc params diff        # See parameter changes
dvc dag                # Visualize pipeline DAG
```

---

## 6. Data Preparation Summary

| Step              | What Was Done                                                       | Tool/Technique Used              |
| ----------------- | ------------------------------------------------------------------- | -------------------------------- |
| Missing Values    | Filled NaN with column mean                                         | `df[column].mean()` + `fillna()` |
| Train-Test Split  | 80% train, 20% test                                                 | `sklearn.train_test_split`       |
| Feature Scaling   | Not done in preprocessing (only during model comparison for SVM/LR) | `StandardScaler`                 |
| Label Encoding    | **NOT needed** — target is already binary (0/1)                     | N/A                              |
| One-Hot Encoding  | **NOT needed** — all features are numerical                         | N/A                              |
| Feature Selection | **NOT done** — all 9 features are relevant water quality indicators | N/A                              |
| Outlier Removal   | **NOT done** — kept all data points                                 | N/A                              |

---

## 7. Project File Structure

```
water-potability-prediction/
├── params.yaml                        ← MLOps: Central hyperparameter config
├── dvc.yaml                           ← MLOps: Pipeline definition (6 stages)
├── dvc.lock                           ← MLOps: Pipeline lock file (MD5 hashes)
│
├── src/
│   ├── data/
│   │   ├── water_potability.csv       ← Original dataset (3276 × 10)
│   │   ├── data_collection.py         ← Step 1: Load CSV, split 80/20
│   │   └── data_prep.py              ← Step 2: Fill missing values (mean)
│   │
│   ├── model/
│   │   ├── model_building.py          ← Step 3: Train Random Forest
│   │   ├── model_eval.py             ← Step 4: Evaluate on test set
│   │   └── model_comparison.py       ← Step 5: Compare RF vs SVM vs LR
│   │
│   └── visualization/
│       └── visualize.py              ← Step 6: Generate all plots
│
├── data/
│   ├── raw/                           ← train.csv, test.csv (from Step 1)
│   └── processed/                     ← Cleaned data (from Step 2)
│
├── models/
│   └── model.pkl                      ← Saved Random Forest model
│
└── reports/
    ├── metrics.json                   ← Model evaluation metrics
    ├── model_comparison.json          ← 3-model comparison results
    └── figures/                       ← All visualization plots
        ├── correlation_heatmap.png
        ├── feature_distributions.png
        ├── boxplots.png
        ├── pairplot.png
        ├── missing_values.png
        ├── model_comparison.png
        └── target_distribution.png
```
