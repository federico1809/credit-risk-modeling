# Credit Risk Modeling - Best Practices & Technical Considerations

## 1. Data Handling Best Practices

### 1.1 Temporal Data Leakage Prevention
**Critical for credit risk modeling:**

```python
# ❌ WRONG: Random split can leak future information
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ✅ CORRECT: Time-based split
cutoff_date = df['issue_d'].quantile(0.8)
train_df = df[df['issue_d'] < cutoff_date]
test_df = df[df['issue_d'] >= cutoff_date]
```

**Why:** Loans issued in 2015 shouldn't be used to predict loans from 2014. In production, you only have historical data to predict future loans.

### 1.2 Target Variable Definition
Lending Club has multiple loan statuses. Define default carefully:

```python
# Default definition (typical in industry)
DEFAULT_STATUSES = [
    'Charged Off',           # Loan defaulted
    'Default',               # In default
    'Does not meet the credit policy. Status:Charged Off'
]

NON_DEFAULT_STATUSES = [
    'Fully Paid',            # Loan paid successfully
    'Current'                # Still being paid (exclude or handle separately)
]

# Exclude ambiguous statuses from training
EXCLUDE_STATUSES = [
    'Late (31-120 days)',    # In between, don't know final outcome yet
    'Late (16-30 days)',
    'In Grace Period'
]
```

### 1.3 Missing Data Strategy
Don't just impute with median/mode blindly:

```python
# Financial logic for missing values
missing_strategy = {
    'emp_length': 0,                    # Missing = unemployed (conservative)
    'mths_since_last_delinq': 999,     # Missing = never delinquent (good sign)
    'revol_util': df['revol_util'].median(),  # Standard imputation
    'dti': df.groupby('grade')['dti'].transform('median')  # Group-wise
}
```

---

## 2. Feature Engineering Domain Knowledge

### 2.1 Credit Risk Indicators

**Key features to engineer:**

```python
# 1. Credit Utilization Rate
df['credit_util_rate'] = df['revol_bal'] / (df['total_rev_hi_lim'] + 1)

# 2. Debt-to-Income Ratio (already in data, but validate)
df['dti_category'] = pd.cut(df['dti'], bins=[0, 10, 20, 30, 100], 
                             labels=['low', 'medium', 'high', 'very_high'])

# 3. Payment-to-Income Ratio
df['installment_pct_income'] = (df['installment'] * 12) / (df['annual_inc'] + 1)

# 4. Credit History Length
df['credit_history_years'] = (
    pd.to_datetime('today') - pd.to_datetime(df['earliest_cr_line'])
).dt.days / 365.25

# 5. Recent Credit Inquiries (risk indicator)
df['inq_last_6mths_high'] = (df['inq_last_6mths'] > 2).astype(int)

# 6. Employment Stability
df['emp_length_numeric'] = df['emp_length'].str.extract('(\d+)').astype(float)
df['stable_employment'] = (df['emp_length_numeric'] >= 5).astype(int)

# 7. Loan Purpose Risk
high_risk_purposes = ['small_business', 'renewable_energy', 'educational']
df['high_risk_purpose'] = df['purpose'].isin(high_risk_purposes).astype(int)
```

### 2.2 Interaction Features

```python
# Grade × Subgrade interactions (credit score proxy)
df['grade_subgrade'] = df['grade'] + df['sub_grade']

# Income × Loan Amount
df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)

# Interest Rate × DTI (high rate + high debt = danger)
df['rate_dti_interaction'] = df['int_rate'] * df['dti']
```

### 2.3 Features to Avoid (Data Leakage)

**⚠️ Don't use these features — they're only known AFTER loan outcome:**

```python
LEAKAGE_FEATURES = [
    'funded_amnt',           # Known only after funding decision
    'total_pymnt',           # Payment history = target leakage
    'total_rec_prncp',       # Amount recovered
    'recoveries',            # Post-default recovery
    'collection_recovery_fee',
    'last_pymnt_d',          # Date of last payment
    'last_pymnt_amnt',
    'next_pymnt_d'
]
```

---

## 3. Model Training Best Practices

### 3.1 Class Imbalance Handling

**Option A: Class Weights (recommended first)**
```python
from sklearn.linear_model import LogisticRegression

# Automatic class balancing
model = LogisticRegression(class_weight='balanced', random_state=42)

# Or manual calculation
class_weights = {
    0: len(y) / (2 * (y == 0).sum()),
    1: len(y) / (2 * (y == 1).sum())
}
model = LogisticRegression(class_weight=class_weights)
```

**Option B: SMOTE (you have experience with this)**
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Only apply SMOTE to training data
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Minority = 50% of majority

pipeline = ImbPipeline([
    ('smote', smote),
    ('model', LogisticRegression())
])

# ✅ CORRECT: SMOTE inside cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline, X_train, y_train, cv=5)

# ❌ WRONG: SMOTE before cross-validation (causes overfitting)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
```

**Option C: Undersampling (when data is large)**
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
```

### 3.2 Stratified Time-Series Split

```python
from sklearn.model_selection import TimeSeriesSplit

# Time-based CV for financial data
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train and validate
    model.fit(X_train_cv, y_train_cv)
    score = model.score(X_val_cv, y_val_cv)
```

### 3.3 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [50, 100, 200],
    'min_samples_leaf': [20, 50, 100],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',  # Or 'average_precision' for imbalanced data
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

---

## 4. Evaluation Metrics for Credit Risk

### 4.1 Confusion Matrix Interpretation

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Financial interpretation
TN, FP, FN, TP = cm.ravel()

# Type I Error (False Positive): Predict default, but loan is good
# Cost: Lost revenue from rejected good customer
# Example: Lose $500 profit per rejected good loan

# Type II Error (False Negative): Predict no default, but loan defaults
# Cost: Loss of principal + interest
# Example: Lose $10,000 per defaulted loan

# Calculate total cost
cost_fp = 500   # Lost opportunity
cost_fn = 10000  # Default loss

total_cost = (FP * cost_fp) + (FN * cost_fn)
```

### 4.2 Business Metrics

**Precision at K (Top K% riskiest loans)**
```python
def precision_at_k(y_true, y_proba, k=0.1):
    """
    Precision in top K% of predicted probabilities.
    Useful for deciding which loans to reject.
    """
    threshold_idx = int(len(y_proba) * k)
    top_k_idx = np.argsort(y_proba)[-threshold_idx:]
    
    return y_true[top_k_idx].mean()

# Example: What's precision in top 10% riskiest loans?
p_at_10 = precision_at_k(y_test, y_proba[:, 1], k=0.1)
print(f"If we reject top 10% risky loans, {p_at_10:.2%} would have defaulted")
```

**Expected Credit Loss (ECL)**
```python
def expected_credit_loss(y_proba, loan_amounts, lgd=0.6):
    """
    ECL = Probability of Default × Loss Given Default × Exposure
    
    lgd: Loss Given Default (typically 0.4-0.7 for unsecured loans)
    """
    ecl = y_proba[:, 1] * lgd * loan_amounts
    return ecl.sum()

# Example
test_ecl = expected_credit_loss(y_proba, X_test['loan_amnt'])
print(f"Expected loss on test portfolio: ${test_ecl:,.0f}")
```

**Profit Curve**
```python
def calculate_profit(y_true, y_proba, threshold, loan_amnt, int_rate):
    """
    Calculate profit under different approval thresholds.
    
    Profit = (Interest Earned on Good Loans) - (Losses on Defaulted Loans)
    """
    y_pred = (y_proba[:, 1] > threshold).astype(int)
    
    # Approved loans
    approved = (y_pred == 0)
    
    # Calculate profit
    good_loans = approved & (y_true == 0)
    bad_loans = approved & (y_true == 1)
    
    # Interest earned (assuming 3-year loan, simplified)
    profit_good = (loan_amnt[good_loans] * int_rate[good_loans] * 3).sum()
    
    # Loss from defaults (lose principal)
    loss_bad = loan_amnt[bad_loans].sum()
    
    total_profit = profit_good - loss_bad
    
    return total_profit, approved.sum()

# Find optimal threshold
thresholds = np.linspace(0.1, 0.9, 50)
profits = []

for thresh in thresholds:
    profit, n_approved = calculate_profit(
        y_test, y_proba, thresh, 
        X_test['loan_amnt'], X_test['int_rate']
    )
    profits.append({'threshold': thresh, 'profit': profit, 'n_approved': n_approved})

profit_df = pd.DataFrame(profits)
optimal_threshold = profit_df.loc[profit_df['profit'].idxmax(), 'threshold']
```

### 4.3 Model Calibration

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Check if predicted probabilities are reliable
prob_true, prob_pred = calibration_curve(y_test, y_proba[:, 1], n_bins=10)

# If poorly calibrated, apply Platt scaling
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)
y_proba_calibrated = calibrated_model.predict_proba(X_test)
```

---

## 5. Model Interpretability

### 5.1 SHAP for Credit Risk

```python
import shap

# Initialize explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Individual prediction explanation
shap.waterfall_plot(shap_values[0])

# Dependence plots (how feature affects prediction)
shap.dependence_plot("int_rate", shap_values, X_test)
```

### 5.2 Partial Dependence Plots

```python
from sklearn.inspection import PartialDependenceDisplay

features = ['int_rate', 'dti', 'annual_inc', 'credit_util_rate']

PartialDependenceDisplay.from_estimator(
    model, X_test, features, 
    kind='average'
)
```

---

## 6. Code Organization Patterns

### 6.1 Configuration Management

**config/config.yaml:**
```yaml
data:
  raw_path: "data/raw/lending_club.csv"
  processed_path: "data/processed/"
  train_test_split_date: "2015-01-01"
  
model:
  random_state: 42
  test_size: 0.2
  cv_folds: 5
  
class_balance:
  method: "smote"  # Options: smote, class_weight, undersampling
  sampling_strategy: 0.5
  
feature_engineering:
  drop_leakage_features: true
  create_interactions: true
  binning_strategy: "quantile"
```

### 6.2 Logging Pattern

```python
import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger with file and console handlers."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)
    
    return logger

# Usage
logger = setup_logger('credit_risk', 'logs/training.log')
logger.info("Starting model training...")
```

---

## 7. Testing Strategy

### 7.1 Data Validation Tests

```python
# tests/test_data_loader.py
import pytest

def test_no_future_leakage(train_df, test_df):
    """Ensure test data is chronologically after train data."""
    assert train_df['issue_d'].max() < test_df['issue_d'].min()

def test_no_leakage_features(feature_list):
    """Ensure no post-loan features are used."""
    leakage = ['total_pymnt', 'recoveries', 'collection_recovery_fee']
    assert not any(feat in feature_list for feat in leakage)

def test_target_distribution(y_train):
    """Check target class distribution is reasonable."""
    default_rate = y_train.mean()
    assert 0.05 < default_rate < 0.30, "Unexpected default rate"
```

---

## 8. Git & Version Control

### .gitignore for ML Projects

```
# Data
data/raw/
data/processed/*.csv
*.pkl
*.h5

# Models
models/*.pkl
models/*.joblib

# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Python
__pycache__/
*.py[cod]
*$py.class
.env

# IDE
.vscode/
.idea/

# Logs
logs/
*.log

# Reports
reports/*.pdf
reports/figures/*.png
```

---

## 9. Documentation Standards

### Model Card Template

Create `MODEL_CARD.md`:

```markdown
# Credit Risk Model Card

## Model Details
- **Model Type:** XGBoost Classifier
- **Version:** 1.0
- **Date:** 2024-02-22
- **Developer:** Federico Ceballos Torres

## Intended Use
- **Primary Use:** Predicting probability of default for personal loans
- **Out-of-Scope:** Business loans, mortgages, credit cards

## Training Data
- **Source:** Lending Club (2007-2015)
- **Size:** 887,379 loans
- **Default Rate:** 14.6%

## Performance
- **ROC-AUC:** 0.68
- **PR-AUC:** 0.42
- **Precision@10%:** 0.35

## Limitations
- Model trained on US data (2007-2015), may not generalize to current economy
- Underrepresents certain demographics
- Requires minimum credit history

## Ethical Considerations
- Risk of discriminatory bias in features like employment length
- Regular fairness audits recommended
```

---

Ready to start implementing Phase 1 (Data & EDA)?
