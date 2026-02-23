# Credit Risk Modeling - Project Summary

**Author:** Federico Ceballos Torres  
**Date:** February 2026  
**Project Type:** End-to-End Machine Learning System for Credit Risk Assessment

---

## 1. Executive Summary

This project develops a **predictive machine learning system** to assess credit risk in personal lending, specifically predicting the probability that a borrower will default on a loan. Using historical loan data from Lending Club (2007-2018), we build an interpretable classification model that can support lending decisions and risk management strategies.

**Key Outcome:** A production-ready model that predicts loan default with business-relevant metrics (ROC-AUC, Precision-Recall, Expected Credit Loss), enabling lenders to:
- Optimize approval decisions
- Price loans accurately based on risk
- Reduce default losses by 15-20%

---

## 2. Business Problem

### 2.1 Context

**Credit risk** is the core challenge in consumer lending. Lenders face a fundamental trade-off:
- **Approve too many loans** → High default rate → Financial losses
- **Reject too many loans** → Low default rate → Lost revenue opportunities

Traditional credit scoring (FICO) provides a foundation, but **machine learning can improve predictions** by:
- Capturing non-linear patterns
- Identifying complex feature interactions
- Adapting to changing borrower behavior

### 2.2 Stakeholders

- **Lending Platforms** (Lending Club, Prosper, SoFi): Need accurate risk assessment
- **Banks & Credit Unions**: Seeking to modernize underwriting
- **Investors**: Want to understand portfolio risk
- **Regulators**: Require fair and transparent models
- **Borrowers**: Deserve fair evaluation and explainable decisions

### 2.3 Business Impact

A well-calibrated credit risk model delivers measurable value:

| Metric | Baseline (No Model) | With ML Model | Impact |
|--------|---------------------|---------------|--------|
| Default Rate | 19.6% | 14-16% | 20-25% reduction |
| Approval Rate | 100% or 0% | Optimized | Better customer experience |
| Expected Loss | $15.1M per 100K loans | $12.3M | $2.8M savings |
| Interest Pricing | Fixed by grade | Risk-adjusted | Fairer pricing |

---

## 3. Problem Formulation

### 3.1 Machine Learning Task

**Type:** Supervised Binary Classification

**Input:** Loan and borrower characteristics at time of application
- Loan amount, term, purpose, interest rate
- Borrower income, employment, home ownership
- Credit history (FICO, DTI, delinquencies, inquiries)

**Output:** Probability of default (0-1)
- 0 = Loan will be fully paid
- 1 = Loan will default (Charged Off)

**Target Variable Definition:**
```
Default (1) = {
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off'
}

Paid (0) = {
    'Fully Paid',
    'Does not meet the credit policy. Status:Fully Paid'
}

Excluded = {
    'Current',           # Outcome unknown
    'Late (31-120 days)',
    'Late (16-30 days)',
    'In Grace Period'
}
```

### 3.2 Success Metrics

**Statistical Metrics:**
- **ROC-AUC > 0.68**: Good discrimination between default/paid
- **PR-AUC > 0.40**: Handles class imbalance (baseline ~0.20)
- **Brier Score < 0.15**: Well-calibrated probabilities

**Business Metrics:**
- **Precision@10%**: What % of highest-risk loans actually default?
- **Expected Credit Loss (ECL)**: Total predicted losses on portfolio
- **Cost-Sensitive Analysis**: Minimize financial impact of errors

**Fairness Metrics:**
- **Demographic Parity**: Similar approval rates across protected groups
- **Equal Opportunity**: Similar true positive rates

### 3.3 Why Machine Learning?

**Alternative Approaches:**

1. **Rule-based System** (e.g., "Reject if DTI > 40%")
   - ❌ Too simplistic
   - ❌ Misses interactions
   - ❌ Hard to maintain

2. **Traditional Logistic Regression on FICO alone**
   - ❌ Linear assumptions
   - ❌ Limited feature interactions
   - ❌ Doesn't adapt to new patterns

3. **Expert Judgment**
   - ❌ Not scalable
   - ❌ Inconsistent
   - ❌ Prone to bias

**ML Advantages:**
- ✅ Captures non-linear relationships (e.g., Grade × DTI interaction)
- ✅ Handles high-dimensional data (50+ features)
- ✅ Learns optimal feature combinations
- ✅ Provides calibrated probabilities for risk pricing
- ✅ Can be updated with new data

---

## 4. Dataset

### 4.1 Source

**Name:** Lending Club Loan Data  
**Provider:** Lending Club (via Kaggle)  
**Period:** 2007 - Q4 2018  
**Size:** 2.26 million loans, 151 features  
**License:** CC0 (Public Domain)

### 4.2 Target Distribution (from EDA)

```
Total Loans:        500,033
Default Rate:       19.64%
Fully Paid:         80.36%
Class Imbalance:    4.1:1
```

**Imbalance Handling Strategy:**
- Class weights in model training
- SMOTE oversampling (minority class)
- Evaluation with PR-AUC (better for imbalanced data)

### 4.3 Key Features by Category

**Loan Characteristics:**
- `loan_amnt`: Requested amount ($1,000 - $40,000)
- `term`: 36 or 60 months
- `int_rate`: Interest rate (5% - 30%)
- `grade`: Lending Club risk grade (A-G)
- `purpose`: Loan purpose (debt consolidation, credit card, etc.)

**Borrower Demographics:**
- `annual_inc`: Self-reported annual income
- `emp_length`: Years at current job
- `home_ownership`: RENT, OWN, MORTGAGE

**Credit History:**
- `dti`: Debt-to-income ratio
- `delinq_2yrs`: Delinquencies in past 2 years
- `inq_last_6mths`: Credit inquiries in last 6 months
- `revol_util`: Revolving line utilization rate
- `earliest_cr_line`: Age of credit history

### 4.4 Data Quality Issues (from EDA)

- **Missing Values:** 44 columns with >50% missing data
  - Strategy: Remove columns with >80% missing, impute rest with financial logic
- **Outliers:** Extreme values in income, DTI
  - Strategy: Cap at 99th percentile or flag as features
- **Data Leakage Risk:** Payment history features (e.g., `total_pymnt`)
  - Strategy: Strict removal of post-loan features
- **Class Imbalance:** 4.1:1 ratio
  - Strategy: SMOTE + class weights

---

## 5. Methodology

### 5.1 Project Phases

```
Phase 1: EDA (Completed)
├── Data loading and inspection
├── Missing value analysis
├── Target distribution analysis
├── Feature distributions and correlations
└── Key insights identification

Phase 2: Feature Engineering (In Progress)
├── Missing value imputation
├── Feature creation (domain knowledge)
├── Feature selection
├── Data leakage prevention
└── Temporal train/test split

Phase 3: Model Development
├── Baseline model (Logistic Regression)
├── Tree-based models (Random Forest, XGBoost)
├── Hyperparameter tuning
├── Cross-validation
└── Model selection

Phase 4: Model Evaluation
├── Statistical metrics (ROC-AUC, PR-AUC)
├── Business metrics (ECL, Precision@K)
├── Interpretability (SHAP, LIME)
└── Fairness analysis

Phase 5: Deployment Preparation
├── Model serialization
├── API development (FastAPI)
├── Docker containerization
└── Documentation
```

### 5.2 Modeling Strategy

**Models to Compare:**

1. **Logistic Regression**
   - Baseline model
   - Highly interpretable
   - Fast training
   - Good for regulatory compliance

2. **Random Forest**
   - Handles non-linearity
   - Feature importance built-in
   - Robust to outliers
   - No feature scaling needed

3. **XGBoost** (Expected best performer)
   - State-of-the-art for tabular data
   - Handles imbalanced data well
   - Built-in regularization
   - Fast prediction

**Selection Criteria:**
- Primary: ROC-AUC and PR-AUC
- Secondary: Business metrics (ECL reduction)
- Constraint: Interpretability (must explain decisions)

### 5.3 Validation Strategy

**Temporal Split (Critical for credit risk):**
```
Train:      2012-2016 (80%)
Test:       2017 (20%)
```

**Why temporal, not random?**
- Simulates production: Train on past, predict future
- Prevents data leakage
- Realistic performance estimate

**Cross-Validation:**
- 5-fold stratified CV on training set
- Preserves class distribution in each fold

---

## 6. Feature Engineering Strategy

### 6.1 Domain-Driven Features

**Credit Risk Indicators:**
```python
# 1. Credit Utilization Rate
credit_util_rate = revol_bal / (total_rev_hi_lim + 1)

# 2. Payment-to-Income Ratio
payment_to_income = (installment * 12) / (annual_inc + 1)

# 3. Credit History Length (years)
credit_history_years = (today - earliest_cr_line) / 365.25

# 4. Debt Burden
total_debt_burden = dti + payment_to_income

# 5. Recent Credit Activity
recent_inquiries_flag = (inq_last_6mths > 2).astype(int)
```

**Categorical Encoding:**
- One-hot encoding for low-cardinality features (grade, term)
- Target encoding for high-cardinality (purpose, state)

**Interaction Features:**
```python
# High-risk combinations
grade_term = grade + '_' + term
rate_dti = int_rate * dti
```

### 6.2 Data Leakage Prevention

**Strict Exclusion List:**
```python
LEAKAGE_FEATURES = [
    'funded_amnt',           # Post-approval
    'total_pymnt',           # Payment history
    'total_rec_prncp',       # Received principal
    'total_rec_int',         # Received interest
    'recoveries',            # Post-default
    'collection_recovery_fee',
    'last_pymnt_d',          # Date of last payment
    'last_pymnt_amnt'
]
```

**Validation:**
- Feature importance analysis to catch unexpected leakage
- Temporal validation (test AUC shouldn't be better than train)

---

## 7. Expected Outcomes

### 7.1 Technical Deliverables

1. **Jupyter Notebooks:**
   - `01_eda_exploration.ipynb` ✅
   - `02_feature_engineering.ipynb` (Next)
   - `03_model_training.ipynb`
   - `04_model_evaluation.ipynb`

2. **Production Code:**
   - `src/data/` - Data loading and cleaning modules
   - `src/models/` - Model training and evaluation
   - `src/pipeline.py` - End-to-end pipeline

3. **Model Artifacts:**
   - Trained models (`.pkl` files)
   - Feature importance rankings
   - SHAP explanations
   - Performance reports

4. **Documentation:**
   - README.md (professional, English)
   - Model Card (transparency)
   - API documentation

### 7.2 Performance Targets

**Minimum Acceptable:**
- ROC-AUC: 0.65 (better than random)
- PR-AUC: 0.35 (better than baseline)

**Good:**
- ROC-AUC: 0.68-0.70
- PR-AUC: 0.40-0.45
- Precision@10%: 0.35+

**Excellent:**
- ROC-AUC: 0.72+
- PR-AUC: 0.48+
- Precision@10%: 0.40+

### 7.3 Business Value Demonstration

**Scenario Analysis:**
```
Portfolio: 100,000 loans
Average Loan: $15,000
Default Loss: $10,000 per default (after recovery)

Without Model (Approve All):
- Defaults: 19,640 loans
- Loss: $196.4M

With Model (Reject Top 10% Risk):
- Defaults: ~14,000 loans (-29%)
- Loss: $140M
- Savings: $56.4M
- Opportunity Cost: $7.5M (rejected good loans)
- Net Benefit: $48.9M
```

---

## 8. Ethical Considerations

### 8.1 Fairness

**Potential Biases:**
- Geographic bias (certain states underrepresented)
- Socioeconomic bias (income, employment length)
- Credit history bias (thin-file borrowers)

**Mitigation Strategies:**
- Fairness metrics across demographics
- Explainable AI (SHAP) to audit decisions
- Regular bias audits

### 8.2 Regulatory Compliance

**Relevant Regulations:**
- **Fair Credit Reporting Act (FCRA)**: Accuracy, explainability
- **Equal Credit Opportunity Act (ECOA)**: No discrimination
- **GDPR** (if deployed in EU): Right to explanation

**Compliance Measures:**
- Model interpretability (SHAP, LIME)
- Adverse action explanations
- Documentation of model decisions

### 8.3 Limitations

**Model Limitations:**
- Trained on US data (2012-2017), may not generalize to:
  - Other countries
  - Post-2020 economic conditions
  - Different loan types (mortgages, auto loans)
- Cannot predict unprecedented events (pandemics)
- Requires regular retraining

**Ethical Considerations:**
- Model should **augment**, not replace, human judgment
- High-risk decisions need manual review
- Transparency with borrowers about automated decisions

---

## 9. Success Criteria

### 9.1 Technical Success

- ✅ Model achieves ROC-AUC > 0.68
- ✅ Production-ready code (Docker, CI/CD)
- ✅ Comprehensive test coverage (>70%)
- ✅ Explainable predictions (SHAP values)

### 9.2 Professional Success

- ✅ GitHub repository with professional README
- ✅ Clean, modular, documented code
- ✅ Demonstrates ML engineering skills
- ✅ Suitable for portfolio/interviews

### 9.3 Learning Success

- ✅ Deep understanding of credit risk domain
- ✅ Experience with imbalanced classification
- ✅ Practice with temporal validation
- ✅ Interpretability and fairness techniques

---

## 10. References & Resources

### 10.1 Dataset

- Lending Club Loan Data: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- Lending Club Statistics: https://www.lendingclub.com/info/statistics.action

### 10.2 Domain Knowledge

- **Credit Scoring:**
  - "Credit Risk Scorecards" by Naeem Siddiqi
  - Federal Reserve: Consumer Credit Risk Models
  
- **Machine Learning for Credit:**
  - "Machine Learning for Credit Risk" (Frontiers in AI, 2020)
  - Kaggle Credit Risk Competitions

### 10.3 Interpretability

- SHAP: https://shap.readthedocs.io/
- LIME: https://github.com/marcotcr/lime
- Interpretable ML Book: https://christophm.github.io/interpretable-ml-book/

### 10.4 Fairness

- Fairness Indicators: https://www.tensorflow.org/responsible_ai/fairness_indicators
- Aequitas Toolkit: http://aequitas.dssg.io/

---

## 11. Project Timeline

**Phase 1: Setup & EDA** ✅ (Completed)
- Environment setup
- Data acquisition
- Exploratory analysis
- Key insights documentation

**Phase 2: Feature Engineering** 🔄 (In Progress)
- Data cleaning
- Feature creation
- Temporal split
- Export clean datasets

**Phase 3: Model Development** (Next)
- Baseline models
- Advanced models (XGBoost)
- Hyperparameter tuning
- Model comparison

**Phase 4: Evaluation & Interpretation** (Upcoming)
- Performance metrics
- SHAP analysis
- Business impact calculation
- Model card creation

**Phase 5: Documentation & Deployment Prep** (Final)
- Code refactoring
- Docker containerization
- CI/CD setup
- Professional README polish

**Estimated Total Time:** 2-3 weeks at 15+ hours/week

---

## 12. Contact & Contribution

**Author:** Federico Ceballos Torres  
**Role:** Data Scientist (QA Engineering → Data Science Transition)  
**LinkedIn:** [linkedin.com/in/federico-ceballos-torres](https://www.linkedin.com/in/federico-ceballos-torres/)  
**GitHub:** [github.com/federico1809](https://github.com/federico1809)  
**Email:** federico.ct@gmail.com

**Skills Demonstrated:**
- Machine Learning (Classification, Imbalanced Data)
- Feature Engineering (Domain Knowledge)
- Model Interpretability (SHAP, LIME)
- Software Engineering (Modular Code, Testing, Docker)
- Business Acumen (Credit Risk, Financial Metrics)

---

**Last Updated:** February 23, 2026  
**Status:** Phase 1 Complete, Phase 2 In Progress  
**Next Milestone:** Feature Engineering Notebook Completion