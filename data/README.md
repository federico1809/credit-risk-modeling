# Data Documentation

## Dataset Overview

**Name:** Lending Club Loan Data  
**Source:** [Kaggle - Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)  
**Period:** 2007 - Q4 2018  
**Size:** ~2.2 million loans  
**File Format:** CSV (compressed as .gz)

---

## Description

This dataset contains complete loan data for all loans issued through Lending Club, a peer-to-peer lending platform. Each row represents a single loan with information about the borrower, loan characteristics, and repayment status.

Lending Club connects borrowers seeking personal loans with investors looking to earn returns. The platform assigns grades (A-G) and subgrades to loans based on credit risk, with Grade A being the least risky.

---

## Target Variable

**Column:** `loan_status`

**Possible Values:**
- `Fully Paid` - Loan was paid back in full ✅
- `Charged Off` - Loan defaulted, unlikely to be recovered ❌
- `Current` - Loan is being paid, outcome unknown 🔄
- `Default` - Loan is in default ❌
- `Late (31-120 days)` - Payment is overdue 🔄
- `Late (16-30 days)` - Payment is overdue 🔄
- `In Grace Period` - Payment is slightly overdue 🔄
- `Does not meet the credit policy. Status:Charged Off` - Old policy, defaulted ❌
- `Does not meet the credit policy. Status:Fully Paid` - Old policy, paid ✅

**Binary Classification:**
- **Positive Class (1):** Default = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off']
- **Negative Class (0):** Paid = ['Fully Paid']
- **Exclude:** ['Current', 'Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period', 'Issued']

---

## Key Features

### Borrower Demographics
- `annual_inc` - Self-reported annual income
- `emp_length` - Employment length in years (0-10+)
- `home_ownership` - Home ownership status (RENT, OWN, MORTGAGE, OTHER)
- `addr_state` - US state of residence
- `zip_code` - First 3 digits of zip code

### Credit History
- `fico_range_low` / `fico_range_high` - FICO credit score range
- `dti` - Debt-to-income ratio
- `delinq_2yrs` - Number of delinquencies in past 2 years
- `earliest_cr_line` - Date of earliest credit line
- `inq_last_6mths` - Number of credit inquiries in last 6 months
- `mths_since_last_delinq` - Months since last delinquency
- `open_acc` - Number of open credit accounts
- `pub_rec` - Number of derogatory public records
- `revol_bal` - Total revolving balance
- `revol_util` - Revolving line utilization rate (% of credit used)
- `total_acc` - Total number of credit accounts

### Loan Characteristics
- `loan_amnt` - Requested loan amount
- `funded_amnt` - Amount funded by investors (⚠️ leakage risk)
- `term` - Loan term (36 or 60 months)
- `int_rate` - Interest rate (%)
- `installment` - Monthly payment amount
- `grade` - LC assigned loan grade (A-G)
- `sub_grade` - LC assigned loan subgrade (A1-G5)
- `purpose` - Loan purpose (debt_consolidation, credit_card, home_improvement, etc.)
- `issue_d` - Date loan was issued

### Payment History (⚠️ DO NOT USE - Data Leakage)
These features are only known AFTER loan outcome and should NOT be used for training:
- `total_pymnt` - Total payments received
- `total_rec_prncp` - Principal received
- `total_rec_int` - Interest received
- `recoveries` - Post-charge-off recovery
- `collection_recovery_fee` - Collection fees
- `last_pymnt_d` - Date of last payment
- `last_pymnt_amnt` - Amount of last payment

---

## Data Quality Issues

### Missing Values
- `emp_length`: ~7% missing - impute as 0 (unemployed)
- `mths_since_last_delinq`: ~50% missing - impute as 999 (never delinquent)
- `mths_since_last_record`: ~80% missing - impute as 999 (no records)
- `revol_util`: ~0.1% missing - impute with median
- `dti`: ~0.02% missing - impute with group median by grade

### Outliers
- `annual_inc`: Some values >$1M (0.1% of data) - cap at 99th percentile
- `dti`: Some values >100 - review for data entry errors
- `revol_util`: Some values >100% - investigate or cap

### Temporal Considerations
- **Economic cycle bias:** Data includes 2007-2008 financial crisis and recovery period
- **Policy changes:** Lending Club updated underwriting policies over time
- **Seasonality:** Loan issuance volume varies by month

---

## Download Instructions

### Using Kaggle API (Recommended)

1. **Install Kaggle API:**
   ```bash
   pip install kaggle
   ```

2. **Get API credentials:**
   - Go to https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - Move `kaggle.json` to `~/.kaggle/`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download dataset:**
   ```bash
   bash scripts/download_data.sh
   ```

### Manual Download
1. Go to https://www.kaggle.com/datasets/wordsforthewise/lending-club
2. Click "Download" button
3. Extract to `data/raw/`
4. Rename to `lending_club.csv`

---

## File Structure

```
data/
├── raw/
│   ├── lending_club.csv          # Original dataset (not tracked in Git)
│   └── .gitkeep
└── processed/
    ├── train_data.csv            # Training set after preprocessing
    ├── test_data.csv             # Test set after preprocessing
    ├── feature_names.txt         # List of features used
    └── .gitkeep
```

---

## Citation

If using this data, please cite:

```
Lending Club. (2019). Lending Club Loan Data. 
Retrieved from https://www.kaggle.com/datasets/wordsforthewise/lending-club
```

---

## Legal & Ethical Considerations

- **Privacy:** Data is anonymized (no names, SSN, exact addresses)
- **Bias:** May contain demographic biases in lending decisions
- **Fair Lending:** Model should be audited for disparate impact
- **Regulatory Compliance:** Must comply with FCRA, ECOA, GDPR if deployed

---

## Additional Resources

- [Lending Club Statistics](https://www.lendingclub.com/info/statistics.action)
- [Feature Definitions](https://resources.lendingclub.com/LCDataDictionary.xlsx)
- [Kaggle Discussion Forum](https://www.kaggle.com/datasets/wordsforthewise/lending-club/discussion)
