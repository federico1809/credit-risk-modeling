# Contributing Guide

## Development Workflow

This document explains how to work on this project, the difference between notebooks and scripts, and how the codebase is organized.

---

## 🏗️ Project Structure Philosophy

This project follows a **dual-track approach**:

1. **Notebooks** (`notebooks/`) - For exploration, experimentation, and documentation
2. **Scripts** (`src/`) - For production-ready, modular, testable code

---

## 📊 Notebooks vs Scripts: When to Use Each

### Jupyter Notebooks (.ipynb)

**Purpose:** Exploration, analysis, experimentation, and communication

**Use notebooks for:**
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature engineering experimentation
- ✅ Model prototyping and comparison
- ✅ Generating visualizations for reports
- ✅ Documenting your thought process
- ✅ Showing results to stakeholders

**Characteristics:**
- Interactive execution (cell by cell)
- Inline visualizations
- Mix of code, markdown, and outputs
- Great for storytelling
- Saved outputs for documentation

**Limitations:**
- Not easily executable as a pipeline
- Harder to version control (JSON format)
- Not modular or reusable
- Not suitable for production deployment

---

### Python Scripts (.py)

**Purpose:** Production code, automation, modularity, and reusability

**Use scripts for:**
- ✅ Production pipelines
- ✅ Modular, reusable functions
- ✅ Code that needs to be tested
- ✅ CLI tools
- ✅ Code that will be imported by other modules
- ✅ Deployment-ready components

**Characteristics:**
- Executable from command line
- Easy to version control (plain text)
- Modular and testable
- Can be parameterized via CLI arguments
- Suitable for CI/CD pipelines

---

## 🔄 Development Workflow

### Phase 1: Exploration (Current)

**Location:** `notebooks/`

**Process:**
1. Start with a notebook for exploration
2. Experiment with different approaches
3. Visualize results interactively
4. Document findings with markdown
5. Keep the notebook as documentation

**Current notebooks:**
- `01_eda_exploration.ipynb` - Data exploration and insights
- `02_feature_engineering.ipynb` - Feature creation and selection
- `03_model_training.ipynb` - Model development and comparison (upcoming)
- `04_model_evaluation.ipynb` - Performance analysis and interpretation (upcoming)

---

### Phase 2: Production (Future - Optional)

**Location:** `src/`

**Process:**
1. Extract working logic from notebooks
2. Refactor into modular functions/classes
3. Add docstrings and type hints
4. Write unit tests
5. Create CLI scripts
6. Integrate into pipeline

**Conversion example:**

**From notebook:**
```python
# In 02_feature_engineering.ipynb
df['credit_util_rate'] = df['revol_util'] / 100
df['payment_to_income'] = (df['installment'] * 12) / (df['annual_inc'] + 1)
```

**To script:**
```python
# src/data/feature_engineer.py

class FeatureEngineer:
    """Create domain-driven features for credit risk modeling."""
    
    def create_credit_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create credit utilization rate feature.
        
        Args:
            df: Input DataFrame with revol_util column
            
        Returns:
            DataFrame with credit_util_rate feature
        """
        df = df.copy()
        df['credit_util_rate'] = df['revol_util'] / 100
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = self.create_credit_utilization(df)
        df = self.create_payment_to_income(df)
        # ... more transformations
        return df
```

---

## 📁 Directory Structure Explained

### `notebooks/`
**Purpose:** Interactive analysis and experimentation

**Contents:**
- EDA notebooks with visualizations
- Feature engineering experiments
- Model training comparisons
- Evaluation and interpretation

**Guidelines:**
- One notebook per major task
- Name with numerical prefix (01_, 02_)
- Include markdown cells explaining reasoning
- Clear outputs (don't commit with huge outputs)
- Executable from top to bottom (Run All should work)

---

### `src/`
**Purpose:** Production-ready, modular code

**Structure:**
```
src/
├── data/
│   ├── data_loader.py      # Load raw data
│   ├── data_cleaner.py     # Clean and preprocess
│   └── feature_engineer.py # Create features
├── models/
│   ├── train.py            # Model training logic
│   ├── predict.py          # Prediction logic
│   └── evaluate.py         # Evaluation metrics
├── utils/
│   ├── config.py           # Configuration management
│   ├── logger.py           # Logging setup
│   └── visualization.py    # Plotting utilities
└── pipeline.py             # End-to-end pipeline orchestration
```

**Guidelines:**
- One module per logical component
- Clear function/class names
- Comprehensive docstrings
- Type hints for function signatures
- Importable by other modules

---

### `tests/`
**Purpose:** Unit tests for src/ modules

**Structure:**
```
tests/
├── test_data_loader.py
├── test_feature_engineer.py
└── test_models.py
```

**Guidelines:**
- Mirror src/ structure
- Test critical functions
- Use pytest fixtures for setup
- Aim for >70% code coverage

---

### `scripts/`
**Purpose:** CLI scripts for automation

**Contents:**
- `download_data.sh` - Data acquisition
- `train_model.py` - Model training from CLI
- `evaluate_model.py` - Model evaluation from CLI

**Usage:**
```bash
python scripts/train_model.py --config config/config.yaml
```

---

### `config/`
**Purpose:** Configuration files

**Contents:**
- `config.yaml` - Main project configuration
- `feature_config.yaml` - Feature engineering parameters

**Benefits:**
- Separate config from code
- Easy to change parameters
- Version controlled
- Environment-specific configs

---

### `data/`
**Purpose:** Data storage (NOT tracked in Git)

**Structure:**
```
data/
├── raw/            # Original, immutable data
├── processed/      # Cleaned, feature-engineered data
└── README.md       # Data documentation
```

**Important:**
- Large data files are in .gitignore
- Only metadata/small samples in Git
- Download instructions in data/README.md

---

## 🧪 Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_feature_engineer.py -v
```

### Writing Tests

```python
# tests/test_feature_engineer.py

import pytest
import pandas as pd
from src.data.feature_engineer import create_credit_utilization

def test_create_credit_utilization():
    """Test credit utilization calculation."""
    # Arrange
    df = pd.DataFrame({'revol_util': [50.0, 75.0, 100.0]})
    
    # Act
    result = create_credit_utilization(df)
    
    # Assert
    assert 'credit_util_rate' in result.columns
    assert result['credit_util_rate'].iloc[0] == 0.5
```

---

## 🎨 Code Style

### Python Style Guide

This project follows **PEP 8** with these tools:

- **black** - Code formatting (line length: 100)
- **flake8** - Linting
- **isort** - Import sorting
- **mypy** - Type checking (optional)

### Formatting Code

```bash
# Format all code
black src/ tests/

# Sort imports
isort src/ tests/

# Check linting
flake8 src/ tests/
```

### Pre-commit Checks

Before committing:
```bash
# Run all checks
black --check src/ tests/
flake8 src/ tests/
pytest tests/
```

---

## 📝 Docstring Format

Use **Google Style** docstrings:

```python
def calculate_default_probability(
    loan_amnt: float,
    int_rate: float,
    dti: float
) -> float:
    """
    Calculate probability of loan default.
    
    Uses a trained model to predict default probability based on
    loan characteristics.
    
    Args:
        loan_amnt: Loan amount in dollars
        int_rate: Interest rate as percentage (e.g., 12.5 for 12.5%)
        dti: Debt-to-income ratio as percentage
    
    Returns:
        Probability of default between 0 and 1
    
    Raises:
        ValueError: If inputs are negative or out of valid range
    
    Example:
        >>> calculate_default_probability(10000, 12.5, 25.0)
        0.23
    """
    # Implementation here
    pass
```

---

## 🔄 Git Workflow

### Branch Strategy

For this project (solo development):
- Work directly on `main` branch
- Create feature branches for major changes (optional)

For team projects:
```
main              # Production-ready code
├── develop       # Active development
    ├── feature/eda
    ├── feature/modeling
    └── feature/deployment
```

### Commit Messages

Use descriptive commit messages:

**Good:**
```
✓ Add feature engineering notebook with credit risk indicators
✓ Implement XGBoost model with hyperparameter tuning
✓ Fix data leakage by removing payment history features
✓ Update README with model performance results
```

**Bad:**
```
✗ update
✗ changes
✗ fix
✗ asdf
```

### Commit Frequency

Commit after completing a logical unit of work:
- ✅ Completed notebook section
- ✅ Implemented new feature
- ✅ Fixed a bug
- ✅ Updated documentation

---

## 📊 Working with Notebooks

### Best Practices

1. **Clear all outputs before committing:**
   ```python
   # In Jupyter: Kernel → Restart & Clear Output
   ```

2. **Make notebooks executable top-to-bottom:**
   - Run All should work without errors
   - No manual steps required

3. **Use meaningful cell separators:**
   ```markdown
   ## Section Name
   Brief description of what this section does
   ```

4. **Keep cells focused:**
   - One logical task per cell
   - Not too long (max ~50 lines)

5. **Save key outputs:**
   ```python
   # Save important figures
   plt.savefig('../reports/figures/feature_correlation.png')
   
   # Save processed data
   df.to_csv('../data/processed/features.csv', index=False)
   ```

---

## 🚀 Running the Pipeline

### Option 1: Notebooks (Current)

Execute notebooks in order:
```bash
# 1. EDA
jupyter notebook notebooks/01_eda_exploration.ipynb

# 2. Feature Engineering
jupyter notebook notebooks/02_feature_engineering.ipynb

# 3. Model Training (upcoming)
jupyter notebook notebooks/03_model_training.ipynb
```

### Option 2: Scripts (Future)

Run as automated pipeline:
```bash
# Full pipeline
python src/pipeline.py --config config/config.yaml

# Individual steps
python scripts/train_model.py --data data/processed/train_data.csv
python scripts/evaluate_model.py --model models/xgboost_model.pkl
```

---

## 🐛 Debugging

### In Notebooks

Use these patterns:
```python
# Inspect DataFrames
df.head()
df.info()
df.describe()

# Check for issues
df.isnull().sum()
df.dtypes

# Visualize distributions
df['column'].hist()
```

### In Scripts

Use logging instead of print:
```python
import logging

logger = logging.getLogger(__name__)
logger.info(f"Processing {len(df)} records")
logger.warning("Missing values detected")
logger.error("Model training failed")
```

---

## 📚 Documentation

### What to Document

1. **README.md** - Project overview, setup, usage
2. **PROJECT_SUMMARY.md** - Detailed context and methodology
3. **CONTRIBUTING.md** - This file (development guide)
4. **data/README.md** - Data sources and descriptions
5. **Docstrings** - In all functions and classes
6. **Markdown cells** - In notebooks explaining reasoning

### What NOT to Include

- ❌ Implementation details in README (keep it high-level)
- ❌ Code snippets in documentation (link to source instead)
- ❌ Outdated information (update or remove)

---

## ✅ Checklist for New Features

Before considering a feature complete:

- [ ] Code works as expected
- [ ] Docstrings added
- [ ] Tests written (if production code)
- [ ] Notebook outputs cleared (if notebook)
- [ ] Code formatted (black, isort)
- [ ] No linting errors (flake8)
- [ ] Documentation updated
- [ ] Committed with descriptive message

---

## 🎓 Learning Resources

### For Notebooks
- [Jupyter Best Practices](https://jupyter-notebook.readthedocs.io/en/stable/)
- [Gallery of Interesting Notebooks](https://github.com/jupyter/jupyter/wiki)

### For Production Code
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [The Hitchhiker's Guide to Python](https://docs.python-guide.org/)

### For ML Projects
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)

---

## 🤝 Questions?

If you're working on this project and have questions:

1. Check existing documentation (this file, README, PROJECT_SUMMARY)
2. Look at existing code for patterns
3. Review commit history for context

---

**Last Updated:** February 2026  
**Maintainer:** Federico Ceballos Torres  
**Contact:** federico.ct@gmail.com
