# Quick Start Guide

Get the Credit Risk Modeling project up and running in 5 minutes!

---

## Prerequisites

- Python 3.9 or higher
- Git
- (Optional) Docker
- Kaggle account (for data download)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/credit-risk-modeling.git
cd credit-risk-modeling
```

### 2. Set Up Python Environment

**Option A: Using Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project as package
pip install -e .
```

**Option B: Using Docker**
```bash
# Build image
docker-compose build

# Run container
docker-compose up -d

# Access container
docker exec -it credit-risk-modeling bash
```

### 3. Configure Kaggle API

```bash
# Create kaggle directory
mkdir -p ~/.kaggle

# Download API credentials:
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Move kaggle.json to ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Download Data

```bash
bash scripts/download_data.sh
```

This will download ~1GB of data from Kaggle. It may take 5-10 minutes depending on your connection.

---

## Project Structure Overview

```
credit-risk-modeling/
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code (modules)
├── data/                   # Data files (not tracked in Git)
├── models/                 # Saved models
├── reports/                # Generated reports and figures
├── config/                 # Configuration files
├── tests/                  # Unit tests
└── scripts/                # Utility scripts
```

---

## Development Workflow

### Phase 1: Exploratory Data Analysis

```bash
# Start Jupyter
jupyter lab

# Open notebook: notebooks/01_eda_exploration.ipynb
```

### Phase 2: Feature Engineering

```bash
# Open notebook: notebooks/02_feature_engineering.ipynb
```

### Phase 3: Model Training

```bash
# Option A: Use notebook
# Open: notebooks/03_model_training.ipynb

# Option B: Use CLI (once implemented)
python scripts/train_model.py --config config/config.yaml
```

### Phase 4: Model Evaluation

```bash
# Open notebook: notebooks/04_model_evaluation.ipynb

# Or use CLI
python scripts/evaluate_model.py --model models/xgboost_model.pkl
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

---

## Code Quality Checks

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/
```

---

## Git Workflow

### Initial Commit

```bash
# Initialize Git repository
git init

# Add files
git add .

# Commit
git commit -m "Initial project setup"

# Add remote (replace with your GitHub repo)
git remote add origin https://github.com/yourusername/credit-risk-modeling.git

# Push
git push -u origin main
```

### Working on Features

```bash
# Create feature branch
git checkout -b feature/eda-analysis

# Make changes, then commit
git add .
git commit -m "Complete EDA with visualizations"

# Push to remote
git push origin feature/eda-analysis
```

---

## Docker Usage

### Build and Run

```bash
# Build image
docker build -t credit-risk-model .

# Run container interactively
docker run -it -v $(pwd):/app credit-risk-model

# Run Jupyter in Docker
docker-compose --profile jupyter up
# Access at: http://localhost:8889
```

### Docker Compose Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild
docker-compose build --no-cache
```

---

## Troubleshooting

### Issue: Kaggle API not working

**Solution:**
```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/

# Verify permissions
chmod 600 ~/.kaggle/kaggle.json

# Test API
kaggle datasets list
```

### Issue: Import errors in notebooks

**Solution:**
```bash
# Install package in editable mode
pip install -e .

# Or add to notebook:
import sys
sys.path.append('/path/to/credit-risk-modeling')
```

### Issue: Out of memory during training

**Solution:**
- Reduce dataset size in config: `train_test_split_date: "2014-01-01"`
- Use smaller model: `n_estimators: 50`
- Close other applications

### Issue: Docker build fails

**Solution:**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker-compose build --no-cache
```

---

## Next Steps

1. ✅ Complete setup
2. 📊 Run EDA notebook (`01_eda_exploration.ipynb`)
3. 🔧 Engineer features (`02_feature_engineering.ipynb`)
4. 🤖 Train models (`03_model_training.ipynb`)
5. 📈 Evaluate and interpret (`04_model_evaluation.ipynb`)
6. 📝 Document findings in README
7. 🚀 Push to GitHub

---

## Useful Commands Reference

```bash
# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter lab

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Check Git status
git status

# Download data
bash scripts/download_data.sh

# Train model (once implemented)
python scripts/train_model.py

# Evaluate model (once implemented)
python scripts/evaluate_model.py
```

---

## Resources

- [Documentation](PROJECT_ARCHITECTURE.md)
- [Best Practices](BEST_PRACTICES.md)
- [Data Info](data/README.md)
- [Kaggle Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

---

Happy Coding! 🚀

If you run into issues, check the troubleshooting section or open an issue on GitHub.
