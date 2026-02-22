# Setup Verification Checklist

Use this checklist to ensure your development environment is properly configured.

---

## ✅ Initial Setup Verification

### Directory Structure
- [ ] All folders created: `data/`, `src/`, `notebooks/`, `models/`, `reports/`, `config/`, `scripts/`, `tests/`, `.github/`
- [ ] `.github/workflows/` folder exists (hidden folder)
- [ ] All `__init__.py` files created in src modules
- [ ] `.gitkeep` files in empty directories

### Configuration Files
- [ ] `requirements.txt` - Python dependencies
- [ ] `.gitignore` - Git ignore rules
- [ ] `config/config.yaml` - Project configuration
- [ ] `setup.py` - Package installation
- [ ] `LICENSE` - MIT License
- [ ] `pytest.ini` - Test configuration
- [ ] `.flake8` - Linting configuration
- [ ] `pyproject.toml` - Black and tool configuration
- [ ] `.env.example` - Environment variables template

### Docker Files
- [ ] `Dockerfile` - Container definition
- [ ] `docker-compose.yml` - Multi-container setup

### Scripts
- [ ] `scripts/download_data.sh` - Data download script
- [ ] Script has execute permissions (`chmod +x`)

### Documentation
- [ ] `README.md` - Main project documentation (English)
- [ ] `PROJECT_ARCHITECTURE.md` - Architecture design
- [ ] `BEST_PRACTICES.md` - Technical guidelines
- [ ] `QUICKSTART.md` - Quick start guide
- [ ] `data/README.md` - Dataset documentation

### GitHub Actions
- [ ] `.github/workflows/ci.yml` - CI/CD pipeline

---

## 🐍 Python Environment Verification

### Virtual Environment
```bash
# Check if virtual environment is activated
which python
# Should show: /path/to/project/venv/bin/python

# Check Python version
python --version
# Should be: Python 3.9.x or higher
```

### Dependencies Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Verify key packages
python -c "import pandas; print(pandas.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
python -c "import xgboost; print(xgboost.__version__)"
python -c "import shap; print(shap.__version__)"
```

### Package Installation
```bash
# Install project as editable package
pip install -e .

# Verify installation
python -c "import src; print('✓ Package installed')"
```

---

## 📦 Kaggle API Verification

### API Setup
```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/kaggle.json

# Check permissions (should be 600)
stat -c %a ~/.kaggle/kaggle.json

# Test API connection
kaggle datasets list --page-size 5
# Should show list of datasets
```

### Data Download Test
```bash
# Run download script
bash scripts/download_data.sh

# Check if data was downloaded
ls -lh data/raw/
# Should show lending_club.csv (large file)
```

---

## 🐳 Docker Verification (Optional)

### Docker Installation
```bash
# Check Docker version
docker --version
docker-compose --version

# Test Docker
docker run hello-world
```

### Build Image
```bash
# Build Docker image
docker build -t credit-risk-model .

# Check if image was created
docker images | grep credit-risk-model
```

### Run Container
```bash
# Run container
docker run -it credit-risk-model python --version

# Test imports inside container
docker run -it credit-risk-model python -c "import pandas; print('✓ Pandas works')"
```

---

## 🧪 Testing Verification

### Run Tests
```bash
# Run pytest (will pass even with no tests yet)
pytest tests/ -v

# Check pytest configuration
pytest --version
```

### Code Quality Tools
```bash
# Test black
black --check src/ tests/
# Should show: "All done! ✨ 🍰 ✨"

# Test flake8
flake8 src/ tests/
# Should show: no output (no errors)

# Test isort
isort --check-only src/ tests/
```

---

## 📓 Jupyter Verification

### Start Jupyter
```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### Create Test Notebook
1. Create a new notebook in `notebooks/`
2. Run this cell:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

print("✓ All imports successful!")
```

---

## 🔧 Git Verification

### Git Configuration
```bash
# Check Git version
git --version

# Verify Git is initialized
git status
# Should show untracked files
```

### Initial Commit
```bash
# Add all files
git add .

# Check what will be committed
git status

# Verify .gitignore is working (data/raw/ should NOT appear)
git status | grep data/raw
# Should show nothing

# Make initial commit
git commit -m "Initial project setup"
```

---

## 🚀 Ready to Start Checklist

Before beginning Phase 1 (EDA), ensure:

- [ ] ✅ Python environment activated
- [ ] ✅ All dependencies installed
- [ ] ✅ Kaggle API configured
- [ ] ✅ Data downloaded (or ready to download)
- [ ] ✅ Jupyter Lab working
- [ ] ✅ Git initialized and .gitignore working
- [ ] ✅ Project structure verified
- [ ] ✅ Tests can run (even if empty)
- [ ] ✅ Code quality tools working

---

## ⚠️ Common Issues and Solutions

### Issue 1: "ModuleNotFoundError: No module named 'src'"
**Solution:**
```bash
pip install -e .
# Or in notebooks:
import sys
sys.path.append('/path/to/credit-risk-modeling')
```

### Issue 2: "Permission denied: ~/.kaggle/kaggle.json"
**Solution:**
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Issue 3: "git status shows data files"
**Solution:**
Verify `.gitignore` is in project root:
```bash
cat .gitignore | grep "data/raw"
```

### Issue 4: "black command not found"
**Solution:**
```bash
pip install black
# Or reinstall all dependencies
pip install -r requirements.txt
```

### Issue 5: "Docker build fails"
**Solution:**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t credit-risk-model .
```

---

## 📊 Next Steps

Once all checks pass, you're ready to begin development:

1. **Phase 1:** Open `notebooks/01_eda_exploration.ipynb` (you'll create this)
2. **Download data:** `bash scripts/download_data.sh`
3. **Start exploring:** Load the dataset and begin EDA

---

## 🆘 Getting Help

If you encounter issues not covered here:

1. Check `QUICKSTART.md` for detailed instructions
2. Review `BEST_PRACTICES.md` for technical guidance
3. Check `PROJECT_ARCHITECTURE.md` for design decisions
4. Open an issue on GitHub
5. Review logs in `logs/` directory (once created)

---

**Status:** Setup Complete! ✅

Ready to build an amazing ML project! 🚀
