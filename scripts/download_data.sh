#!/bin/bash

# Script to download Lending Club dataset from Kaggle
# Usage: bash scripts/download_data.sh

set -e  # Exit on error

echo "=========================================="
echo "Lending Club Data Download Script"
echo "=========================================="

# Check if Kaggle API is configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle API credentials not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Move the downloaded kaggle.json to ~/.kaggle/"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data/raw

echo ""
echo "Downloading Lending Club dataset..."
echo "This may take several minutes (file is ~1GB)..."

# Download dataset using Kaggle API
kaggle datasets download -d wordsforthewise/lending-club -p data/raw/

echo ""
echo "Unzipping dataset..."
cd data/raw/
unzip -o lending-club.zip
rm lending-club.zip  # Remove zip file to save space

echo ""
echo "=========================================="
echo "Download complete!"
echo "Dataset location: data/raw/accepted_2007_to_2018Q4.csv.gz"
echo "=========================================="

# Optional: Extract gzipped file
if [ -f "accepted_2007_to_2018Q4.csv.gz" ]; then
    echo ""
    echo "Extracting gzipped CSV..."
    gunzip -f accepted_2007_to_2018Q4.csv.gz
    
    # Rename to simpler name
    mv accepted_2007_to_2018Q4.csv lending_club.csv
    
    echo "Final dataset: data/raw/lending_club.csv"
    echo ""
    
    # Show dataset size
    echo "Dataset size:"
    ls -lh lending_club.csv
fi

echo ""
echo "✓ Setup complete! You can now run the EDA notebook."
