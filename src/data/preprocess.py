"""
Feature Engineering and Preprocessing Pipeline

Extracts feature engineering logic from Notebook 2.
Handles data loading, feature creation, and preprocessing for the credit risk model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for credit risk modeling.
    
    Applies the same transformations used in training:
    - Creates domain-knowledge features
    - Handles missing values
    - Returns features in the correct order
    """
    
    def __init__(self, feature_names: List[str] = None):
        """
        Initialize the feature engineer.
        
        Args:
            feature_names: List of feature names expected by the model.
                          If None, will be loaded from models/optimized_features.json
        """
        self.feature_names = feature_names
        if feature_names is None:
            self._load_feature_names()
    
    def _load_feature_names(self):
        """Load feature names from the saved model artifacts."""
        import json
        feature_path = Path("models/optimized_features.json")
        
        if not feature_path.exists():
            # Fallback to baseline model features
            feature_path = Path("models/feature_names.json")
        
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        else:
            raise FileNotFoundError(
                "Feature names file not found. Expected at models/optimized_features.json "
                "or models/feature_names.json"
            )
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-knowledge features.
        
        Features created (from Notebook 2):
        - loan_to_income: Loan amount / Annual income
        - payment_to_income: Monthly payment / Monthly income
        - credit_util_rate: Revolving balance / Revolving credit limit
        - rate_dti_interaction: Interest rate * DTI
        - total_debt_burden: (Installment + Revolving balance) / Annual income
        - stable_employment: Employment length >= 5 years
        - high_inquiries: Recent inquiries >= 2
        - has_delinquencies: Any delinquencies in last 2 years
        - high_risk_purpose: Purpose in high-risk categories
        
        Args:
            data: Input DataFrame with raw loan features
            
        Returns:
            DataFrame with engineered features added
        """
        df = data.copy()
        
        # 1. Loan to Income Ratio
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)  # +1 to avoid division by zero
        
        # 2. Payment to Income Ratio
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            monthly_income = df['annual_inc'] / 12
            df['payment_to_income'] = df['installment'] / (monthly_income + 1)
        
        # 3. Credit Utilization Rate
        if 'revol_bal' in df.columns and 'revol_util' in df.columns:
            # revol_util is already a percentage, but we recalculate for consistency
            df['credit_util_rate'] = df['revol_util'] / 100  # Convert to decimal
        
        # 4. Interest Rate * DTI Interaction
        if 'int_rate' in df.columns and 'dti' in df.columns:
            df['rate_dti_interaction'] = df['int_rate'] * df['dti']
        
        # 5. Total Debt Burden
        if 'installment' in df.columns and 'revol_bal' in df.columns and 'annual_inc' in df.columns:
            annual_installment = df['installment'] * 12
            df['total_debt_burden'] = (annual_installment + df['revol_bal']) / (df['annual_inc'] + 1)
        
        # 6. Stable Employment (binary)
        if 'emp_length' in df.columns:
            # emp_length is typically "< 1 year", "1 year", "2 years", ..., "10+ years"
            df['stable_employment'] = (
                df['emp_length'].astype(str).str.extract(r'(\d+)')[0].fillna(0).astype(float) >= 5
            ).astype(int)
        
        # 7. High Inquiries (binary)
        if 'inq_last_6mths' in df.columns:
            df['high_inquiries'] = (df['inq_last_6mths'] >= 2).astype(int)
        
        # 8. Has Delinquencies (binary)
        if 'delinq_2yrs' in df.columns:
            df['has_delinquencies'] = (df['delinq_2yrs'] > 0).astype(int)
        
        # 9. High Risk Purpose (binary)
        if 'purpose' in df.columns:
            high_risk_purposes = ['small_business', 'other', 'moving', 'vacation']
            df['high_risk_purpose'] = df['purpose'].isin(high_risk_purposes).astype(int)
        
        logger.info(f"Created {9} engineered features")
        return df
    
    def select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select only the features needed by the model.
        
        Args:
            data: DataFrame with all features
            
        Returns:
            DataFrame with only model features in correct order
        """
        # Get features that exist in both data and feature_names
        available_features = [f for f in self.feature_names if f in data.columns]
        
        if len(available_features) < len(self.feature_names):
            missing = set(self.feature_names) - set(available_features)
            logger.warning(f"Missing {len(missing)} features: {missing}")
        
        # Return features in the correct order
        return data[available_features]
    
    def transform(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Input data (DataFrame or dict for single loan)
            
        Returns:
            Preprocessed DataFrame ready for model prediction
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Create engineered features
        data_with_features = self.create_features(data)
        
        # Select model features
        model_features = self.select_features(data_with_features)
        
        logger.info(f"Processed {len(model_features)} samples with {len(model_features.columns)} features")
        
        return model_features


def load_and_preprocess(
    data_path: Union[str, Path] = None,
    data: Union[pd.DataFrame, Dict] = None,
    feature_names_path: Union[str, Path] = None
) -> pd.DataFrame:
    """
    Convenience function to load and preprocess data.
    
    Args:
        data_path: Path to CSV file to load (optional if data is provided)
        data: DataFrame or dict with loan data (optional if data_path is provided)
        feature_names_path: Path to feature names JSON (optional)
        
    Returns:
        Preprocessed DataFrame ready for prediction
        
    Example:
        >>> # From CSV file
        >>> features = load_and_preprocess(data_path='data/new_loans.csv')
        
        >>> # From DataFrame
        >>> df = pd.read_csv('loans.csv')
        >>> features = load_and_preprocess(data=df)
        
        >>> # Single loan dict
        >>> loan = {'loan_amnt': 15000, 'annual_inc': 60000, ...}
        >>> features = load_and_preprocess(data=loan)
    """
    # Load data if path provided
    if data_path is not None:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}: {len(data)} rows")
    
    if data is None:
        raise ValueError("Must provide either data_path or data")
    
    # Load feature names if path provided
    feature_names = None
    if feature_names_path is not None:
        import json
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
    
    # Create feature engineer and transform
    engineer = FeatureEngineer(feature_names=feature_names)
    processed_data = engineer.transform(data)
    
    return processed_data


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("=" * 50)
    
    # Example single loan
    example_loan = {
        'loan_amnt': 15000,
        'annual_inc': 60000,
        'int_rate': 10.5,
        'dti': 15.5,
        'installment': 500,
        'revol_bal': 5000,
        'revol_util': 50,
        'emp_length': '5 years',
        'inq_last_6mths': 1,
        'delinq_2yrs': 0,
        'purpose': 'debt_consolidation'
    }
    
    try:
        features = load_and_preprocess(data=example_loan)
        print(f"\nProcessed features shape: {features.shape}")
        print(f"Features: {features.columns.tolist()[:10]}...")
    except Exception as e:
        print(f"Example failed (expected - needs model files): {e}")
