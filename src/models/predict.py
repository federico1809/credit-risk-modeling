"""
Credit Risk Prediction Module

Loads trained model and makes predictions with optimal threshold.
"""

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskPredictor:
    """
    Credit risk predictor using trained XGBoost model.
    
    Handles:
    - Model loading (optimized or baseline)
    - Probability prediction
    - Binary decision with optimal threshold
    - ECL (Expected Credit Loss) calculation
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = "models/optimized_model.pkl",
        threshold: float = 0.228,
        loss_per_default: float = 10000.0
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to pickled model file
            threshold: Decision threshold for default classification (from Notebook 4)
            loss_per_default: Expected loss per default in dollars
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.loss_per_default = loss_per_default
        self.model = None
        self.imputer = None
        
        self._load_model()
        self._load_imputer()
    
    def _load_model(self):
        """Load the trained model from pickle file."""
        if not self.model_path.exists():
            # Try baseline model as fallback
            fallback_path = Path("models/best_model.pkl")
            if fallback_path.exists():
                logger.warning(f"Model not found at {self.model_path}, using {fallback_path}")
                self.model_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path} or {fallback_path}"
                )
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        logger.info(f"Loaded model from {self.model_path}")
    
    def _load_imputer(self):
        """Load the data imputer for handling missing values."""
        imputer_path = Path("models/data_imputer.pkl")
        
        if imputer_path.exists():
            with open(imputer_path, 'rb') as f:
                self.imputer = pickle.load(f)
            logger.info(f"Loaded imputer from {imputer_path}")
        else:
            logger.warning("No imputer found - missing values may cause errors")
            self.imputer = None
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict default probabilities.
        
        Args:
            X: Feature DataFrame (already preprocessed)
            
        Returns:
            Array of default probabilities (shape: [n_samples])
        """
        # Apply imputer if available
        if self.imputer is not None:
            X_imputed = pd.DataFrame(
                self.imputer.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_imputed = X
        
        # Predict probabilities
        # XGBoost returns [prob_class_0, prob_class_1], we want prob_class_1
        probas = self.model.predict_proba(X_imputed)[:, 1]
        
        return probas
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary default decisions using optimal threshold.
        
        Args:
            X: Feature DataFrame (already preprocessed)
            
        Returns:
            Binary predictions (1 = default, 0 = no default)
        """
        probas = self.predict_proba(X)
        decisions = (probas >= self.threshold).astype(int)
        
        return decisions
    
    def calculate_ecl(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate Expected Credit Loss for each loan.
        
        ECL = Probability of Default * Loss Given Default
        
        Args:
            probabilities: Default probabilities
            
        Returns:
            Expected credit loss per loan in dollars
        """
        ecl = probabilities * self.loss_per_default
        return ecl
    
    def predict_full(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Complete prediction with probabilities, decisions, and ECL.
        
        Args:
            X: Feature DataFrame (already preprocessed)
            
        Returns:
            DataFrame with columns:
            - default_probability: Probability of default
            - decision: Binary decision (1=reject, 0=approve)
            - decision_label: Text label ("Reject" or "Approve")
            - expected_loss: Expected credit loss in dollars
        """
        # Get probabilities
        probas = self.predict_proba(X)
        
        # Get binary decisions
        decisions = (probas >= self.threshold).astype(int)
        
        # Calculate ECL
        ecl = self.calculate_ecl(probas)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'default_probability': probas,
            'decision': decisions,
            'decision_label': ['Reject' if d == 1 else 'Approve' for d in decisions],
            'expected_loss': ecl
        }, index=X.index)
        
        return results
    
    def predict_single(
        self,
        features: pd.DataFrame
    ) -> Dict[str, Union[float, str, int]]:
        """
        Predict for a single loan and return as dictionary.
        
        Args:
            features: Feature DataFrame with 1 row
            
        Returns:
            Dictionary with prediction results
        """
        results_df = self.predict_full(features)
        
        # Convert to dict
        result_dict = {
            'default_probability': float(results_df['default_probability'].iloc[0]),
            'decision': int(results_df['decision'].iloc[0]),
            'decision_label': results_df['decision_label'].iloc[0],
            'expected_loss': float(results_df['expected_loss'].iloc[0]),
            'threshold_used': self.threshold
        }
        
        return result_dict


def predict_default_risk(
    X: Union[pd.DataFrame, Dict],
    model_path: Union[str, Path] = "models/optimized_model.pkl",
    threshold: float = 0.228,
    return_proba: bool = False
) -> Union[np.ndarray, pd.DataFrame, Dict]:
    """
    Convenience function for quick predictions.
    
    Args:
        X: Features (DataFrame or dict for single loan)
        model_path: Path to model file
        threshold: Decision threshold
        return_proba: If True, return only probabilities. If False, return full results.
        
    Returns:
        - If return_proba=True: Array of probabilities
        - If return_proba=False and X is DataFrame: DataFrame with full results
        - If return_proba=False and X is dict: Dictionary with results
        
    Example:
        >>> # Single loan
        >>> loan_features = {...}  # preprocessed features
        >>> result = predict_default_risk(loan_features)
        >>> print(result['decision_label'])  # "Approve" or "Reject"
        
        >>> # Batch prediction
        >>> df = pd.read_csv('preprocessed_loans.csv')
        >>> results = predict_default_risk(df)
        >>> print(results[['default_probability', 'decision_label']])
    """
    predictor = CreditRiskPredictor(
        model_path=model_path,
        threshold=threshold
    )
    
    # Handle dict input (single loan)
    is_single = isinstance(X, dict)
    if is_single:
        X = pd.DataFrame([X])
    
    # Return probabilities only if requested
    if return_proba:
        return predictor.predict_proba(X)
    
    # Return full results
    if is_single:
        return predictor.predict_single(X)
    else:
        return predictor.predict_full(X)


if __name__ == "__main__":
    # Example usage
    print("Credit Risk Prediction Module")
    print("=" * 50)
    
    # This will fail without actual model files, but shows the API
    try:
        predictor = CreditRiskPredictor()
        print(f"✓ Model loaded successfully")
        print(f"  Threshold: {predictor.threshold}")
        print(f"  Loss per default: ${predictor.loss_per_default:,.0f}")
    except FileNotFoundError as e:
        print(f"✗ Model not found (expected): {e}")
        print("\nThis is normal - model files needed in models/ directory")
