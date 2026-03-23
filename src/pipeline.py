"""
End-to-End Credit Risk Pipeline

Orchestrates preprocessing and prediction for complete workflow.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Dict
import logging

from .data.preprocess import FeatureEngineer
from .models.predict import CreditRiskPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskPipeline:
    """
    Complete pipeline for credit risk assessment.
    
    Workflow:
    1. Load raw loan data
    2. Engineer features
    3. Make predictions
    4. Return results with business metrics
    
    Example:
        >>> pipeline = CreditRiskPipeline()
        >>> 
        >>> # Single loan
        >>> loan = {'loan_amnt': 15000, 'annual_inc': 60000, ...}
        >>> result = pipeline.predict(loan)
        >>> print(result['decision_label'])  # "Approve" or "Reject"
        >>> 
        >>> # Batch prediction
        >>> df = pd.read_csv('new_loans.csv')
        >>> results = pipeline.predict(df)
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = "models/optimized_model.pkl",
        feature_names_path: Union[str, Path] = None,
        threshold: float = 0.228,
        loss_per_default: float = 10000.0
    ):
        """
        Initialize the pipeline.
        
        Args:
            model_path: Path to trained model pickle file
            feature_names_path: Path to feature names JSON (optional)
            threshold: Decision threshold for loan approval
            loss_per_default: Expected loss per default in dollars
        """
        self.threshold = threshold
        self.loss_per_default = loss_per_default
        
        # Initialize feature engineer
        logger.info("Initializing feature engineer...")
        if feature_names_path:
            import json
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
            self.feature_engineer = FeatureEngineer(feature_names=feature_names)
        else:
            self.feature_engineer = FeatureEngineer()
        
        # Initialize predictor
        logger.info("Initializing predictor...")
        self.predictor = CreditRiskPredictor(
            model_path=model_path,
            threshold=threshold,
            loss_per_default=loss_per_default
        )
        
        logger.info("✓ Pipeline initialized successfully")
    
    def preprocess(
        self,
        data: Union[pd.DataFrame, Dict]
    ) -> pd.DataFrame:
        """
        Preprocess raw loan data into model features.
        
        Args:
            data: Raw loan data (DataFrame or dict)
            
        Returns:
            Preprocessed features ready for prediction
        """
        return self.feature_engineer.transform(data)
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict],
        skip_preprocessing: bool = False
    ) -> Union[pd.DataFrame, Dict]:
        """
        Complete prediction workflow.
        
        Args:
            data: Raw loan data OR preprocessed features
            skip_preprocessing: If True, assumes data is already preprocessed
            
        Returns:
            - If input is dict: Dictionary with prediction
            - If input is DataFrame: DataFrame with predictions
        """
        is_single = isinstance(data, dict)
        
        # Preprocess if needed
        if not skip_preprocessing:
            logger.info("Preprocessing data...")
            features = self.preprocess(data)
        else:
            features = data if isinstance(data, pd.DataFrame) else pd.DataFrame([data])
        
        # Make predictions
        logger.info("Making predictions...")
        if is_single:
            results = self.predictor.predict_single(features)
        else:
            results = self.predictor.predict_full(features)
        
        return results
    
    def predict_from_csv(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path] = None,
        skip_preprocessing: bool = False
    ) -> pd.DataFrame:
        """
        Load CSV, predict, and optionally save results.
        
        Args:
            input_path: Path to input CSV with loan data
            output_path: Path to save predictions CSV (optional)
            skip_preprocessing: If True, assumes CSV has preprocessed features
            
        Returns:
            DataFrame with predictions
        """
        # Load data
        logger.info(f"Loading data from {input_path}")
        data = pd.read_csv(input_path)
        
        # Make predictions
        results = self.predict(data, skip_preprocessing=skip_preprocessing)
        
        # Save if output path provided
        if output_path:
            logger.info(f"Saving predictions to {output_path}")
            results.to_csv(output_path, index=False)
        
        return results
    
    def get_summary_stats(
        self,
        predictions: pd.DataFrame
    ) -> Dict:
        """
        Calculate summary statistics for a batch of predictions.
        
        Args:
            predictions: DataFrame with prediction results
            
        Returns:
            Dictionary with summary metrics
        """
        total_loans = len(predictions)
        total_approved = (predictions['decision'] == 0).sum()
        total_rejected = (predictions['decision'] == 1).sum()
        approval_rate = (total_approved / total_loans) * 100
        
        avg_probability = predictions['default_probability'].mean()
        total_ecl = predictions['expected_loss'].sum()
        avg_ecl = predictions['expected_loss'].mean()
        
        high_risk = (predictions['default_probability'] >= 0.5).sum()
        medium_risk = (
            (predictions['default_probability'] >= 0.25) & 
            (predictions['default_probability'] < 0.5)
        ).sum()
        low_risk = (predictions['default_probability'] < 0.25).sum()
        
        summary = {
            'total_loans': total_loans,
            'approved': total_approved,
            'rejected': total_rejected,
            'approval_rate_pct': approval_rate,
            'avg_default_probability': avg_probability,
            'total_expected_loss': total_ecl,
            'avg_expected_loss': avg_ecl,
            'risk_distribution': {
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk
            },
            'threshold_used': self.threshold
        }
        
        return summary


def predict_loans(
    data: Union[pd.DataFrame, Dict, str, Path],
    model_path: Union[str, Path] = "models/optimized_model.pkl",
    threshold: float = 0.228,
    output_path: Union[str, Path] = None
) -> Union[pd.DataFrame, Dict]:
    """
    Convenience function for quick predictions.
    
    Args:
        data: Loan data (DataFrame, dict, or path to CSV)
        model_path: Path to model file
        threshold: Decision threshold
        output_path: Path to save results CSV (optional, only for batch)
        
    Returns:
        Prediction results
        
    Example:
        >>> # Single loan
        >>> loan = {'loan_amnt': 15000, 'annual_inc': 60000, ...}
        >>> result = predict_loans(loan)
        
        >>> # From CSV
        >>> results = predict_loans('new_loans.csv', output_path='predictions.csv')
    """
    pipeline = CreditRiskPipeline(
        model_path=model_path,
        threshold=threshold
    )
    
    # Handle file path input
    if isinstance(data, (str, Path)):
        return pipeline.predict_from_csv(data, output_path=output_path)
    
    # Handle dict/DataFrame input
    return pipeline.predict(data)


if __name__ == "__main__":
    # Example usage
    print("Credit Risk Pipeline")
    print("=" * 50)
    
    try:
        pipeline = CreditRiskPipeline()
        print("✓ Pipeline initialized successfully")
        print(f"  Model threshold: {pipeline.threshold}")
        print(f"  Loss per default: ${pipeline.loss_per_default:,.0f}")
        print("\nReady for predictions!")
        
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        print("\nThis is expected without model files in models/ directory")
        print("\nExample usage once models are available:")
        print("  from src.pipeline import CreditRiskPipeline")
        print("  pipeline = CreditRiskPipeline()")
        print("  result = pipeline.predict({'loan_amnt': 15000, ...})")
