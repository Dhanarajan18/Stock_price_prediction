"""
Evaluation module for assessing model performance.
Implements MAE, RMSE, and MAPE metrics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates stock price prediction models using various metrics.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        pass
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        mae = np.mean(np.abs(y_true - y_pred))
        return mae
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return rmse
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value (as percentage)
        """
        # Avoid division by zero
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error (MSE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MSE value
        """
        mse = np.mean((y_true - y_pred) ** 2)
        return mse
    
    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R-squared (coefficient of determination).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R-squared value
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        return r2
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy (as percentage)
        """
        if len(y_true) < 2:
            return 0.0
        
        # Calculate actual and predicted directions
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Calculate accuracy
        correct = np.sum(true_direction == pred_direction)
        total = len(true_direction)
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        return accuracy
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Evaluate model using all metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary with all metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Ensure arrays are 1D
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Ensure arrays have the same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Calculate all metrics
        metrics = {
            'MAE': self.mean_absolute_error(y_true, y_pred),
            'RMSE': self.root_mean_squared_error(y_true, y_pred),
            'MAPE': self.mean_absolute_percentage_error(y_true, y_pred),
            'MSE': self.mean_squared_error(y_true, y_pred),
            'R2': self.r_squared(y_true, y_pred),
            'Directional_Accuracy': self.directional_accuracy(y_true, y_pred)
        }
        
        return metrics
    
    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            y_true: True values
            predictions: Dictionary of {model_name: predictions}
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing models...")
        
        results = {}
        
        for model_name, y_pred in predictions.items():
            metrics = self.evaluate(y_true, y_pred, model_name)
            results[model_name] = metrics
        
        # Create DataFrame
        df_results = pd.DataFrame(results).T
        
        # Sort by RMSE (lower is better)
        df_results = df_results.sort_values('RMSE')
        
        return df_results
    
    def print_evaluation_report(
        self,
        metrics: Dict[str, float],
        model_name: str = "Model"
    ) -> None:
        """
        Print a formatted evaluation report.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        print("\n" + "="*60)
        print(f"Evaluation Report: {model_name}")
        print("="*60)
        
        print(f"\nError Metrics:")
        print(f"  Mean Absolute Error (MAE):           {metrics['MAE']:.4f}")
        print(f"  Root Mean Squared Error (RMSE):      {metrics['RMSE']:.4f}")
        print(f"  Mean Squared Error (MSE):            {metrics['MSE']:.4f}")
        print(f"  Mean Absolute Percentage Error:      {metrics['MAPE']:.2f}%")
        
        print(f"\nPerformance Metrics:")
        print(f"  R-Squared (R²):                      {metrics['R2']:.4f}")
        print(f"  Directional Accuracy:                {metrics['Directional_Accuracy']:.2f}%")
        
        print("="*60 + "\n")


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    print_report: bool = True
) -> Dict[str, float]:
    """
    Convenience function to evaluate a model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        print_report: Whether to print evaluation report
        
    Returns:
        Dictionary with metrics
    """
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, model_name)
    
    if print_report:
        evaluator.print_evaluation_report(metrics, model_name)
    
    return metrics


def evaluate_predictions(
    y_true: np.ndarray,
    arima_pred: np.ndarray = None,
    lstm_pred: np.ndarray = None
) -> pd.DataFrame:
    """
    Evaluate and compare ARIMA and LSTM predictions.
    
    Args:
        y_true: True values
        arima_pred: ARIMA predictions
        lstm_pred: LSTM predictions
        
    Returns:
        DataFrame with comparison results
    """
    evaluator = ModelEvaluator()
    predictions = {}
    
    if arima_pred is not None:
        predictions['ARIMA'] = arima_pred
    
    if lstm_pred is not None:
        predictions['LSTM'] = lstm_pred
    
    if not predictions:
        logger.warning("No predictions provided for evaluation")
        return pd.DataFrame()
    
    # Compare models
    results = evaluator.compare_models(y_true, predictions)
    
    # Print individual reports
    for model_name, pred in predictions.items():
        metrics = evaluator.evaluate(y_true, pred, model_name)
        evaluator.print_evaluation_report(metrics, model_name)
    
    # Print comparison
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print(results)
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    # Test the evaluator
    print("Testing Model Evaluator...")
    
    # Create sample data
    np.random.seed(42)
    y_true = np.random.uniform(100, 200, 50)
    
    # Simulate predictions with some noise
    arima_pred = y_true + np.random.normal(0, 5, 50)
    lstm_pred = y_true + np.random.normal(0, 3, 50)
    
    print(f"True values shape: {y_true.shape}")
    print(f"ARIMA predictions shape: {arima_pred.shape}")
    print(f"LSTM predictions shape: {lstm_pred.shape}")
    
    # Evaluate predictions
    results = evaluate_predictions(y_true, arima_pred, lstm_pred)
    
    print("\nEvaluation completed!")
