"""
Model evaluation module for the House Price Prediction Project
Master 2 DS - Machine Learning

This module contains functions for evaluating model performance and computing metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
import logging
from pathlib import Path

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate various regression metrics.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    metrics : list of str, optional
        List of metrics to calculate. If None, calculates all available.
        Available metrics: 'mae', 'mse', 'rmse', 'r2', 'mape', 'explained_variance'
    
    Returns
    -------
    dict
        Dictionary with metric names as keys and values as scores.
    
    Examples
    --------
    >>> metrics = calculate_metrics(y_true, y_pred)
    >>> print(f"R2 Score: {metrics['r2']:.4f}")
    """
    if metrics is None:
        metrics = ['mae', 'mse', 'rmse', 'r2', 'mape', 'explained_variance']
    
    results = {}
    
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate requested metrics
    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_true, y_pred)
    
    if 'mse' in metrics:
        results['mse'] = mean_squared_error(y_true, y_pred)
    
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if 'r2' in metrics:
        results['r2'] = r2_score(y_true, y_pred)
    
    if 'mape' in metrics:
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            results['mape'] = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
        else:
            results['mape'] = np.nan
    
    if 'explained_variance' in metrics:
        results['explained_variance'] = explained_variance_score(y_true, y_pred)
    
    # Additional custom metrics
    if 'mean_error' in metrics:
        results['mean_error'] = np.mean(y_pred - y_true)
    
    if 'std_error' in metrics:
        results['std_error'] = np.std(y_pred - y_true)
    
    if 'max_error' in metrics:
        results['max_error'] = np.max(np.abs(y_pred - y_true))
    
    return results


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a model on test data and return comprehensive metrics.
    
    Parameters
    ----------
    model : estimator
        Trained model with predict method.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target values.
    model_name : str, default="Model"
        Name of the model for logging.
    verbose : bool, default=True
        Whether to print evaluation results.
    
    Returns
    -------
    dict
        Dictionary containing predictions, metrics, and analysis.
    
    Examples
    --------
    >>> results = evaluate_model(model, X_test, y_test, model_name="Random Forest")
    >>> print(f"Test R2: {results['metrics']['r2']:.4f}")
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Create results dictionary
    results = {
        'model_name': model_name,
        'predictions': y_pred,
        'metrics': metrics,
        'residuals': residuals,
        'residual_stats': {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'q25': residuals.quantile(0.25),
            'q50': residuals.quantile(0.50),
            'q75': residuals.quantile(0.75)
        }
    }
    
    # Calculate percentage within tolerance
    tolerance_levels = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%
    results['tolerance_accuracy'] = {}
    for tol in tolerance_levels:
        within_tol = np.abs(residuals / y_test) <= tol
        results['tolerance_accuracy'][f'within_{int(tol*100)}pct'] = within_tol.mean()
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS - {model_name}")
        print("=" * 60)
        print("\n📊 Metrics:")
        print(f"  MAE:  ${metrics['mae']:,.0f}")
        print(f"  RMSE: ${metrics['rmse']:,.0f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        if not np.isnan(metrics['mape']):
            print(f"  MAPE: {metrics['mape']*100:.2f}%")
        
        print("\n📈 Residual Analysis:")
        print(f"  Mean: ${results['residual_stats']['mean']:,.0f}")
        print(f"  Std:  ${results['residual_stats']['std']:,.0f}")
        print(f"  Range: ${results['residual_stats']['min']:,.0f} to ${results['residual_stats']['max']:,.0f}")
        
        print("\n🎯 Prediction Accuracy:")
        for tol_name, accuracy in results['tolerance_accuracy'].items():
            tol_pct = int(tol_name.split('_')[1].replace('pct', ''))
            print(f"  Within {tol_pct}%: {accuracy:.1%}")
        
        print("=" * 60)
    
    logger.info(f"✓ Model '{model_name}' evaluated: R²={metrics['r2']:.4f}, MAE=${metrics['mae']:,.0f}")
    
    return results


def evaluate_multiple_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate multiple models and return comparison DataFrame.
    
    Parameters
    ----------
    models : dict
        Dictionary with model names as keys and models as values.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target values.
    verbose : bool, default=True
        Whether to print results.
    
    Returns
    -------
    pd.DataFrame
        Comparison of all models' performance.
    
    Examples
    --------
    >>> comparison = evaluate_multiple_models(models, X_test, y_test)
    >>> print(comparison.sort_values('r2', ascending=False))
    """
    results = []
    
    for name, model in models.items():
        eval_results = evaluate_model(model, X_test, y_test, name, verbose=False)
        
        # Extract key metrics for comparison
        model_summary = {
            'Model': name,
            'MAE': eval_results['metrics']['mae'],
            'RMSE': eval_results['metrics']['rmse'],
            'R²': eval_results['metrics']['r2'],
            'MAPE': eval_results['metrics']['mape'] * 100 if not np.isnan(eval_results['metrics']['mape']) else np.nan,
            'Within 10%': eval_results['tolerance_accuracy']['within_10pct'] * 100,
            'Within 20%': eval_results['tolerance_accuracy']['within_20pct'] * 100
        }
        results.append(model_summary)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    
    if verbose:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print("\n", comparison_df.to_string(index=False))
        print("\n" + "=" * 60)
        
        # Identify best model
        best_model = comparison_df.iloc[0]['Model']
        best_r2 = comparison_df.iloc[0]['R²']
        print(f"\n🏆 Best Model: {best_model} (R² = {best_r2:.4f})")
        print("=" * 60)
    
    return comparison_df


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Extract feature importance from a model.
    
    Parameters
    ----------
    model : estimator
        Trained model with feature_importances_ or coef_ attribute.
    feature_names : list of str
        Names of features.
    top_n : int, default=10
        Number of top features to return.
    
    Returns
    -------
    pd.DataFrame
        Feature importance DataFrame sorted by importance.
    
    Examples
    --------
    >>> importance = get_feature_importance(model, X_train.columns)
    >>> print(importance.head(10))
    """
    importance_df = None
    
    # Try to get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (Random Forest, Gradient Boosting)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Importance_Type': 'Feature Importance'
        })
        
    elif hasattr(model, 'coef_'):
        # Linear models
        coefficients = model.coef_
        if coefficients.ndim == 1:
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(coefficients),
                'Coefficient': coefficients,
                'Importance_Type': 'Absolute Coefficient'
            })
    
    if importance_df is not None:
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Add percentage contribution
        importance_df['Importance_Pct'] = (
            importance_df['Importance'] / importance_df['Importance'].sum() * 100
        )
        
        # Add cumulative percentage
        importance_df['Cumulative_Pct'] = importance_df['Importance_Pct'].cumsum()
        
        # Reset index
        importance_df = importance_df.reset_index(drop=True)
        
        # Return top N features
        if top_n and len(importance_df) > top_n:
            importance_df = importance_df.head(top_n)
        
        logger.info(f"✓ Feature importance extracted: Top {len(importance_df)} features")
        
        return importance_df
    else:
        logger.warning(f"Model type {type(model).__name__} does not have feature importance")
        return pd.DataFrame()


def analyze_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Model",
    n_samples: int = 10
) -> pd.DataFrame:
    """
    Analyze individual predictions with errors and percentages.
    
    Parameters
    ----------
    y_true : pd.Series
        True values.
    y_pred : np.ndarray
        Predicted values.
    model_name : str, default="Model"
        Name of the model.
    n_samples : int, default=10
        Number of samples to show.
    
    Returns
    -------
    pd.DataFrame
        Analysis of predictions with errors.
    
    Examples
    --------
    >>> analysis = analyze_predictions(y_test, predictions)
    >>> print(analysis.head(10))
    """
    analysis_df = pd.DataFrame({
        'Actual': y_true.values,
        'Predicted': y_pred,
        'Error': y_pred - y_true.values,
        'Abs_Error': np.abs(y_pred - y_true.values),
        'Pct_Error': (y_pred - y_true.values) / y_true.values * 100
    })
    
    # Sort by absolute error (worst predictions first)
    analysis_df = analysis_df.sort_values('Abs_Error', ascending=False)
    
    # Add ranking
    analysis_df['Error_Rank'] = range(1, len(analysis_df) + 1)
    
    # Reset index to keep original index
    analysis_df = analysis_df.reset_index(names='Original_Index')
    
    if n_samples:
        print(f"\n📊 Top {n_samples} Prediction Errors - {model_name}:")
        print("-" * 60)
        display_df = analysis_df.head(n_samples)[
            ['Original_Index', 'Actual', 'Predicted', 'Error', 'Pct_Error']
        ]
        print(display_df.to_string(index=False))
    
    return analysis_df


def check_overfitting(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.15
) -> Dict[str, Any]:
    """
    Check for overfitting by comparing train and test performance.
    
    Parameters
    ----------
    model : estimator
        Trained model.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    model_name : str, default="Model"
        Name of the model.
    threshold : float, default=0.15
        Maximum acceptable difference between train and test scores.
    
    Returns
    -------
    dict
        Overfitting analysis results.
    
    Examples
    --------
    >>> overfitting = check_overfitting(model, X_train, y_train, X_test, y_test)
    >>> if overfitting['is_overfitting']:
    ...     print("⚠️ Model shows signs of overfitting!")
    """
    # Calculate train performance
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    
    # Calculate test performance
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Calculate differences
    r2_diff = train_metrics['r2'] - test_metrics['r2']
    rmse_diff_pct = (test_metrics['rmse'] - train_metrics['rmse']) / train_metrics['rmse'] * 100
    
    # Determine if overfitting
    is_overfitting = r2_diff > threshold
    
    results = {
        'model_name': model_name,
        'train_r2': train_metrics['r2'],
        'test_r2': test_metrics['r2'],
        'r2_difference': r2_diff,
        'train_rmse': train_metrics['rmse'],
        'test_rmse': test_metrics['rmse'],
        'rmse_increase_pct': rmse_diff_pct,
        'is_overfitting': is_overfitting,
        'severity': 'High' if r2_diff > 0.25 else 'Medium' if r2_diff > 0.15 else 'Low' if r2_diff > 0.05 else 'None'
    }
    
    print(f"\n🔍 Overfitting Check - {model_name}:")
    print("-" * 40)
    print(f"Train R²: {results['train_r2']:.4f}")
    print(f"Test R²:  {results['test_r2']:.4f}")
    print(f"Difference: {results['r2_difference']:.4f}")
    print(f"Status: {'⚠️ OVERFITTING' if is_overfitting else '✅ OK'} (Severity: {results['severity']})")
    
    return results


def create_evaluation_report(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive evaluation report for all models.
    
    Parameters
    ----------
    models : dict
        Dictionary of trained models.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    save_path : str or Path, optional
        Path to save the report.
    
    Returns
    -------
    dict
        Complete evaluation report.
    
    Examples
    --------
    >>> report = create_evaluation_report(models, X_train, y_train, X_test, y_test)
    >>> print(report['best_model'])
    """
    report = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': X_train.shape[1],
            'target_mean_train': y_train.mean(),
            'target_mean_test': y_test.mean(),
            'target_std_train': y_train.std(),
            'target_std_test': y_test.std()
        },
        'model_results': {},
        'comparison': None,
        'best_model': None,
        'overfitting_analysis': {}
    }
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("=" * 70)
    
    # Evaluate each model
    for name, model in models.items():
        print(f"\n📊 Evaluating {name}...")
        report['model_results'][name] = evaluate_model(
            model, X_test, y_test, name, verbose=False
        )
        
        # Check overfitting
        report['overfitting_analysis'][name] = check_overfitting(
            model, X_train, y_train, X_test, y_test, name
        )
    
    # Create comparison
    report['comparison'] = evaluate_multiple_models(models, X_test, y_test, verbose=True)
    
    # Identify best model
    best_idx = report['comparison']['R²'].idxmax()
    report['best_model'] = {
        'name': report['comparison'].loc[best_idx, 'Model'],
        'r2': report['comparison'].loc[best_idx, 'R²'],
        'mae': report['comparison'].loc[best_idx, 'MAE'],
        'rmse': report['comparison'].loc[best_idx, 'RMSE']
    }
    
    # Save report if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        import json
        
        # Convert numpy/pandas objects for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.to_dict()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        # Create serializable report
        json_report = json.dumps(report, default=convert_to_serializable, indent=2)
        
        with open(save_path, 'w') as f:
            f.write(json_report)
        
        logger.info(f"✓ Report saved to: {save_path}")
    
    return report


if __name__ == "__main__":
    """
    Test the evaluation functions if run directly.
    """
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # True values
    y_true = pd.Series(np.random.uniform(100000, 500000, n_samples), name='price')
    
    # Create predictions with some error
    y_pred_good = y_true + np.random.normal(0, 20000, n_samples)
    y_pred_bad = y_true + np.random.normal(10000, 50000, n_samples)
    
    print("Testing Evaluation Functions")
    print("=" * 60)
    
    # Test metrics calculation
    print("\n1. Testing Metrics Calculation:")
    metrics_good = calculate_metrics(y_true, y_pred_good)
    metrics_bad = calculate_metrics(y_true, y_pred_bad)
    
    print("\nGood Model Metrics:")
    for metric, value in metrics_good.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nBad Model Metrics:")
    for metric, value in metrics_bad.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test prediction analysis
    print("\n2. Testing Prediction Analysis:")
    analysis = analyze_predictions(y_true, y_pred_good, "Test Model", n_samples=5)
    
    print("\n✓ All evaluation tests completed successfully!")
