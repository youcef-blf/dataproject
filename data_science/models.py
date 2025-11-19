"""
Machine Learning models for the House Price Prediction Project
Master 2 DS - Machine Learning

This module contains implementations of various regression models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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


class BaselineModel(BaseEstimator, RegressorMixin):
    """
    Baseline model that predicts the mean price based on number of bedrooms.
    
    This simple model serves as a reference point for more complex models.
    
    Parameters
    ----------
    feature_col : str, default='bedrooms'
        The feature to use for grouping (typically 'bedrooms').
    
    Attributes
    ----------
    feature_means_ : dict
        Dictionary mapping feature values to mean prices.
    overall_mean_ : float
        Overall mean price (used when feature value not seen in training).
    
    Examples
    --------
    >>> baseline = BaselineModel()
    >>> baseline.fit(X_train, y_train)
    >>> predictions = baseline.predict(X_test)
    """
    
    def __init__(self, feature_col: str = 'bedrooms'):
        self.feature_col = feature_col
        self.feature_means_ = {}
        self.overall_mean_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the baseline model by calculating mean prices per feature value.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Training target values.
        
        Returns
        -------
        self
            Fitted model.
        """
        # Check if feature column exists
        if self.feature_col not in X.columns:
            raise ValueError(f"Feature '{self.feature_col}' not found in X")
        
        # Calculate overall mean
        self.overall_mean_ = y.mean()
        
        # Calculate mean price for each feature value
        data = pd.DataFrame({self.feature_col: X[self.feature_col], 'price': y})
        self.feature_means_ = data.groupby(self.feature_col)['price'].mean().to_dict()
        
        logger.info(f"✓ Baseline model fitted using '{self.feature_col}'")
        logger.info(f"  Overall mean: {self.overall_mean_:,.0f}")
        logger.info(f"  Feature groups: {len(self.feature_means_)}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the baseline model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on.
        
        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if self.overall_mean_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        for value in X[self.feature_col]:
            # Use feature mean if available, otherwise use overall mean
            pred = self.feature_means_.get(value, self.overall_mean_)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {'feature_col': self.feature_col}
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for param, value in params.items():
            setattr(self, param, value)
        return self


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_col: Optional[str] = None
) -> BaselineModel:
    """
    Train a baseline model.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    feature_col : str, optional
        Feature to use for baseline. If None, uses config.BASELINE_FEATURE.
    
    Returns
    -------
    BaselineModel
        Trained baseline model.
    
    Examples
    --------
    >>> baseline = train_baseline_model(X_train, y_train)
    >>> print(f"Baseline model trained with {len(baseline.feature_means_)} groups")
    """
    if feature_col is None:
        feature_col = config.BASELINE_FEATURE
    
    model = BaselineModel(feature_col=feature_col)
    model.fit(X_train, y_train)
    
    return model


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    normalize: bool = False
) -> LinearRegression:
    """
    Train a linear regression model with optional regularization.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    regularization : {None, 'ridge', 'lasso'}, optional
        Type of regularization to apply.
    alpha : float, default=1.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to fit intercept.
    normalize : bool, default=False
        Whether to normalize features.
    
    Returns
    -------
    LinearRegression or Ridge or Lasso
        Trained linear model.
    
    Examples
    --------
    >>> lr_model = train_linear_regression(X_train, y_train)
    >>> print(f"Linear model coefficients: {lr_model.coef_}")
    """
    if regularization == 'ridge':
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        model_name = f"Ridge (alpha={alpha})"
    elif regularization == 'lasso':
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000)
        model_name = f"Lasso (alpha={alpha})"
    else:
        model = LinearRegression(fit_intercept=fit_intercept)
        model_name = "Linear Regression"
    
    # Train model
    model.fit(X_train, y_train)
    
    logger.info(f"✓ {model_name} model trained")
    logger.info(f"  Number of features: {X_train.shape[1]}")
    logger.info(f"  Intercept: {model.intercept_:,.2f}")
    
    # Log feature importance (absolute coefficients)
    if hasattr(model, 'coef_'):
        top_features_idx = np.argsort(np.abs(model.coef_))[-5:]
        top_features = [(X_train.columns[i], model.coef_[i]) for i in top_features_idx]
        logger.info(f"  Top 5 features by coefficient magnitude:")
        for feat, coef in reversed(top_features):
            logger.info(f"    - {feat}: {coef:,.2f}")
    
    return model


def train_ensemble_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    **kwargs
) -> Union[RandomForestRegressor, GradientBoostingRegressor]:
    """
    Train an ensemble model (Random Forest or Gradient Boosting).
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    model_type : {'random_forest', 'gradient_boosting'}, default='random_forest'
        Type of ensemble model.
    **kwargs
        Additional parameters for the model.
    
    Returns
    -------
    RandomForestRegressor or GradientBoostingRegressor
        Trained ensemble model.
    
    Examples
    --------
    >>> rf_model = train_ensemble_model(X_train, y_train, model_type='random_forest')
    >>> print(f"Feature importances: {rf_model.feature_importances_}")
    """
    if model_type == 'random_forest':
        # Get default parameters
        params = config.RF_DEFAULT_PARAMS.copy()
        params.update(kwargs)
        
        model = RandomForestRegressor(**params)
        model_name = "Random Forest"
        
    elif model_type == 'gradient_boosting':
        # Get default parameters
        params = config.GB_DEFAULT_PARAMS.copy()
        params.update(kwargs)
        
        model = GradientBoostingRegressor(**params)
        model_name = "Gradient Boosting"
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    logger.info(f"✓ {model_name} model trained")
    logger.info(f"  Parameters: {params}")
    
    # Log feature importance
    if hasattr(model, 'feature_importances_'):
        top_features_idx = np.argsort(model.feature_importances_)[-5:]
        top_features = [(X_train.columns[i], model.feature_importances_[i]) 
                       for i in top_features_idx]
        logger.info(f"  Top 5 features by importance:")
        for feat, importance in reversed(top_features):
            logger.info(f"    - {feat}: {importance:.4f}")
    
    return model


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    search_type: str = 'grid',
    param_grid: Optional[Dict] = None,
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = 'r2',
    n_jobs: int = -1,
    verbose: int = 1
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    model_type : {'random_forest', 'gradient_boosting'}, default='random_forest'
        Type of model to tune.
    search_type : {'grid', 'random'}, default='grid'
        Type of search to perform.
    param_grid : dict, optional
        Parameter grid for search. If None, uses config defaults.
    n_iter : int, default=20
        Number of iterations for RandomizedSearchCV.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default='r2'
        Scoring metric.
    n_jobs : int, default=-1
        Number of parallel jobs.
    verbose : int, default=1
        Verbosity level.
    
    Returns
    -------
    tuple
        (best_model, best_params, cv_results_df)
    
    Examples
    --------
    >>> best_model, best_params, results = tune_hyperparameters(
    ...     X_train, y_train, model_type='random_forest'
    ... )
    >>> print(f"Best parameters: {best_params}")
    """
    # Select base model and parameter grid
    if model_type == 'random_forest':
        base_model = RandomForestRegressor(random_state=config.RANDOM_STATE)
        if param_grid is None:
            param_grid = config.PARAM_GRID_RF
    elif model_type == 'gradient_boosting':
        base_model = GradientBoostingRegressor(random_state=config.RANDOM_STATE)
        if param_grid is None:
            param_grid = config.PARAM_GRID_GB
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Starting hyperparameter tuning for {model_type}...")
    logger.info(f"Search type: {search_type}")
    logger.info(f"Parameter grid: {param_grid}")
    
    # Perform search
    if search_type == 'grid':
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )
    elif search_type == 'random':
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=config.RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown search type: {search_type}")
    
    # Fit search
    search.fit(X_train, y_train)
    
    # Get results
    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_results_df = pd.DataFrame(search.cv_results_)
    
    logger.info(f"✓ Hyperparameter tuning complete")
    logger.info(f"  Best score: {search.best_score_:.4f}")
    logger.info(f"  Best parameters: {best_params}")
    
    return best_model, best_params, cv_results_df


def get_learning_curves(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_sizes: Optional[List[int]] = None,
    cv: int = 5,
    scoring: str = 'r2',
    n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate learning curves for a model.
    
    Parameters
    ----------
    model : estimator
        The model to evaluate.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    train_sizes : list of int, optional
        Training set sizes. If None, uses config.TRAIN_SIZES.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default='r2'
        Scoring metric.
    n_jobs : int, default=-1
        Number of parallel jobs.
    
    Returns
    -------
    tuple
        (train_sizes_abs, train_scores, val_scores)
    
    Examples
    --------
    >>> sizes, train_scores, val_scores = get_learning_curves(model, X_train, y_train)
    >>> print(f"Training sizes: {sizes}")
    """
    if train_sizes is None:
        train_sizes = config.TRAIN_SIZES
    
    # Filter train sizes that are valid for the dataset
    max_size = int(len(X_train) * (cv - 1) / cv)  # Maximum size for CV
    train_sizes = [size for size in train_sizes if size <= max_size]
    
    logger.info(f"Generating learning curves for {type(model).__name__}...")
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )
    
    logger.info(f"✓ Learning curves generated for {len(train_sizes_abs)} sizes")
    
    return train_sizes_abs, train_scores, val_scores


def perform_cross_validation(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: Union[str, List[str]] = 'r2'
) -> Union[float, Dict[str, float]]:
    """
    Perform cross-validation on a model.
    
    Parameters
    ----------
    model : estimator
        The model to evaluate.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    cv : int, default=5
        Number of folds.
    scoring : str or list of str, default='r2'
        Scoring metric(s).
    
    Returns
    -------
    float or dict
        Cross-validation score(s).
    
    Examples
    --------
    >>> score = perform_cross_validation(model, X_train, y_train)
    >>> print(f"CV Score: {score:.4f}")
    """
    if isinstance(scoring, str):
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        result = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        logger.info(f"✓ Cross-validation ({cv} folds): {scoring}={result['mean']:.4f} (±{result['std']:.4f})")
        return result
    else:
        # Multiple scoring metrics
        from sklearn.model_selection import cross_validate
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        
        results = {}
        for metric in scoring:
            key = f'test_{metric}'
            if key in scores:
                results[metric] = {
                    'mean': scores[key].mean(),
                    'std': scores[key].std(),
                    'scores': scores[key]
                }
                logger.info(f"✓ CV {metric}: {results[metric]['mean']:.4f} (±{results[metric]['std']:.4f})")
        
        return results


def save_model(
    model: Any,
    filepath: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None
) -> Path:
    """
    Save a trained model to disk.
    
    Parameters
    ----------
    model : estimator
        The model to save.
    filepath : str or Path, optional
        Path to save the model. If None, auto-generates based on model_name.
    model_name : str, optional
        Name for the model file.
    
    Returns
    -------
    Path
        Path where the model was saved.
    
    Examples
    --------
    >>> path = save_model(model, model_name='best_rf_model')
    >>> print(f"Model saved to: {path}")
    """
    if filepath is None:
        if model_name is None:
            model_name = type(model).__name__.lower()
        
        models_dir = Path(config.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = models_dir / f"{model_name}.pkl"
    else:
        filepath = Path(filepath)
    
    joblib.dump(model, filepath)
    logger.info(f"✓ Model saved to: {filepath}")
    
    return filepath


def load_model(filepath: Union[str, Path]) -> Any:
    """
    Load a saved model from disk.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the saved model.
    
    Returns
    -------
    estimator
        The loaded model.
    
    Examples
    --------
    >>> model = load_model('models/best_model.pkl')
    >>> predictions = model.predict(X_test)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    logger.info(f"✓ Model loaded from: {filepath}")
    
    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    include_baseline: bool = True,
    include_linear: bool = True,
    include_rf: bool = True,
    include_gb: bool = True
) -> Dict[str, Any]:
    """
    Train all model types and return them in a dictionary.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    include_baseline : bool, default=True
        Whether to include baseline model.
    include_linear : bool, default=True
        Whether to include linear regression.
    include_rf : bool, default=True
        Whether to include Random Forest.
    include_gb : bool, default=True
        Whether to include Gradient Boosting.
    
    Returns
    -------
    dict
        Dictionary with model names as keys and trained models as values.
    
    Examples
    --------
    >>> models = train_all_models(X_train, y_train)
    >>> for name, model in models.items():
    ...     print(f"{name}: {type(model).__name__}")
    """
    models = {}
    
    logger.info("=" * 60)
    logger.info("TRAINING ALL MODELS")
    logger.info("=" * 60)
    
    if include_baseline:
        logger.info("\n1. Training Baseline Model...")
        models['baseline'] = train_baseline_model(X_train, y_train)
    
    if include_linear:
        logger.info("\n2. Training Linear Regression...")
        models['linear_regression'] = train_linear_regression(X_train, y_train)
    
    if include_rf:
        logger.info("\n3. Training Random Forest...")
        models['random_forest'] = train_ensemble_model(
            X_train, y_train,
            model_type='random_forest'
        )
    
    if include_gb:
        logger.info("\n4. Training Gradient Boosting...")
        models['gradient_boosting'] = train_ensemble_model(
            X_train, y_train,
            model_type='gradient_boosting'
        )
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ All models trained successfully ({len(models)} models)")
    
    return models


if __name__ == "__main__":
    """
    Test the model functions if run directly.
    """
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_train['bedrooms'] = np.random.randint(1, 5, n_samples)
    
    # Create target with some relationship to features
    y_train = (
        1000 * X_train['bedrooms'] +
        500 * X_train['feature_0'] +
        300 * X_train['feature_1'] +
        np.random.normal(0, 100, n_samples) +
        100000
    )
    y_train = pd.Series(y_train, name='price')
    
    print("Test Data Created:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Price range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    
    # Test all models
    print("\n" + "=" * 60)
    print("Testing All Models")
    print("=" * 60)
    
    models = train_all_models(X_train, y_train)
    
    print("\n✓ All tests completed successfully!")
    print(f"Models trained: {list(models.keys())}")
