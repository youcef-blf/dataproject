"""
Data splitting module for the House Price Prediction Project
Master 2 DS - Machine Learning

This module contains functions for splitting data into train and test sets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
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


def split_data(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
    stratify_target: bool = True,
    n_strata: int = 5
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split labeled data into train and test sets.
    
    This function divides the dataset into training and test sets while
    ensuring similar distribution of the target variable in both sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Complete labeled DataFrame with features and target.
    test_ratio : float, default=0.2
        Proportion of data to include in test set (between 0 and 1).
    random_state : int, default=42
        Random seed for reproducibility.
    stratify_target : bool, default=True
        Whether to stratify based on target distribution.
    n_strata : int, default=5
        Number of strata for stratified split (for continuous target).
    
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test) - Training and test sets.
    
    Raises
    ------
    ValueError
        If test_ratio is not between 0 and 1.
        If target column is not found in DataFrame.
    
    Examples
    --------
    >>> X_train, y_train, X_test, y_test = split_data(df, test_ratio=0.2)
    >>> print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    Train size: 647, Test size: 162
    """
    # Validate inputs
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    
    # Check if target column exists
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in DataFrame")
    
    # Separate features and target
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    
    # Log initial info
    logger.info(f"Splitting data: {len(df)} total samples")
    logger.info(f"Test ratio: {test_ratio:.1%}")
    
    if stratify_target and len(y.unique()) > 2:
        # For continuous target, create bins for stratification
        try:
            # Create stratification bins based on target quantiles
            y_binned = pd.qcut(y, q=n_strata, labels=False, duplicates='drop')
            
            # Perform stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_ratio,
                random_state=random_state,
                stratify=y_binned
            )
            
            logger.info(f"✓ Stratified split performed with {n_strata} strata")
            
        except Exception as e:
            # Fall back to regular split if stratification fails
            logger.warning(f"Stratification failed: {e}. Using regular split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_ratio,
                random_state=random_state
            )
    else:
        # Regular split without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=random_state,
            stratify=y if stratify_target and len(y.unique()) <= 10 else None
        )
        
        logger.info("✓ Regular split performed")
    
    # Verify distribution similarity
    train_mean = y_train.mean()
    test_mean = y_test.mean()
    diff_percent = abs(train_mean - test_mean) / train_mean * 100
    
    logger.info(f"✓ Split complete:")
    logger.info(f"  - Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"  - Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    logger.info(f"  - Target mean - Train: {train_mean:,.0f}, Test: {test_mean:,.0f} ({diff_percent:.1f}% difference)")
    
    # Check for data leakage
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    if train_indices.intersection(test_indices):
        logger.warning("⚠️ Data leakage detected: Some indices appear in both train and test sets!")
    else:
        logger.info("✓ No data leakage: Train and test sets have no overlapping indices")
    
    return X_train, y_train, X_test, y_test


def save_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: Optional[Union[str, Path]] = None,
    save_combined: bool = True
) -> Dict[str, Path]:
    """
    Save train and test splits to CSV files.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    save_dir : str or Path, optional
        Directory to save files. If None, uses config.SPLITS_DIR.
    save_combined : bool, default=True
        Whether to also save combined (X + y) DataFrames.
    
    Returns
    -------
    dict
        Dictionary with paths to saved files.
    
    Examples
    --------
    >>> paths = save_splits(X_train, y_train, X_test, y_test)
    >>> print(paths['X_train'])
    data/splits/X_train.csv
    """
    if save_dir is None:
        save_dir = config.SPLITS_DIR
    else:
        save_dir = Path(save_dir)
    
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # Save individual files
    X_train_path = save_dir / "X_train.csv"
    X_train.to_csv(X_train_path, index=True)
    saved_paths['X_train'] = X_train_path
    logger.info(f"✓ Saved X_train: {X_train_path}")
    
    y_train_path = save_dir / "y_train.csv"
    y_train.to_csv(y_train_path, index=True, header=True)
    saved_paths['y_train'] = y_train_path
    logger.info(f"✓ Saved y_train: {y_train_path}")
    
    X_test_path = save_dir / "X_test.csv"
    X_test.to_csv(X_test_path, index=True)
    saved_paths['X_test'] = X_test_path
    logger.info(f"✓ Saved X_test: {X_test_path}")
    
    y_test_path = save_dir / "y_test.csv"
    y_test.to_csv(y_test_path, index=True, header=True)
    saved_paths['y_test'] = y_test_path
    logger.info(f"✓ Saved y_test: {y_test_path}")
    
    # Save combined DataFrames if requested
    if save_combined:
        train_combined = pd.concat([X_train, y_train], axis=1)
        train_path = save_dir / "train.csv"
        train_combined.to_csv(train_path, index=True)
        saved_paths['train'] = train_path
        logger.info(f"✓ Saved combined train: {train_path}")
        
        test_combined = pd.concat([X_test, y_test], axis=1)
        test_path = save_dir / "test.csv"
        test_combined.to_csv(test_path, index=True)
        saved_paths['test'] = test_path
        logger.info(f"✓ Saved combined test: {test_path}")
    
    return saved_paths


def load_splits(
    load_dir: Optional[Union[str, Path]] = None,
    load_combined: bool = False
) -> Union[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], 
           Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load previously saved train and test splits.
    
    Parameters
    ----------
    load_dir : str or Path, optional
        Directory containing split files. If None, uses config.SPLITS_DIR.
    load_combined : bool, default=False
        Whether to load combined DataFrames instead of separate X and y.
    
    Returns
    -------
    tuple
        If load_combined=False: (X_train, y_train, X_test, y_test)
        If load_combined=True: (train_df, test_df)
    
    Examples
    --------
    >>> X_train, y_train, X_test, y_test = load_splits()
    >>> print(f"Loaded train size: {len(X_train)}")
    """
    if load_dir is None:
        load_dir = config.SPLITS_DIR
    else:
        load_dir = Path(load_dir)
    
    if load_combined:
        # Load combined files
        train_path = load_dir / "train.csv"
        test_path = load_dir / "test.csv"
        
        train_df = pd.read_csv(train_path, index_col=0)
        test_df = pd.read_csv(test_path, index_col=0)
        
        logger.info(f"✓ Loaded combined splits: train={len(train_df)}, test={len(test_df)}")
        
        return train_df, test_df
    else:
        # Load separate files
        X_train = pd.read_csv(load_dir / "X_train.csv", index_col=0)
        y_train = pd.read_csv(load_dir / "y_train.csv", index_col=0).squeeze()
        X_test = pd.read_csv(load_dir / "X_test.csv", index_col=0)
        y_test = pd.read_csv(load_dir / "y_test.csv", index_col=0).squeeze()
        
        logger.info(f"✓ Loaded splits: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return X_train, y_train, X_test, y_test


def get_train_test_info(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Get summary statistics comparing train and test sets.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    
    Returns
    -------
    pd.DataFrame
        Summary statistics for train and test sets.
    
    Examples
    --------
    >>> info = get_train_test_info(X_train, y_train, X_test, y_test)
    >>> print(info)
    """
    # Basic statistics
    info = pd.DataFrame({
        'Train Size': [len(X_train)],
        'Test Size': [len(X_test)],
        'Train %': [len(X_train) / (len(X_train) + len(X_test)) * 100],
        'Test %': [len(X_test) / (len(X_train) + len(X_test)) * 100],
        'N Features': [X_train.shape[1]],
        'Target Mean (Train)': [y_train.mean()],
        'Target Mean (Test)': [y_test.mean()],
        'Target Std (Train)': [y_train.std()],
        'Target Std (Test)': [y_test.std()],
        'Target Min (Train)': [y_train.min()],
        'Target Min (Test)': [y_test.min()],
        'Target Max (Train)': [y_train.max()],
        'Target Max (Test)': [y_test.max()],
    })
    
    return info.T


def create_validation_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create cross-validation splits from training data.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    n_splits : int, default=5
        Number of cross-validation folds.
    random_state : int, default=42
        Random seed.
    
    Returns
    -------
    list
        List of (train_indices, val_indices) tuples for each fold.
    
    Examples
    --------
    >>> cv_splits = create_validation_splits(X_train, y_train, n_splits=5)
    >>> print(f"Number of folds: {len(cv_splits)}")
    Number of folds: 5
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, val_idx in kf.split(X_train):
        splits.append((train_idx, val_idx))
    
    logger.info(f"✓ Created {n_splits} cross-validation splits")
    
    # Verify split sizes
    for i, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
    
    return splits


def check_target_distribution(
    y_train: pd.Series,
    y_test: pd.Series,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Compare target distribution between train and test sets.
    
    Parameters
    ----------
    y_train : pd.Series
        Training target values.
    y_test : pd.Series
        Test target values.
    n_bins : int, default=10
        Number of bins for histogram comparison.
    
    Returns
    -------
    pd.DataFrame
        Distribution comparison statistics.
    
    Examples
    --------
    >>> dist_info = check_target_distribution(y_train, y_test)
    >>> print(dist_info)
    """
    # Create bins based on combined data
    y_combined = pd.concat([y_train, y_test])
    bins = np.linspace(y_combined.min(), y_combined.max(), n_bins + 1)
    
    # Calculate distributions
    train_hist, _ = np.histogram(y_train, bins=bins)
    test_hist, _ = np.histogram(y_test, bins=bins)
    
    # Normalize to percentages
    train_pct = train_hist / len(y_train) * 100
    test_pct = test_hist / len(y_test) * 100
    
    # Create comparison DataFrame
    dist_df = pd.DataFrame({
        'Bin Range': [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(n_bins)],
        'Train %': train_pct,
        'Test %': test_pct,
        'Difference': train_pct - test_pct
    })
    
    # Calculate similarity score (1 - mean absolute difference)
    similarity = 1 - np.mean(np.abs(dist_df['Difference'])) / 100
    
    logger.info(f"✓ Target distribution similarity: {similarity:.2%}")
    
    return dist_df


if __name__ == "__main__":
    """
    Test the splitting functions if run directly.
    """
    # Create test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'price': np.random.uniform(100000, 500000, 100)
    })
    
    print("Test Data:")
    print(test_data.head())
    print(f"\nShape: {test_data.shape}")
    
    # Test split_data function
    X_train, y_train, X_test, y_test = split_data(test_data, test_ratio=0.2)
    
    print(f"\n✓ Split Results:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Test distribution check
    dist_info = check_target_distribution(y_train, y_test, n_bins=5)
    print("\n✓ Target Distribution Comparison:")
    print(dist_info)
    
    # Test train/test info
    info = get_train_test_info(X_train, y_train, X_test, y_test)
    print("\n✓ Train/Test Summary:")
    print(info)
