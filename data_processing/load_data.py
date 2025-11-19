"""
Data loading module for the House Price Prediction Project
Master 2 DS - Machine Learning

This module contains functions for loading and initial inspection of data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

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


def load_data(
    filepath: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load housing data from CSV file.
    
    This function loads the housing dataset and optionally displays
    basic information about it.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. If None, uses the path from config.
    verbose : bool, default=True
        If True, prints basic information about the loaded data.
    
    Returns
    -------
    pd.DataFrame
        The loaded housing dataset.
    
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file is empty or cannot be parsed.
    
    Examples
    --------
    >>> df = load_data()
    ✓ Data loaded successfully from data/raw/house_prices.csv
    Shape: (809, 14)
    
    >>> df = load_data("custom_path.csv", verbose=False)
    """
    # Use default path if not provided
    if filepath is None:
        filepath = config.RAW_DATA_PATH
    else:
        filepath = Path(filepath)
    
    # Check if file exists
    if not filepath.exists():
        error_msg = f"File not found: {filepath}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Load the data
        logger.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Check if dataframe is empty
        if df.empty:
            raise ValueError("The loaded dataframe is empty")
        
        # Success message
        logger.info(f"✓ Data loaded successfully from {filepath}")
        
        if verbose:
            print(f"✓ Data loaded successfully from {filepath}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        
        return df
        
    except pd.errors.EmptyDataError:
        error_msg = "The file is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing the CSV file: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    except Exception as e:
        error_msg = f"Unexpected error loading data: {str(e)}"
        logger.error(error_msg)
        raise


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    
    Returns
    -------
    dict
        Dictionary containing various statistics about the data.
    
    Examples
    --------
    >>> info = get_data_info(df)
    >>> print(info['shape'])
    (809, 14)
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 ** 2,
        'duplicates': df.duplicated().sum(),
        'numerical_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns)
    }
    
    return info


def display_data_info(df: pd.DataFrame, detailed: bool = True) -> None:
    """
    Display comprehensive information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    detailed : bool, default=True
        If True, shows detailed statistics for all columns.
    
    Examples
    --------
    >>> display_data_info(df)
    """
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    info = get_data_info(df)
    
    print(f"\n📊 Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
    print(f"💾 Memory usage: {info['memory_usage_mb']:.2f} MB")
    print(f"🔄 Duplicate rows: {info['duplicates']}")
    
    print("\n📝 Column Types:")
    print(f"  - Numerical: {len(info['numerical_columns'])} columns")
    print(f"  - Categorical: {len(info['categorical_columns'])} columns")
    
    if detailed:
        print("\n🔍 Detailed Column Information:")
        print("-" * 40)
        
        col_info_df = pd.DataFrame({
            'Type': pd.Series(info['dtypes']),
            'Missing': pd.Series(info['missing_values']),
            'Missing %': pd.Series(info['missing_percentage'])
        })
        print(col_info_df.to_string())
        
        print("\n📊 Numerical Columns Statistics:")
        print("-" * 40)
        print(df[info['numerical_columns']].describe().round(2).to_string())
        
        print("\n🏷️ Categorical Columns Unique Values:")
        print("-" * 40)
        for col in info['categorical_columns']:
            n_unique = df[col].nunique()
            print(f"{col}: {n_unique} unique values")
            if n_unique <= 10:
                value_counts = df[col].value_counts()
                for val, count in value_counts.head(5).items():
                    print(f"  - {val}: {count} ({count/len(df)*100:.1f}%)")
                if n_unique > 5:
                    print(f"  ... and {n_unique - 5} more")
    
    print("\n" + "=" * 60)


def check_data_quality(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Check data quality and identify potential issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check.
    
    Returns
    -------
    tuple
        (is_clean, issues) where is_clean is True if no issues found,
        and issues is a list of identified problems.
    
    Examples
    --------
    >>> is_clean, issues = check_data_quality(df)
    >>> if not is_clean:
    ...     for issue in issues:
    ...         print(f"⚠️ {issue}")
    """
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    high_missing = missing[missing > len(df) * 0.5]
    if not high_missing.empty:
        for col, count in high_missing.items():
            pct = count / len(df) * 100
            issues.append(f"Column '{col}' has {pct:.1f}% missing values")
    
    # Check for duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        issues.append(f"Found {n_duplicates} duplicate rows")
    
    # Check for inconsistent column names
    inconsistent_cols = []
    for col in df.columns:
        if col != col.strip():
            inconsistent_cols.append(f"'{col}' has extra spaces")
        if any(c.isupper() for c in col):
            inconsistent_cols.append(f"'{col}' has uppercase letters")
        if ' ' in col:
            inconsistent_cols.append(f"'{col}' contains spaces")
    
    if inconsistent_cols:
        issues.append(f"Inconsistent column names: {', '.join(set(inconsistent_cols))}")
    
    # Check target variable
    if config.TARGET_COLUMN in df.columns or 'price' in df.columns.str.lower():
        target_col = config.TARGET_COLUMN if config.TARGET_COLUMN in df.columns else 'price'
        if df[target_col].isnull().any():
            n_missing = df[target_col].isnull().sum()
            issues.append(f"Target variable '{target_col}' has {n_missing} missing values")
        
        # Check for negative prices
        if df[target_col].dtype in [np.float64, np.int64]:
            if (df[target_col] <= 0).any():
                n_invalid = (df[target_col] <= 0).sum()
                issues.append(f"Target variable '{target_col}' has {n_invalid} non-positive values")
    
    # Check for mixed types in categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        unique_types = df[col].dropna().apply(type).unique()
        if len(unique_types) > 1:
            issues.append(f"Column '{col}' has mixed data types")
    
    is_clean = len(issues) == 0
    
    return is_clean, issues


def get_initial_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate initial statistical summary of the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to analyze.
    
    Returns
    -------
    pd.DataFrame
        Statistical summary including custom metrics.
    
    Examples
    --------
    >>> stats = get_initial_statistics(df)
    >>> print(stats)
    """
    # Basic statistics
    stats = df.describe(include='all').T
    
    # Add additional statistics
    stats['missing_count'] = df.isnull().sum()
    stats['missing_pct'] = (stats['missing_count'] / len(df) * 100).round(2)
    stats['dtype'] = df.dtypes
    
    # For numerical columns, add skewness and kurtosis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        stats.loc[numerical_cols, 'skewness'] = df[numerical_cols].skew()
        stats.loc[numerical_cols, 'kurtosis'] = df[numerical_cols].kurtosis()
    
    return stats


if __name__ == "__main__":
    """
    Test the loading functions if run directly.
    """
    # Test with a sample file path
    try:
        # Create a test dataframe
        test_data = pd.DataFrame({
            'price': [100000, 200000, 150000],
            'area': [1000, 1500, 1200],
            'bedrooms': [2, 3, 2]
        })
        
        # Save temporarily
        test_path = Path("/tmp/test_housing.csv")
        test_data.to_csv(test_path, index=False)
        
        # Test loading
        df = load_data(test_path, verbose=True)
        
        # Test info functions
        display_data_info(df)
        
        # Check quality
        is_clean, issues = check_data_quality(df)
        print(f"\n✓ Data quality check: {'PASSED' if is_clean else 'ISSUES FOUND'}")
        if not is_clean:
            for issue in issues:
                print(f"  ⚠️ {issue}")
        
        # Clean up
        test_path.unlink()
        
    except Exception as e:
        print(f"Error in test: {e}")
