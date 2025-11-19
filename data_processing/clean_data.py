"""
Data cleaning module for the House Price Prediction Project
Master 2 DS - Machine Learning

This module contains functions for cleaning and formatting the housing dataset.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
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


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase with underscores.
    
    This function converts all column names to lowercase, replaces spaces
    with underscores, and applies the mapping from config.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with columns to standardize.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names.
    
    Examples
    --------
    >>> df = pd.DataFrame({'Price': [100], 'Area ': [50]})
    >>> df_clean = standardize_column_names(df)
    >>> print(df_clean.columns.tolist())
    ['price', 'area']
    """
    df_copy = df.copy()
    
    # First, clean up basic issues
    df_copy.columns = df_copy.columns.str.strip()  # Remove leading/trailing spaces
    df_copy.columns = df_copy.columns.str.replace(' ', '_')  # Replace spaces with underscores
    df_copy.columns = df_copy.columns.str.lower()  # Convert to lowercase
    
    # Apply specific mappings from config if available
    if hasattr(config, 'COLUMN_MAPPING'):
        # Create reverse mapping for current names
        current_mapping = {}
        for old_name, new_name in config.COLUMN_MAPPING.items():
            # Try different variations of the column name
            for col in df_copy.columns:
                if col.lower().replace('_', '').replace(' ', '') == old_name.lower().replace('_', '').replace(' ', ''):
                    current_mapping[col] = new_name
                    break
        
        if current_mapping:
            df_copy = df_copy.rename(columns=current_mapping)
            logger.info(f"Applied column mappings: {current_mapping}")
    
    logger.info(f"✓ Column names standardized: {df_copy.columns.tolist()}")
    
    return df_copy


def clean_furnishing_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize furnishing status values.
    
    Unifies different cases (furnished, FURNISHED, Furnished) to lowercase.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing furnishing_status column.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned furnishing status values.
    
    Examples
    --------
    >>> df = pd.DataFrame({'furnishing_status': ['FURNISHED', 'furnished', 'Furnished']})
    >>> df_clean = clean_furnishing_status(df)
    >>> print(df_clean['furnishing_status'].unique())
    ['furnished']
    """
    df_copy = df.copy()
    
    if 'furnishing_status' in df_copy.columns:
        # First, handle missing values
        df_copy['furnishing_status'] = df_copy['furnishing_status'].fillna('unknown')
        
        # Convert to string and strip whitespace
        df_copy['furnishing_status'] = df_copy['furnishing_status'].astype(str).str.strip()
        
        # Apply mapping from config
        if hasattr(config, 'FURNISHING_STATUS_MAPPING'):
            df_copy['furnishing_status'] = df_copy['furnishing_status'].map(
                lambda x: config.FURNISHING_STATUS_MAPPING.get(x, x.lower())
            )
        else:
            # Default: convert to lowercase
            df_copy['furnishing_status'] = df_copy['furnishing_status'].str.lower()
        
        # Log the unique values after cleaning
        unique_vals = df_copy['furnishing_status'].unique()
        logger.info(f"✓ Furnishing status cleaned. Unique values: {unique_vals.tolist()}")
    else:
        logger.warning("Column 'furnishing_status' not found in DataFrame")
    
    return df_copy


def clean_binary_columns(df: pd.DataFrame, binary_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean and standardize binary (Yes/No) columns.
    
    Converts Yes/No values to lowercase and handles missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing binary columns.
    binary_cols : list of str, optional
        List of binary column names. If None, uses config.BINARY_COLUMNS.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned binary columns.
    
    Examples
    --------
    >>> df = pd.DataFrame({'mainroad': ['Yes', 'NO', 'yes']})
    >>> df_clean = clean_binary_columns(df, ['mainroad'])
    >>> print(df_clean['mainroad'].unique())
    ['yes' 'no']
    """
    df_copy = df.copy()
    
    if binary_cols is None:
        binary_cols = config.BINARY_COLUMNS if hasattr(config, 'BINARY_COLUMNS') else []
    
    for col in binary_cols:
        if col in df_copy.columns:
            # Convert to string, strip whitespace, and lowercase
            df_copy[col] = df_copy[col].astype(str).str.strip().str.lower()
            
            # Replace common variations
            df_copy[col] = df_copy[col].replace({
                'yes': 'yes',
                'y': 'yes',
                'true': 'yes',
                '1': 'yes',
                'no': 'no',
                'n': 'no',
                'false': 'no',
                '0': 'no',
                'nan': np.nan,
                'none': np.nan
            })
            
            # Count missing values
            n_missing = df_copy[col].isnull().sum()
            if n_missing > 0:
                logger.info(f"  Column '{col}': {n_missing} missing values")
    
    logger.info(f"✓ Binary columns cleaned: {[col for col in binary_cols if col in df_copy.columns]}")
    
    return df_copy


def handle_missing_values(
    df: pd.DataFrame,
    strategy: Dict[str, str] = None,
    threshold: float = 0.95
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Handle missing values in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with missing values.
    strategy : dict, optional
        Dictionary mapping column names to strategies:
        - 'drop': Drop the column
        - 'mean': Fill with mean (numerical only)
        - 'median': Fill with median (numerical only)
        - 'mode': Fill with mode
        - 'forward_fill': Forward fill
        - 'backward_fill': Backward fill
        - 'constant_VALUE': Fill with VALUE
    threshold : float, default=0.95
        Threshold for dropping columns (drop if more than threshold missing).
    
    Returns
    -------
    tuple
        (cleaned_df, actions_taken) where actions_taken documents what was done.
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, np.nan], 'b': [np.nan, np.nan, np.nan]})
    >>> df_clean, actions = handle_missing_values(df, threshold=0.5)
    >>> print(actions)
    {'b': 'dropped (100.0% missing)'}
    """
    df_copy = df.copy()
    actions_taken = {}
    
    # Calculate missing percentages
    missing_pct = (df_copy.isnull().sum() / len(df_copy))
    
    # First, handle columns that should be dropped due to high missing percentage
    cols_to_drop = []
    for col in df_copy.columns:
        if missing_pct[col] > threshold:
            cols_to_drop.append(col)
            actions_taken[col] = f"dropped ({missing_pct[col]*100:.1f}% missing)"
    
    # Drop columns with too many missing values
    if cols_to_drop:
        df_copy = df_copy.drop(columns=cols_to_drop)
        logger.info(f"✓ Dropped columns with >{threshold*100}% missing: {cols_to_drop}")
    
    # Apply strategies for remaining columns with missing values
    if strategy is None:
        strategy = {}
    
    remaining_missing = df_copy.isnull().sum()
    for col in df_copy.columns:
        if remaining_missing[col] > 0:
            col_strategy = strategy.get(col, 'auto')
            
            if col_strategy == 'drop':
                df_copy = df_copy.drop(columns=[col])
                actions_taken[col] = "dropped (by strategy)"
                
            elif col_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_copy[col]):
                mean_val = df_copy[col].mean()
                df_copy[col].fillna(mean_val, inplace=True)
                actions_taken[col] = f"filled with mean ({mean_val:.2f})"
                
            elif col_strategy == 'median' and pd.api.types.is_numeric_dtype(df_copy[col]):
                median_val = df_copy[col].median()
                df_copy[col].fillna(median_val, inplace=True)
                actions_taken[col] = f"filled with median ({median_val:.2f})"
                
            elif col_strategy == 'mode' or col_strategy == 'auto':
                if not df_copy[col].empty:
                    mode_val = df_copy[col].mode()
                    if len(mode_val) > 0:
                        df_copy[col].fillna(mode_val[0], inplace=True)
                        actions_taken[col] = f"filled with mode ({mode_val[0]})"
                        
            elif col_strategy == 'forward_fill':
                df_copy[col].fillna(method='ffill', inplace=True)
                actions_taken[col] = "forward filled"
                
            elif col_strategy == 'backward_fill':
                df_copy[col].fillna(method='bfill', inplace=True)
                actions_taken[col] = "backward filled"
                
            elif col_strategy.startswith('constant_'):
                const_val = col_strategy.replace('constant_', '')
                df_copy[col].fillna(const_val, inplace=True)
                actions_taken[col] = f"filled with constant ({const_val})"
    
    # Log summary
    remaining_missing_total = df_copy.isnull().sum().sum()
    logger.info(f"✓ Missing values handled. Remaining: {remaining_missing_total}")
    
    return df_copy, actions_taken


def remove_duplicates(df: pd.DataFrame, keep: str = 'first') -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing duplicates.
    keep : {'first', 'last', False}, default='first'
        - 'first': Keep the first occurrence
        - 'last': Keep the last occurrence  
        - False: Drop all duplicates
    
    Returns
    -------
    tuple
        (cleaned_df, n_removed) where n_removed is the number of rows removed.
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 1, 2], 'b': [2, 2, 3]})
    >>> df_clean, n_removed = remove_duplicates(df)
    >>> print(f"Removed {n_removed} duplicates")
    Removed 1 duplicates
    """
    initial_shape = df.shape[0]
    df_copy = df.drop_duplicates(keep=keep)
    n_removed = initial_shape - df_copy.shape[0]
    
    if n_removed > 0:
        logger.info(f"✓ Removed {n_removed} duplicate rows")
    else:
        logger.info("✓ No duplicate rows found")
    
    return df_copy, n_removed


def remove_invalid_target_rows(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    remove_negative: bool = True,
    remove_zero: bool = False
) -> Tuple[pd.DataFrame, int]:
    """
    Remove rows with invalid target values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target column.
    target_col : str, optional
        Name of the target column. If None, uses config.TARGET_COLUMN.
    remove_negative : bool, default=True
        Whether to remove rows with negative target values.
    remove_zero : bool, default=False
        Whether to remove rows with zero target values.
    
    Returns
    -------
    tuple
        (cleaned_df, n_removed) where n_removed is the number of rows removed.
    
    Examples
    --------
    >>> df = pd.DataFrame({'price': [100, -50, 0, 200]})
    >>> df_clean, n_removed = remove_invalid_target_rows(df, 'price')
    >>> print(f"Removed {n_removed} invalid rows")
    Removed 1 invalid rows
    """
    df_copy = df.copy()
    
    if target_col is None:
        target_col = config.TARGET_COLUMN
    
    if target_col not in df_copy.columns:
        logger.warning(f"Target column '{target_col}' not found in DataFrame")
        return df_copy, 0
    
    initial_shape = df_copy.shape[0]
    
    # Remove rows with missing target
    df_copy = df_copy[df_copy[target_col].notna()]
    
    # Remove negative values if requested
    if remove_negative and pd.api.types.is_numeric_dtype(df_copy[target_col]):
        df_copy = df_copy[df_copy[target_col] > 0] if remove_zero else df_copy[df_copy[target_col] >= 0]
    
    n_removed = initial_shape - df_copy.shape[0]
    
    if n_removed > 0:
        logger.info(f"✓ Removed {n_removed} rows with invalid target values")
    
    return df_copy, n_removed


def clean_data_pipeline(
    df: pd.DataFrame,
    standardize_columns: bool = True,
    clean_furnishing: bool = True,
    clean_binary: bool = True,
    handle_missing: bool = True,
    remove_duplicates_flag: bool = True,
    remove_invalid_target: bool = True,
    missing_threshold: float = 0.95,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Complete data cleaning pipeline.
    
    Applies all cleaning steps in sequence and returns cleaned DataFrame
    along with a summary of actions taken.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame to clean.
    standardize_columns : bool, default=True
        Whether to standardize column names.
    clean_furnishing : bool, default=True
        Whether to clean furnishing status values.
    clean_binary : bool, default=True
        Whether to clean binary columns.
    handle_missing : bool, default=True
        Whether to handle missing values.
    remove_duplicates_flag : bool, default=True
        Whether to remove duplicate rows.
    remove_invalid_target : bool, default=True
        Whether to remove rows with invalid target values.
    missing_threshold : float, default=0.95
        Threshold for dropping columns with missing values.
    verbose : bool, default=True
        Whether to print progress information.
    
    Returns
    -------
    tuple
        (cleaned_df, cleaning_report) where cleaning_report contains
        information about all cleaning actions taken.
    
    Examples
    --------
    >>> df_clean, report = clean_data_pipeline(df)
    >>> print(f"Final shape: {df_clean.shape}")
    >>> print(f"Rows removed: {report['rows_removed']}")
    """
    cleaning_report = {
        'initial_shape': df.shape,
        'steps_performed': [],
        'rows_removed': 0,
        'columns_removed': 0,
        'missing_values_handled': {},
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("DATA CLEANING PIPELINE")
        print("=" * 60)
        print(f"Initial shape: {df.shape}")
    
    df_clean = df.copy()
    initial_rows = df_clean.shape[0]
    initial_cols = df_clean.shape[1]
    
    # Step 1: Standardize column names
    if standardize_columns:
        df_clean = standardize_column_names(df_clean)
        cleaning_report['steps_performed'].append('standardize_columns')
        if verbose:
            print("✓ Column names standardized")
    
    # Step 2: Remove invalid target rows first (before other cleaning)
    if remove_invalid_target:
        df_clean, n_removed = remove_invalid_target_rows(df_clean)
        cleaning_report['rows_removed'] += n_removed
        cleaning_report['steps_performed'].append('remove_invalid_target')
        if verbose:
            print(f"✓ Removed {n_removed} rows with invalid target values")
    
    # Step 3: Clean furnishing status
    if clean_furnishing:
        df_clean = clean_furnishing_status(df_clean)
        cleaning_report['steps_performed'].append('clean_furnishing')
        if verbose:
            print("✓ Furnishing status cleaned")
    
    # Step 4: Clean binary columns
    if clean_binary:
        df_clean = clean_binary_columns(df_clean)
        cleaning_report['steps_performed'].append('clean_binary')
        if verbose:
            print("✓ Binary columns cleaned")
    
    # Step 5: Handle missing values
    if handle_missing:
        df_clean, missing_actions = handle_missing_values(df_clean, threshold=missing_threshold)
        cleaning_report['missing_values_handled'] = missing_actions
        cleaning_report['steps_performed'].append('handle_missing')
        
        # Count dropped columns
        cols_dropped = sum(1 for action in missing_actions.values() if 'dropped' in action)
        cleaning_report['columns_removed'] += cols_dropped
        
        if verbose:
            print(f"✓ Missing values handled ({len(missing_actions)} columns affected)")
    
    # Step 6: Remove duplicates
    if remove_duplicates_flag:
        df_clean, n_removed = remove_duplicates(df_clean)
        cleaning_report['rows_removed'] += n_removed
        cleaning_report['steps_performed'].append('remove_duplicates')
        if verbose:
            print(f"✓ Removed {n_removed} duplicate rows")
    
    # Final statistics
    cleaning_report['final_shape'] = df_clean.shape
    cleaning_report['columns_removed'] = initial_cols - df_clean.shape[1]
    
    if verbose:
        print("\n" + "-" * 40)
        print("CLEANING SUMMARY:")
        print(f"Initial shape: {cleaning_report['initial_shape']}")
        print(f"Final shape: {cleaning_report['final_shape']}")
        print(f"Rows removed: {cleaning_report['rows_removed']}")
        print(f"Columns removed: {cleaning_report['columns_removed']}")
        print("=" * 60)
    
    return df_clean, cleaning_report


if __name__ == "__main__":
    """
    Test the cleaning functions if run directly.
    """
    # Create test data
    test_data = pd.DataFrame({
        'Price': [100000, 200000, -50000, 200000, np.nan],
        'AreA': [1000, 1500, 1200, 1500, 800],
        'bedrooms': [2, 3, 2, 3, np.nan],
        'mainroad': ['Yes', 'no', 'YES', 'no', np.nan],
        'furnishing STATUS': ['FURNISHED', 'semi-furnished', 'Furnished', 'unfurnished', np.nan],
        'houSeaGe': [np.nan, np.nan, np.nan, np.nan, np.nan]  # 100% missing
    })
    
    print("Original Data:")
    print(test_data)
    print("\nMissing values:")
    print(test_data.isnull().sum())
    
    # Test pipeline
    df_clean, report = clean_data_pipeline(test_data)
    
    print("\nCleaned Data:")
    print(df_clean)
    print("\nCleaning Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
