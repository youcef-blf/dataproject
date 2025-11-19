"""
Data preprocessing module for the House Price Prediction Project
Master 2 DS - Machine Learning

This module contains functions for encoding variables and preparing data for modeling.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
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


def encode_binary_columns(
    df: pd.DataFrame,
    binary_cols: Optional[List[str]] = None,
    true_value: str = 'yes',
    false_value: str = 'no'
) -> pd.DataFrame:
    """
    Encode binary columns to 0/1.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with binary columns.
    binary_cols : list of str, optional
        List of binary columns to encode. If None, uses config.BINARY_COLUMNS.
    true_value : str, default='yes'
        Value to encode as 1.
    false_value : str, default='no'
        Value to encode as 0.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with encoded binary columns.
    
    Examples
    --------
    >>> df = pd.DataFrame({'mainroad': ['yes', 'no', 'yes']})
    >>> df_encoded = encode_binary_columns(df, ['mainroad'])
    >>> print(df_encoded['mainroad'].values)
    [1 0 1]
    """
    df_copy = df.copy()
    
    if binary_cols is None:
        binary_cols = config.BINARY_COLUMNS if hasattr(config, 'BINARY_COLUMNS') else []
    
    for col in binary_cols:
        if col in df_copy.columns:
            # Create binary encoding
            df_copy[col] = df_copy[col].map({
                true_value: 1,
                false_value: 0
            })
            
            # Check for unmapped values
            if df_copy[col].isnull().any():
                unmapped = df[col][df_copy[col].isnull()].unique()
                logger.warning(f"Column '{col}' has unmapped values: {unmapped}")
                # Fill unmapped with mode or 0
                mode_val = df_copy[col].mode()
                if len(mode_val) > 0:
                    df_copy[col].fillna(mode_val[0], inplace=True)
                else:
                    df_copy[col].fillna(0, inplace=True)
            
            # Convert to integer
            df_copy[col] = df_copy[col].astype(int)
    
    logger.info(f"✓ Binary columns encoded: {[col for col in binary_cols if col in df_copy.columns]}")
    
    return df_copy


def encode_categorical_columns(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    encoding_type: str = 'onehot',
    handle_unknown: str = 'ignore'
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Encode categorical columns using OneHot or Label encoding.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with categorical columns.
    categorical_cols : list of str, optional
        List of categorical columns to encode.
    encoding_type : {'onehot', 'label'}, default='onehot'
        Type of encoding to use.
    handle_unknown : {'error', 'ignore'}, default='ignore'
        How to handle unknown categories during transform.
    
    Returns
    -------
    tuple
        (encoded_df, encoders) where encoders is a dict of fitted encoders.
    
    Examples
    --------
    >>> df = pd.DataFrame({'status': ['furnished', 'unfurnished', 'semi-furnished']})
    >>> df_encoded, encoders = encode_categorical_columns(df, ['status'])
    >>> print(df_encoded.columns.tolist())
    ['status_furnished', 'status_semi-furnished', 'status_unfurnished']
    """
    df_copy = df.copy()
    encoders = {}
    
    if categorical_cols is None:
        categorical_cols = config.CATEGORICAL_COLUMNS if hasattr(config, 'CATEGORICAL_COLUMNS') else []
    
    for col in categorical_cols:
        if col not in df_copy.columns:
            continue
            
        if encoding_type == 'onehot':
            # One-hot encoding
            # Get unique values excluding NaN
            unique_vals = df_copy[col].dropna().unique()
            
            # Create dummy variables
            dummies = pd.get_dummies(
                df_copy[col],
                prefix=col,
                dummy_na=False  # Don't create column for NaN
            )
            
            # Drop original column and add dummy columns
            df_copy = df_copy.drop(columns=[col])
            df_copy = pd.concat([df_copy, dummies], axis=1)
            
            # Store encoder info for later use
            encoders[col] = {
                'type': 'onehot',
                'categories': unique_vals.tolist(),
                'columns': dummies.columns.tolist()
            }
            
            logger.info(f"✓ One-hot encoded '{col}': {len(unique_vals)} categories → {len(dummies.columns)} columns")
            
        elif encoding_type == 'label':
            # Label encoding
            le = LabelEncoder()
            
            # Handle missing values
            mask = df_copy[col].notna()
            df_copy.loc[mask, col] = le.fit_transform(df_copy.loc[mask, col])
            df_copy[col] = df_copy[col].fillna(-1).astype(int)  # -1 for missing values
            
            encoders[col] = {
                'type': 'label',
                'encoder': le,
                'classes': le.classes_.tolist()
            }
            
            logger.info(f"✓ Label encoded '{col}': {len(le.classes_)} categories")
    
    return df_copy, encoders


def scale_numerical_features(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    scaling_type: str = 'standard',
    exclude_target: bool = True
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Scale numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numerical features.
    numerical_cols : list of str, optional
        List of numerical columns to scale. If None, detects automatically.
    scaling_type : {'standard', 'minmax', 'none'}, default='standard'
        Type of scaling to apply.
    exclude_target : bool, default=True
        Whether to exclude the target column from scaling.
    
    Returns
    -------
    tuple
        (scaled_df, scalers) where scalers is a dict of fitted scalers.
    
    Examples
    --------
    >>> df = pd.DataFrame({'area': [1000, 2000, 1500], 'price': [100, 200, 150]})
    >>> df_scaled, scalers = scale_numerical_features(df, ['area'])
    >>> print(df_scaled['area'].mean())  # Should be close to 0 for standard scaling
    0.0
    """
    df_copy = df.copy()
    scalers = {}
    
    if scaling_type == 'none':
        logger.info("No scaling applied")
        return df_copy, scalers
    
    # Determine numerical columns if not specified
    if numerical_cols is None:
        numerical_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target if requested
        if exclude_target and config.TARGET_COLUMN in numerical_cols:
            numerical_cols.remove(config.TARGET_COLUMN)
    
    if len(numerical_cols) == 0:
        logger.warning("No numerical columns to scale")
        return df_copy, scalers
    
    # Choose scaler
    if scaling_type == 'standard':
        scaler = StandardScaler()
    elif scaling_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling type: {scaling_type}")
    
    # Fit and transform
    df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])
    
    # Store scaler info
    scalers['numerical'] = {
        'type': scaling_type,
        'scaler': scaler,
        'columns': numerical_cols
    }
    
    logger.info(f"✓ Applied {scaling_type} scaling to {len(numerical_cols)} columns")
    
    return df_copy, scalers


def create_feature_interactions(
    df: pd.DataFrame,
    interactions: Optional[List[Tuple[str, str]]] = None,
    include_ratios: bool = True,
    include_products: bool = True
) -> pd.DataFrame:
    """
    Create feature interactions and engineered features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features.
    interactions : list of tuples, optional
        List of (col1, col2) pairs for interactions.
    include_ratios : bool, default=True
        Whether to create ratio features.
    include_products : bool, default=True
        Whether to create product features.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional interaction features.
    
    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> df_new = create_feature_interactions(df, [('a', 'b')])
    >>> print('a_times_b' in df_new.columns)
    True
    """
    df_copy = df.copy()
    
    # Default interactions if not specified
    if interactions is None:
        interactions = []
        
        # Add some domain-specific interactions
        if 'area' in df_copy.columns and 'bedrooms' in df_copy.columns:
            interactions.append(('area', 'bedrooms'))
        
        if 'bathrooms' in df_copy.columns and 'bedrooms' in df_copy.columns:
            interactions.append(('bathrooms', 'bedrooms'))
        
        if 'area' in df_copy.columns and 'prefarea' in df_copy.columns:
            interactions.append(('area', 'prefarea'))
    
    # Create interactions
    for col1, col2 in interactions:
        if col1 in df_copy.columns and col2 in df_copy.columns:
            # Check if columns are numerical
            if pd.api.types.is_numeric_dtype(df_copy[col1]) and pd.api.types.is_numeric_dtype(df_copy[col2]):
                
                if include_products:
                    df_copy[f'{col1}_times_{col2}'] = df_copy[col1] * df_copy[col2]
                    logger.info(f"✓ Created product feature: {col1}_times_{col2}")
                
                if include_ratios and (df_copy[col2] != 0).all():
                    df_copy[f'{col1}_per_{col2}'] = df_copy[col1] / df_copy[col2].replace(0, np.nan)
                    logger.info(f"✓ Created ratio feature: {col1}_per_{col2}")
    
    # Additional engineered features
    if 'area' in df_copy.columns and 'price' in df_copy.columns:
        # Price per square foot (useful for analysis but not for prediction)
        if (df_copy['area'] != 0).all():
            df_copy['price_per_sqft'] = df_copy['price'] / df_copy['area'].replace(0, np.nan)
            logger.info("✓ Created price_per_sqft feature")
    
    # Total amenities score
    amenity_cols = ['mainroad', 'guestroom', 'basement', 'hot_water_heating', 
                    'airconditioning', 'prefarea']
    amenity_cols = [col for col in amenity_cols if col in df_copy.columns]
    
    if amenity_cols:
        # Check if columns are binary (0/1)
        all_binary = all(
            df_copy[col].dropna().isin([0, 1]).all() 
            for col in amenity_cols
        )
        
        if all_binary:
            df_copy['amenity_score'] = df_copy[amenity_cols].sum(axis=1)
            logger.info(f"✓ Created amenity_score from {len(amenity_cols)} features")
    
    return df_copy


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    drop_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Complete DataFrame.
    target_col : str, optional
        Name of target column. If None, uses config.TARGET_COLUMN.
    drop_cols : list of str, optional
        Additional columns to drop from features.
    
    Returns
    -------
    tuple
        (X, y) where X is features DataFrame and y is target Series.
    
    Examples
    --------
    >>> df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'target': [5, 6]})
    >>> X, y = prepare_features_and_target(df, 'target')
    >>> print(X.columns.tolist())
    ['feature1', 'feature2']
    """
    if target_col is None:
        target_col = config.TARGET_COLUMN
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Separate target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    
    # Drop additional columns if specified
    if drop_cols:
        cols_to_drop = [col for col in drop_cols if col in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns: {cols_to_drop}")
    
    # Drop columns that might leak target information
    leak_cols = ['price_per_sqft'] if 'price_per_sqft' in X.columns else []
    if leak_cols:
        X = X.drop(columns=leak_cols)
        logger.info(f"Dropped potential leakage columns: {leak_cols}")
    
    logger.info(f"✓ Features shape: {X.shape}, Target shape: {y.shape}")
    
    return X, y


def preprocessing_pipeline(
    df: pd.DataFrame,
    encode_binary: bool = True,
    encode_categorical: bool = True,
    categorical_encoding: str = 'onehot',
    scale_features: bool = False,
    scaling_type: str = 'standard',
    create_interactions: bool = False,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, any]]:
    """
    Complete preprocessing pipeline.
    
    Applies all preprocessing steps and returns ready-to-model data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame to preprocess.
    encode_binary : bool, default=True
        Whether to encode binary columns.
    encode_categorical : bool, default=True
        Whether to encode categorical columns.
    categorical_encoding : str, default='onehot'
        Type of categorical encoding ('onehot' or 'label').
    scale_features : bool, default=False
        Whether to scale numerical features.
    scaling_type : str, default='standard'
        Type of scaling ('standard' or 'minmax').
    create_interactions : bool, default=False
        Whether to create feature interactions.
    verbose : bool, default=True
        Whether to print progress information.
    
    Returns
    -------
    tuple
        (X, y, preprocessing_info) where X is features, y is target,
        and preprocessing_info contains encoders and scalers.
    
    Examples
    --------
    >>> X, y, info = preprocessing_pipeline(df_clean)
    >>> print(f"Features shape: {X.shape}")
    >>> print(f"Target shape: {y.shape}")
    """
    preprocessing_info = {
        'steps_performed': [],
        'encoders': {},
        'scalers': {},
        'original_shape': df.shape,
        'feature_names': []
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"Initial shape: {df.shape}")
    
    df_processed = df.copy()
    
    # Step 1: Encode binary columns
    if encode_binary:
        df_processed = encode_binary_columns(df_processed)
        preprocessing_info['steps_performed'].append('encode_binary')
        if verbose:
            print("✓ Binary columns encoded")
    
    # Step 2: Encode categorical columns
    if encode_categorical:
        df_processed, cat_encoders = encode_categorical_columns(
            df_processed,
            encoding_type=categorical_encoding
        )
        preprocessing_info['encoders'].update(cat_encoders)
        preprocessing_info['steps_performed'].append('encode_categorical')
        if verbose:
            print(f"✓ Categorical columns encoded ({categorical_encoding})")
    
    # Step 3: Create feature interactions (before scaling)
    if create_interactions:
        df_processed = create_feature_interactions(df_processed)
        preprocessing_info['steps_performed'].append('create_interactions')
        if verbose:
            print("✓ Feature interactions created")
    
    # Step 4: Separate features and target
    X, y = prepare_features_and_target(df_processed)
    preprocessing_info['feature_names'] = X.columns.tolist()
    
    # Step 5: Scale features
    if scale_features:
        X, scalers = scale_numerical_features(
            X,
            scaling_type=scaling_type
        )
        preprocessing_info['scalers'] = scalers
        preprocessing_info['steps_performed'].append('scale_features')
        if verbose:
            print(f"✓ Features scaled ({scaling_type})")
    
    # Final info
    preprocessing_info['final_shape'] = X.shape
    preprocessing_info['n_features'] = X.shape[1]
    
    if verbose:
        print("\n" + "-" * 40)
        print("PREPROCESSING SUMMARY:")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature types:")
        print(f"  - Numerical: {len(X.select_dtypes(include=[np.number]).columns)}")
        print(f"  - Binary: {len([col for col in X.columns if X[col].nunique() == 2])}")
        print("=" * 60)
    
    return X, y, preprocessing_info


if __name__ == "__main__":
    """
    Test the preprocessing functions if run directly.
    """
    # Create test data
    test_data = pd.DataFrame({
        'price': [100000, 200000, 150000, 300000],
        'area': [1000, 1500, 1200, 2000],
        'bedrooms': [2, 3, 2, 4],
        'bathrooms': [1, 2, 1, 2],
        'mainroad': ['yes', 'no', 'yes', 'yes'],
        'guestroom': ['no', 'yes', 'no', 'yes'],
        'furnishing_status': ['furnished', 'semi-furnished', 'unfurnished', 'furnished']
    })
    
    print("Original Data:")
    print(test_data)
    print(f"\nData types:")
    print(test_data.dtypes)
    
    # Test preprocessing pipeline
    X, y, info = preprocessing_pipeline(test_data)
    
    print("\nPreprocessed Features:")
    print(X)
    print(f"\nFeature columns ({len(X.columns)}):")
    print(X.columns.tolist())
    print(f"\nTarget statistics:")
    print(y.describe())
