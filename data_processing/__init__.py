"""
Data Processing Module
======================

This module contains all functions related to data loading, cleaning,
and preprocessing for the house price prediction project.
"""

from .load_data import (
    load_data,
    get_data_info,
    display_data_info,
    check_data_quality,
    get_initial_statistics
)

from .clean_data import (
    standardize_column_names,
    clean_furnishing_status,
    clean_binary_columns,
    handle_missing_values,
    remove_duplicates,
    remove_invalid_target_rows,
    clean_data_pipeline
)

from .preprocessing import (
    encode_binary_columns,
    encode_categorical_columns,
    scale_numerical_features,
    create_feature_interactions,
    prepare_features_and_target,
    preprocessing_pipeline
)

__all__ = [
    # Load data functions
    "load_data",
    "get_data_info",
    "display_data_info",
    "check_data_quality",
    "get_initial_statistics",
    # Clean data functions
    "standardize_column_names",
    "clean_furnishing_status",
    "clean_binary_columns",
    "handle_missing_values",
    "remove_duplicates",
    "remove_invalid_target_rows",
    "clean_data_pipeline",
    # Preprocessing functions
    "encode_binary_columns",
    "encode_categorical_columns",
    "scale_numerical_features",
    "create_feature_interactions",
    "prepare_features_and_target",
    "preprocessing_pipeline"
]
