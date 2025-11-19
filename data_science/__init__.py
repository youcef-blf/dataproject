"""
Data Science Module
===================

This module contains all functions related to data splitting,
model training, and evaluation for the house price prediction project.
"""

from .data import (
    split_data,
    save_splits,
    load_splits,
    get_train_test_info,
    create_validation_splits,
    check_target_distribution
)

from .models import (
    BaselineModel,
    train_baseline_model,
    train_linear_regression,
    train_ensemble_model,
    tune_hyperparameters,
    get_learning_curves,
    perform_cross_validation,
    save_model,
    load_model,
    train_all_models
)

from .evaluation import (
    calculate_metrics,
    evaluate_model,
    evaluate_multiple_models,
    get_feature_importance,
    analyze_predictions,
    check_overfitting,
    create_evaluation_report
)

__all__ = [
    # Data splitting functions
    "split_data",
    "save_splits",
    "load_splits",
    "get_train_test_info",
    "create_validation_splits",
    "check_target_distribution",
    # Model functions
    "BaselineModel",
    "train_baseline_model",
    "train_linear_regression",
    "train_ensemble_model",
    "tune_hyperparameters",
    "get_learning_curves",
    "perform_cross_validation",
    "save_model",
    "load_model",
    "train_all_models",
    # Evaluation functions
    "calculate_metrics",
    "evaluate_model",
    "evaluate_multiple_models",
    "get_feature_importance",
    "analyze_predictions",
    "check_overfitting",
    "create_evaluation_report"
]
