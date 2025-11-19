"""
Figures Module
==============

This module contains all functions related to data visualization
and plot generation for the house price prediction project.
"""

from .eda_plots import (
    plot_missing_values,
    plot_distributions,
    plot_categorical_distributions,
    plot_correlation_matrix,
    plot_target_relationships,
    plot_pairplot,
    plot_outliers,
    create_eda_report
)

from .model_plots import (
    plot_predictions_vs_actual,
    plot_feature_importance,
    plot_learning_curves,
    plot_model_comparison,
    plot_error_distribution,
    plot_hyperparameter_search,
    create_model_report
)

__all__ = [
    # EDA plots
    "plot_missing_values",
    "plot_distributions",
    "plot_categorical_distributions",
    "plot_correlation_matrix",
    "plot_target_relationships",
    "plot_pairplot",
    "plot_outliers",
    "create_eda_report",
    # Model plots
    "plot_predictions_vs_actual",
    "plot_feature_importance",
    "plot_learning_curves",
    "plot_model_comparison",
    "plot_error_distribution",
    "plot_hyperparameter_search",
    "create_model_report"
]
