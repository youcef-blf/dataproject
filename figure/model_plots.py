"""
Model visualization module for the House Price Prediction Project
Master 2 DS - Machine Learning

This module contains functions for visualizing model performance and results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Union, Any, Dict
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

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(config.COLOR_PALETTE)


def plot_predictions_vs_actual(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: str = "Model",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot predicted vs actual values with distribution comparison.
    
    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    model_name : str, default="Model"
        Name of the model for title.
    figsize : tuple, default=(12, 5)
        Figure size.
    save_path : str or Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
    
    Returns
    -------
    plt.Figure
        The created figure.
    
    Examples
    --------
    >>> fig = plot_predictions_vs_actual(y_test, predictions, "Random Forest")
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Plot 1: Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    axes[0].plot(y_true, p(y_true), 'g-', alpha=0.7, label='Fitted Line')
    
    axes[0].set_xlabel('Actual Values', fontsize=10)
    axes[0].set_ylabel('Predicted Values', fontsize=10)
    axes[0].set_title('Predictions vs Actual', fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=9)
    
    # Add metrics text
    axes[0].text(0.02, 0.98, f'R² = {r2:.4f}\nMAE = ${mae:,.0f}',
                transform=axes[0].transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Plot 2: Residuals
    residuals = y_pred - y_true
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=1)
    
    # Add ±1 std lines
    std_residuals = residuals.std()
    axes[1].axhline(y=std_residuals, color='orange', linestyle=':', alpha=0.7, label=f'±1 STD')
    axes[1].axhline(y=-std_residuals, color='orange', linestyle=':', alpha=0.7)
    
    axes[1].set_xlabel('Predicted Values', fontsize=10)
    axes[1].set_ylabel('Residuals', fontsize=10)
    axes[1].set_title('Residual Plot', fontsize=11, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    
    # Plot 3: Distribution comparison
    axes[2].hist(y_true, bins=30, alpha=0.5, label='Actual', color='blue', density=True)
    axes[2].hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='red', density=True)
    
    axes[2].set_xlabel('Values', fontsize=10)
    axes[2].set_ylabel('Density', fontsize=10)
    axes[2].set_title('Distribution Comparison', fontsize=11, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=9)
    
    plt.suptitle(f'{model_name} - Performance Visualization', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot feature importance as bar chart.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    top_n : int, default=15
        Number of top features to display.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
    
    Returns
    -------
    plt.Figure
        The created figure.
    
    Examples
    --------
    >>> fig = plot_feature_importance(importance_df, top_n=10)
    """
    # Select top features
    plot_df = importance_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(plot_df)), plot_df['Importance'].values)
    
    # Color bars with gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_df)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Set labels
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['Feature'].values)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(f'Top {min(top_n, len(plot_df))} Feature Importances', fontsize=14, fontweight='bold')
    
    # Add importance values on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        importance = row['Importance']
        if 'Importance_Pct' in plot_df.columns:
            pct = row['Importance_Pct']
            label = f'{importance:.4f} ({pct:.1f}%)'
        else:
            label = f'{importance:.4f}'
        
        ax.text(importance, i, f' {label}', 
                ha='left', va='center', fontsize=9)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    # Invert y-axis to have most important at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_learning_curves(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    model_name: str = "Model",
    scoring_name: str = "Score",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot learning curves showing training and validation scores.
    
    Parameters
    ----------
    train_sizes : np.ndarray
        Training set sizes.
    train_scores : np.ndarray
        Training scores (shape: n_sizes x n_cv_folds).
    val_scores : np.ndarray
        Validation scores (shape: n_sizes x n_cv_folds).
    model_name : str, default="Model"
        Name of the model.
    scoring_name : str, default="Score"
        Name of the scoring metric.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
    
    Returns
    -------
    plt.Figure
        The created figure.
    
    Examples
    --------
    >>> fig = plot_learning_curves(sizes, train_scores, val_scores, "Random Forest", "R²")
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='green', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='green')
    
    # Add labels and title
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel(scoring_name, fontsize=11)
    ax.set_title(f'Learning Curves - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for final scores
    final_train = train_mean[-1]
    final_val = val_mean[-1]
    gap = final_train - final_val
    
    ax.text(0.02, 0.98, 
            f'Final Training: {final_train:.4f}\n'
            f'Final Validation: {final_val:.4f}\n'
            f'Gap: {gap:.4f}',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot comparison of multiple models across different metrics.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with model comparison results.
    metrics : list of str, optional
        Metrics to plot. If None, plots all numerical columns.
    figsize : tuple, default=(14, 8)
        Figure size.
    save_path : str or Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
    
    Returns
    -------
    plt.Figure
        The created figure.
    
    Examples
    --------
    >>> fig = plot_model_comparison(comparison_df, metrics=['R²', 'MAE', 'RMSE'])
    """
    if metrics is None:
        # Select all numerical columns except 'Model' column
        metrics = [col for col in comparison_df.columns 
                  if col != 'Model' and pd.api.types.is_numeric_dtype(comparison_df[col])]
    
    n_metrics = len(metrics)
    n_models = len(comparison_df)
    
    # Create subplots
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        # Create bar plot for each metric
        bars = axes[i].bar(range(n_models), comparison_df[metric].values,
                          color=config.COLOR_PALETTE[:n_models])
        
        # Customize plot
        axes[i].set_xticks(range(n_models))
        axes[i].set_xticklabels(comparison_df['Model'].values, rotation=45, ha='right')
        axes[i].set_ylabel(metric, fontsize=10)
        axes[i].set_title(f'{metric} Comparison', fontsize=11, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if pd.notna(height):
                if metric in ['R²', 'Within 10%', 'Within 20%']:
                    label = f'{height:.2%}' if height <= 1 else f'{height:.1f}%'
                elif metric in ['MAE', 'RMSE']:
                    label = f'${height:,.0f}'
                else:
                    label = f'{height:.2f}'
                
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom', fontsize=8)
        
        # Highlight best model
        if metric == 'R²' or 'Within' in metric:
            best_idx = comparison_df[metric].idxmax()
        else:
            best_idx = comparison_df[metric].idxmin()
        
        if pd.notna(best_idx):
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(2)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_error_distribution(
    residuals: Union[pd.Series, np.ndarray],
    model_name: str = "Model",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot error/residual distribution analysis.
    
    Parameters
    ----------
    residuals : array-like
        Residuals (predicted - actual).
    model_name : str, default="Model"
        Name of the model.
    figsize : tuple, default=(12, 5)
        Figure size.
    save_path : str or Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
    
    Returns
    -------
    plt.Figure
        The created figure.
    
    Examples
    --------
    >>> fig = plot_error_distribution(residuals, "Random Forest")
    """
    residuals = np.array(residuals)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Histogram with KDE
    sns.histplot(residuals, kde=True, ax=axes[0], color='skyblue')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
    axes[0].axvline(x=residuals.mean(), color='green', linestyle='--', 
                   alpha=0.7, label=f'Mean: ${residuals.mean():,.0f}')
    axes[0].set_xlabel('Residuals', fontsize=10)
    axes[0].set_ylabel('Count', fontsize=10)
    axes[0].set_title('Residual Distribution', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    
    # Plot 2: Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Theoretical Quantiles', fontsize=10)
    axes[1].set_ylabel('Sample Quantiles', fontsize=10)
    
    # Plot 3: Box plot with outliers
    box = axes[2].boxplot(residuals, vert=True, patch_artist=True)
    box['boxes'][0].set_facecolor('lightcoral')
    axes[2].set_ylabel('Residuals', fontsize=10)
    axes[2].set_title('Residual Box Plot', fontsize=11, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: ${residuals.mean():,.0f}\n'
    stats_text += f'Std: ${residuals.std():,.0f}\n'
    stats_text += f'Min: ${residuals.min():,.0f}\n'
    stats_text += f'Max: ${residuals.max():,.0f}'
    
    axes[2].text(0.02, 0.98, stats_text,
                transform=axes[2].transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.suptitle(f'{model_name} - Error Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_hyperparameter_search(
    cv_results_df: pd.DataFrame,
    param_name: str,
    scoring: str = 'mean_test_score',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot hyperparameter search results.
    
    Parameters
    ----------
    cv_results_df : pd.DataFrame
        Cross-validation results from GridSearchCV or RandomizedSearchCV.
    param_name : str
        Name of the parameter to plot.
    scoring : str, default='mean_test_score'
        Scoring column to plot.
    figsize : tuple, default=(10, 6)
        Figure size.
    save_path : str or Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
    
    Returns
    -------
    plt.Figure
        The created figure.
    
    Examples
    --------
    >>> fig = plot_hyperparameter_search(cv_results, 'param_n_estimators')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract parameter column
    param_col = f'param_{param_name}'
    
    if param_col not in cv_results_df.columns:
        logger.warning(f"Parameter column '{param_col}' not found")
        return None
    
    # Get unique parameter values
    param_values = cv_results_df[param_col].unique()
    
    # Calculate mean and std for each parameter value
    mean_scores = []
    std_scores = []
    
    for val in param_values:
        mask = cv_results_df[param_col] == val
        mean_scores.append(cv_results_df[mask][scoring].mean())
        if f'std_test_score' in cv_results_df.columns:
            std_scores.append(cv_results_df[mask]['std_test_score'].mean())
        else:
            std_scores.append(0)
    
    # Convert to numeric if possible
    try:
        param_values = pd.to_numeric(param_values)
        # Sort by parameter value
        sorted_idx = np.argsort(param_values)
        param_values = param_values[sorted_idx]
        mean_scores = np.array(mean_scores)[sorted_idx]
        std_scores = np.array(std_scores)[sorted_idx]
        
        # Line plot for numerical parameters
        ax.plot(param_values, mean_scores, 'o-', color='blue', label='Mean Score')
        if any(std_scores):
            ax.fill_between(param_values, 
                           np.array(mean_scores) - np.array(std_scores),
                           np.array(mean_scores) + np.array(std_scores),
                           alpha=0.2, color='blue')
    except:
        # Bar plot for categorical parameters
        x_pos = np.arange(len(param_values))
        bars = ax.bar(x_pos, mean_scores, yerr=std_scores if any(std_scores) else None,
                      capsize=5, color='skyblue')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_values, rotation=45, ha='right')
        
        # Highlight best
        best_idx = np.argmax(mean_scores)
        bars[best_idx].set_color('coral')
    
    ax.set_xlabel(param_name, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Hyperparameter Tuning: {param_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark best value
    best_idx = np.argmax(mean_scores)
    best_val = param_values[best_idx]
    best_score = mean_scores[best_idx]
    
    ax.annotate(f'Best: {best_val}\nScore: {best_score:.4f}',
               xy=(best_val if isinstance(best_val, (int, float)) else best_idx, best_score),
               xytext=(10, 10), textcoords='offset points',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_model_report(
    models_results: Dict[str, Dict],
    comparison_df: pd.DataFrame,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Dict[str, plt.Figure]:
    """
    Create comprehensive model evaluation report with all visualizations.
    
    Parameters
    ----------
    models_results : dict
        Dictionary with model names as keys and evaluation results as values.
    comparison_df : pd.DataFrame
        Model comparison DataFrame.
    save_dir : str or Path, optional
        Directory to save figures.
    show : bool, default=True
        Whether to display plots.
    
    Returns
    -------
    dict
        Dictionary with figure names as keys and figures as values.
    
    Examples
    --------
    >>> figures = create_model_report(models_results, comparison_df, save_dir='figures/')
    """
    figures = {}
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Creating Model Evaluation Report")
    logger.info("=" * 60)
    
    # 1. Model comparison
    logger.info("1. Creating model comparison plot...")
    figures['model_comparison'] = plot_model_comparison(
        comparison_df,
        save_path=save_dir / 'model_comparison.png' if save_dir else None,
        show=show
    )
    
    # 2. Individual model predictions
    for model_name, results in models_results.items():
        if 'predictions' in results and 'y_true' in results:
            logger.info(f"2. Creating predictions plot for {model_name}...")
            fig_name = f'predictions_{model_name.lower().replace(" ", "_")}'
            figures[fig_name] = plot_predictions_vs_actual(
                results['y_true'],
                results['predictions'],
                model_name,
                save_path=save_dir / f'{fig_name}.png' if save_dir else None,
                show=show
            )
            
            # Error distribution
            if 'residuals' in results:
                logger.info(f"3. Creating error distribution plot for {model_name}...")
                fig_name = f'errors_{model_name.lower().replace(" ", "_")}'
                figures[fig_name] = plot_error_distribution(
                    results['residuals'],
                    model_name,
                    save_path=save_dir / f'{fig_name}.png' if save_dir else None,
                    show=show
                )
    
    logger.info(f"✓ Model Report complete. {len(figures)} figures created.")
    
    return figures


if __name__ == "__main__":
    """
    Test the model plotting functions if run directly.
    """
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 100
    
    # True and predicted values
    y_true = np.random.uniform(100000, 500000, n_samples)
    y_pred = y_true + np.random.normal(0, 20000, n_samples)
    residuals = y_pred - y_true
    
    # Feature importance data
    feature_names = [f'Feature_{i}' for i in range(10)]
    importances = np.random.exponential(0.1, 10)
    importances = importances / importances.sum()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Importance_Pct': importances * 100
    }).sort_values('Importance', ascending=False)
    
    # Model comparison data
    comparison_df = pd.DataFrame({
        'Model': ['Baseline', 'Linear', 'Random Forest', 'Gradient Boosting'],
        'R²': [0.45, 0.72, 0.85, 0.83],
        'MAE': [45000, 32000, 25000, 26000],
        'RMSE': [52000, 38000, 29000, 30000],
        'Within 10%': [35, 55, 68, 65]
    })
    
    print("Test Data Created")
    print("=" * 60)
    
    # Test plotting functions
    print("\n1. Testing predictions vs actual plot...")
    fig1 = plot_predictions_vs_actual(y_true, y_pred, "Test Model", show=False)
    
    print("\n2. Testing feature importance plot...")
    fig2 = plot_feature_importance(importance_df, top_n=5, show=False)
    
    print("\n3. Testing model comparison plot...")
    fig3 = plot_model_comparison(comparison_df, show=False)
    
    print("\n4. Testing error distribution plot...")
    fig4 = plot_error_distribution(residuals, "Test Model", show=False)
    
    print("\n✓ All model plotting tests completed successfully!")
