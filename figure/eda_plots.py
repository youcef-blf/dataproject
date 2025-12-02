"""
EDA visualization module for the House Price Prediction Project
Master 2 DS - Machine Learning

This module contains functions for exploratory data analysis visualizations.
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


def plot_missing_values(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot missing values heatmap and bar chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.
    figsize : tuple, default=(12, 6)
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
    >>> fig = plot_missing_values(df)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
    
    # Plot 1: Bar chart of missing values
    missing_pct[missing_pct > 0].plot(kind='bar', ax=axes[0], color='coral')
    axes[0].set_title('Missing Values by Column (%)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Columns')
    axes[0].set_ylabel('Percentage Missing')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for i, v in enumerate(missing_pct[missing_pct > 0]):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
    
    # Plot 2: Heatmap of missing values
    sns.heatmap(
        df.isnull(),
        cbar=True,
        cmap='RdYlBu_r',
        ax=axes[1],
        cbar_kws={'label': 'Missing'}
    )
    axes[1].set_title('Missing Values Pattern', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Samples')
    
    plt.suptitle('Missing Values Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_distributions(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot distributions of numerical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numerical columns.
    numerical_cols : list of str, optional
        List of columns to plot. If None, plots all numerical columns.
    n_cols : int, default=3
        Number of columns in subplot grid.
    figsize : tuple, default=(15, 10)
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
    >>> fig = plot_distributions(df, numerical_cols=['price', 'area'])
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_plots = len(numerical_cols)
    n_rows = (n_plots - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i, col in enumerate(numerical_cols):
        # Plot histogram with KDE
        sns.histplot(
            data=df,
            x=col,
            kde=True,
            ax=axes[i],
            color=config.COLOR_PALETTE[i % len(config.COLOR_PALETTE)]
        )
        
        # Add statistics
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.1f}')
        axes[i].axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.1f}')
        
        axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].legend(fontsize=8)
        
        # Add skewness info
        skew = df[col].skew()
        axes[i].text(0.02, 0.98, f'Skew: {skew:.2f}', 
                    transform=axes[i].transAxes, 
                    fontsize=9, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Numerical Variables Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot distributions of categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with categorical columns.
    categorical_cols : list of str, optional
        List of columns to plot.
    n_cols : int, default=3
        Number of columns in subplot grid.
    figsize : tuple, default=(15, 10)
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
    >>> fig = plot_categorical_distributions(df, categorical_cols=['mainroad', 'guestroom'])
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        # Also include binary columns that might be encoded as 0/1
        binary_cols = [col for col in df.columns 
                      if df[col].nunique() == 2 and col not in categorical_cols]
        categorical_cols.extend(binary_cols)
    
    n_plots = len(categorical_cols)
    if n_plots == 0:
        logger.warning("No categorical columns to plot")
        return None
    
    n_rows = (n_plots - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i, col in enumerate(categorical_cols):
        # Count values
        value_counts = df[col].value_counts()
        
        # Create bar plot
        sns.barplot(
            x=value_counts.index,
            y=value_counts.values,
            ax=axes[i],
            palette=config.COLOR_PALETTE
        )
        
        axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Count')
        
        # Rotate labels if necessary
        if len(value_counts) > 3:
            axes[i].tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for j, v in enumerate(value_counts.values):
            axes[i].text(j, v + len(df)*0.01, str(v), ha='center', fontsize=9)
        
        # Add percentage info
        total = value_counts.sum()
        pct_text = '\n'.join([f'{val}: {cnt/total*100:.1f}%' 
                             for val, cnt in value_counts.items()])
        axes[i].text(0.98, 0.98, pct_text,
                    transform=axes[i].transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Categorical Variables Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = 'pearson',
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = True,
    mask_upper: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numerical columns.
    method : {'pearson', 'spearman', 'kendall'}, default='pearson'
        Correlation method.
    figsize : tuple, default=(12, 10)
        Figure size.
    annot : bool, default=True
        Whether to annotate cells with correlation values.
    mask_upper : bool, default=True
        Whether to mask upper triangle.
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
    >>> fig = plot_correlation_matrix(df, method='pearson')
    """
    # Select numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numerical_df.corr(method=method)
    
    # Create mask for upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"},
        ax=ax
    )
    
    ax.set_title(f'{method.capitalize()} Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_target_relationships(
    df: pd.DataFrame,
    target_col: str = 'price',
    feature_cols: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot relationships between features and target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    target_col : str, default='price'
        Name of target column.
    feature_cols : list of str, optional
        Features to plot. If None, plots all numerical features.
    n_cols : int, default=3
        Number of columns in subplot grid.
    figsize : tuple, default=(15, 12)
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
    >>> fig = plot_target_relationships(df, target_col='price')
    """
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found")
        return None
    
    if feature_cols is None:
        # Select numerical features (excluding target)
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != target_col]
    
    n_plots = len(feature_cols)
    n_rows = (n_plots - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i, col in enumerate(feature_cols):
        # Determine plot type based on feature type
        if df[col].nunique() <= 10:
            # Box plot for discrete features
            sns.boxplot(data=df, x=col, y=target_col, ax=axes[i])
            axes[i].set_xlabel(col)
        else:
            # Scatter plot for continuous features
            sns.scatterplot(data=df, x=col, y=target_col, ax=axes[i], alpha=0.6)
            
            # Add regression line
            sns.regplot(data=df, x=col, y=target_col, ax=axes[i], 
                       scatter=False, color='red', label='Trend')
            
            # Calculate and display correlation
            corr = df[[col, target_col]].corr().iloc[0, 1]
            axes[i].text(0.02, 0.98, f'Corr: {corr:.3f}',
                        transform=axes[i].transAxes,
                        fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        axes[i].set_title(f'{col} vs {target_col}', fontsize=11, fontweight='bold')
        axes[i].set_ylabel(target_col if i % n_cols == 0 else '')
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Feature Relationships with {target_col}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_pairplot(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    hue: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Any:
    """
    Create pairplot for selected columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to plot.
    columns : list of str, optional
        Columns to include. If None, includes top 5 numerical columns by variance.
    hue : str, optional
        Column name for color encoding.
    figsize : tuple, default=(12, 12)
        Figure size (approximate for pairplot).
    save_path : str or Path, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the plot.
    
    Returns
    -------
    seaborn.PairGrid
        The created pairplot.
    
    Examples
    --------
    >>> pairplot = plot_pairplot(df, columns=['price', 'area', 'bedrooms'])
    """
    if columns is None:
        # Select top 5 numerical columns by variance
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 5:
            variances = df[numerical_cols].var()
            columns = variances.nlargest(5).index.tolist()
        else:
            columns = numerical_cols.tolist()
    
    # Create pairplot
    pairplot = sns.pairplot(
        df[columns + [hue]] if hue and hue not in columns else df[columns],
        hue=hue,
        diag_kind='kde',
        plot_kws={'alpha': 0.6},
        height=figsize[0] / len(columns),
        aspect=1
    )
    
    pairplot.fig.suptitle('Pairwise Relationships', fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        pairplot.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return pairplot


def plot_outliers(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    method: str = 'iqr',
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot outliers using box plots and identify outlier counts.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numerical columns.
    numerical_cols : list of str, optional
        Columns to analyze for outliers.
    method : {'iqr', 'zscore'}, default='iqr'
        Method for outlier detection.
    n_cols : int, default=3
        Number of columns in subplot grid.
    figsize : tuple, default=(15, 10)
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
    >>> fig = plot_outliers(df, method='iqr')
    """
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_plots = len(numerical_cols)
    n_rows = (n_plots - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    outlier_counts = {}
    
    for i, col in enumerate(numerical_cols):
        # Create box plot
        sns.boxplot(data=df, y=col, ax=axes[i], color=config.COLOR_PALETTE[i % len(config.COLOR_PALETTE)])
        
        # Detect outliers
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        else:  # zscore method
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[col][z_scores > 3]
        
        outlier_counts[col] = len(outliers)
        
        axes[i].set_title(f'{col}\n({len(outliers)} outliers)', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('')
        
        # Add outlier percentage
        pct_outliers = len(outliers) / len(df) * 100
        axes[i].text(0.98, 0.98, f'{pct_outliers:.1f}%',
                    transform=axes[i].transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Outlier Analysis ({method.upper()} method)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Log outlier summary
    logger.info("Outlier Summary:")
    for col, count in outlier_counts.items():
        logger.info(f"  {col}: {count} outliers ({count/len(df)*100:.1f}%)")
    
    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"✓ Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_eda_report(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Dict[str, plt.Figure]:
    """
    Create comprehensive EDA report with all visualizations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.
    target_col : str, optional
        Target column name.
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
    >>> figures = create_eda_report(df, target_col='price', save_dir='figures/')
    """
    figures = {}
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Creating EDA Report")
    logger.info("=" * 60)
    
    # 1. Missing values
    logger.info("1. Creating missing values plot...")
    figures['missing_values'] = plot_missing_values(
        df,
        save_path=save_dir / 'missing_values.png' if save_dir else None,
        show=show
    )
    
    # 2. Numerical distributions
    logger.info("2. Creating numerical distributions plot...")
    figures['numerical_distributions'] = plot_distributions(
        df,
        save_path=save_dir / 'numerical_distributions.png' if save_dir else None,
        show=show
    )
    
    # 3. Categorical distributions
    logger.info("3. Creating categorical distributions plot...")
    figures['categorical_distributions'] = plot_categorical_distributions(
        df,
        save_path=save_dir / 'categorical_distributions.png' if save_dir else None,
        show=show
    )
    
    # 4. Correlation matrix
    logger.info("4. Creating correlation matrix...")
    figures['correlation_matrix'] = plot_correlation_matrix(
        df,
        save_path=save_dir / 'correlation_matrix.png' if save_dir else None,
        show=show
    )
    
    # 5. Target relationships
    if target_col and target_col in df.columns:
        logger.info("5. Creating target relationships plot...")
        figures['target_relationships'] = plot_target_relationships(
            df,
            target_col=target_col,
            save_path=save_dir / 'target_relationships.png' if save_dir else None,
            show=show
        )
    
    # 6. Outliers
    logger.info("6. Creating outliers plot...")
    figures['outliers'] = plot_outliers(
        df,
        save_path=save_dir / 'outliers.png' if save_dir else None,
        show=show
    )
    
    logger.info(f"✓ EDA Report complete. {len(figures)} figures created.")
    
    return figures


if __name__ == "__main__":
    """
    Test the EDA plotting functions if run directly.
    """
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 200
    
    test_data = pd.DataFrame({
        'price': np.random.lognormal(12, 0.5, n_samples),
        'area': np.random.normal(1500, 500, n_samples),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'bathrooms': np.random.choice([1, 2, 3], n_samples),
        'mainroad': np.random.choice(['yes', 'no'], n_samples),
        'guestroom': np.random.choice(['yes', 'no'], n_samples),
        'furnishing_status': np.random.choice(['furnished', 'semi-furnished', 'unfurnished'], n_samples)
    })
    
    # Add some missing values
    test_data.loc[np.random.choice(test_data.index, 10), 'area'] = np.nan
    test_data.loc[np.random.choice(test_data.index, 5), 'bathrooms'] = np.nan
    
    print("Test Data Created:")
    print(test_data.info())
    
    # Test EDA report
    print("\n" + "=" * 60)
    print("Testing EDA Report Generation")
    print("=" * 60)
    
    figures = create_eda_report(test_data, target_col='price', show=False)
    
    print(f"\n✓ EDA tests completed successfully!")
    print(f"Figures created: {list(figures.keys())}")
