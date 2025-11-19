"""
Configuration file for the House Price Prediction Project
Master 2 DS - Machine Learning

This file contains all the parameters and constants used throughout the project.
"""

import os
from pathlib import Path


# ==============================================================================
# PROJECT PATHS
# ==============================================================================

# Project root directory
ROOT_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Model directory
MODELS_DIR = ROOT_DIR / "models"

# Figures directory
FIGURES_DIR = ROOT_DIR / "figures"

# Results directory
RESULTS_DIR = ROOT_DIR / "results"


# ==============================================================================
# DATA FILES
# ==============================================================================

# Raw data file
RAW_DATA_FILE = "house_prices.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILE

# Processed data file
PROCESSED_DATA_FILE = "house_prices_cleaned.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / PROCESSED_DATA_FILE

# Split data files
TRAIN_DATA_FILE = "train.csv"
TEST_DATA_FILE = "test.csv"
X_TRAIN_FILE = "X_train.csv"
Y_TRAIN_FILE = "y_train.csv"
X_TEST_FILE = "X_test.csv"
Y_TEST_FILE = "y_test.csv"


# ==============================================================================
# DATA PROCESSING PARAMETERS
# ==============================================================================

# Target variable
TARGET_COLUMN = "price"

# Columns to drop (House Age has 99% missing values)
COLUMNS_TO_DROP = ["house_age"]

# Binary columns (Yes/No)
BINARY_COLUMNS = [
    "mainroad",
    "guestroom", 
    "basement",
    "hot_water_heating",
    "airconditioning",
    "prefarea"
]

# Categorical columns
CATEGORICAL_COLUMNS = ["furnishing_status"]

# Numerical columns (will be determined dynamically but listed for reference)
NUMERICAL_COLUMNS = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]

# Column name mapping for standardization
COLUMN_MAPPING = {
    "price": "price",
    "AreA": "area",
    "bedrooms": "bedrooms",
    "BATHROOMS": "bathrooms",
    "stories": "stories",
    "mainroad": "mainroad",
    "guestroom": "guestroom",
    "basement": "basement",
    "hotwaterheating": "hot_water_heating",
    "air conditioning": "airconditioning",
    "parking": "parking",
    "prefarea": "prefarea",
    "furnishing STATUS": "furnishing_status",
    "houSeaGe": "house_age"
}

# Standardization for furnishing status values
FURNISHING_STATUS_MAPPING = {
    "furnished": "furnished",
    "FURNISHED": "furnished",
    "Furnished": "furnished",
    "semi-furnished": "semi-furnished",
    "unfurnished": "unfurnished"
}


# ==============================================================================
# MODELING PARAMETERS
# ==============================================================================

# Train/Test split parameters
TEST_SIZE = 0.2  # Ratio of test set
RANDOM_STATE = 42  # Random seed for reproducibility
SHUFFLE = True  # Whether to shuffle data before splitting

# Cross-validation parameters
CV_FOLDS = 5  # Number of folds for cross-validation

# Model parameters
BASELINE_FEATURE = "bedrooms"  # Feature for baseline model

# Random Forest default parameters
RF_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE
}

# Gradient Boosting default parameters
GB_DEFAULT_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": RANDOM_STATE
}

# Training sample sizes for learning curves
TRAIN_SIZES = [10, 50, 100, 250, 500]

# Hyperparameter tuning
PARAM_GRID_RF = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

PARAM_GRID_GB = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.3],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0]
}

# Number of iterations for RandomizedSearchCV
N_ITER_RANDOM = 20


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

# Metrics to use
METRICS = ["mae", "rmse", "r2", "mape"]

# Primary metric for model selection
PRIMARY_METRIC = "r2"


# ==============================================================================
# VISUALIZATION PARAMETERS
# ==============================================================================

# Figure size defaults
FIG_SIZE_DEFAULT = (10, 6)
FIG_SIZE_LARGE = (15, 10)
FIG_SIZE_SMALL = (8, 5)

# Color palette
COLOR_PALETTE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
]

# Seaborn style
SEABORN_STYLE = "whitegrid"

# DPI for saving figures
FIGURE_DPI = 100


# ==============================================================================
# LOGGING PARAMETERS
# ==============================================================================

# Logging level
LOG_LEVEL = "INFO"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ==============================================================================
# VALIDATION PARAMETERS
# ==============================================================================

# Minimum acceptable R2 score
MIN_R2_SCORE = 0.6

# Maximum acceptable training/test score difference (to detect overfitting)
MAX_SCORE_DIFFERENCE = 0.15


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_directories():
    """Create all necessary project directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        RESULTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory created/verified: {directory}")


def get_config_summary():
    """Return a summary of the configuration as a string."""
    summary = []
    summary.append("=" * 60)
    summary.append("PROJECT CONFIGURATION SUMMARY")
    summary.append("=" * 60)
    summary.append(f"Root Directory: {ROOT_DIR}")
    summary.append(f"Test Size: {TEST_SIZE * 100:.0f}%")
    summary.append(f"Random State: {RANDOM_STATE}")
    summary.append(f"CV Folds: {CV_FOLDS}")
    summary.append(f"Primary Metric: {PRIMARY_METRIC}")
    summary.append(f"Target Column: {TARGET_COLUMN}")
    summary.append("=" * 60)
    
    return "\n".join(summary)


if __name__ == "__main__":
    # If run directly, create directories and show configuration
    create_directories()
    print("\n" + get_config_summary())
