#!/usr/bin/env python3
"""
Main script for the House Price Prediction Project
Master 2 DS - Machine Learning

This script provides a command-line interface to run the complete
ML pipeline: data processing, model training, and evaluation.

Usage:
    python main.py --step=data-proc    # Preprocess the data
    python main.py --step=train        # Train models
    python main.py --step=test         # Evaluate on test set
    python main.py --step=all          # Run complete pipeline

Authors: Youcef, Mohamed-Amine
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
import config

# Import data processing modules
from data_processing.load_data import load_data, display_data_info, check_data_quality
from data_processing.clean_data import clean_data_pipeline
from data_processing.preprocessing import preprocessing_pipeline

# Import data science modules
from data_science.data import split_data, save_splits, load_splits, get_train_test_info
from data_science.models import train_all_models, save_model, load_model, tune_hyperparameters
from data_science.evaluation import (
    evaluate_model,
    evaluate_multiple_models,
    create_evaluation_report,
    get_feature_importance
)

# Import visualization modules
from figure.eda_plots import create_eda_report
from figure.model_plots import (
    plot_predictions_vs_actual,
    plot_feature_importance,
    plot_model_comparison,
    create_model_report
)


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def step_data_processing(args):
    """
    Execute data processing pipeline.
    
    Steps:
        1. Load raw data
        2. Display data info and quality check
        3. Clean data
        4. Preprocess data (encoding, feature engineering)
        5. Split into train/test sets
        6. Save processed data and splits
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test) - The data splits.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA PROCESSING")
    print("=" * 70)
    
    # 1. Load raw data
    print("\n📂 Loading raw data...")
    try:
        df = load_data(config.RAW_DATA_PATH, verbose=True)
    except FileNotFoundError:
        logger.error(f"Data file not found: {config.RAW_DATA_PATH}")
        logger.info("Please ensure the data file exists at the specified path.")
        print(f"\n💡 Expected location: {config.RAW_DATA_PATH}")
        print(f"   Create the directory and add your CSV file:")
        print(f"   mkdir -p {config.RAW_DATA_DIR}")
        sys.exit(1)
    
    # 2. Display data info
    print("\n📊 Data Overview:")
    display_data_info(df, detailed=args.verbose)
    
    # 3. Check data quality
    print("\n🔍 Checking data quality...")
    is_clean, issues = check_data_quality(df)
    if not is_clean:
        print("⚠️  Data quality issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ No major data quality issues detected")
    
    # 4. Clean data
    print("\n🧹 Cleaning data...")
    df_cleaned, cleaning_report = clean_data_pipeline(
        df,
        verbose=args.verbose
    )
    
    # Save cleaned data
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"✅ Cleaned data saved to: {config.PROCESSED_DATA_PATH}")
    
    # 5. Preprocess data
    print("\n⚙️  Preprocessing data...")
    X, y, preprocessing_info = preprocessing_pipeline(
        df_cleaned,
        encode_binary=True,
        encode_categorical=True,
        categorical_encoding='onehot',
        scale_features=False,  # Tree models don't need scaling
        create_interactions=args.feature_engineering,
        verbose=args.verbose
    )
    
    # 6. Split data
    print("\n✂️  Splitting data into train/test sets...")
    
    # Recombine for splitting
    df_processed = X.copy()
    df_processed[config.TARGET_COLUMN] = y
    
    X_train, y_train, X_test, y_test = split_data(
        df_processed,
        test_ratio=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # Display split info
    print("\n📈 Train/Test Split Summary:")
    split_info = get_train_test_info(X_train, y_train, X_test, y_test)
    print(split_info.to_string())
    
    # Save splits
    print("\n💾 Saving data splits...")
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = save_splits(X_train, y_train, X_test, y_test)
    print(f"✅ Splits saved to: {config.SPLITS_DIR}")
    
    # Generate EDA report if requested
    if args.generate_plots:
        print("\n📊 Generating EDA visualizations...")
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        eda_figures = create_eda_report(
            df_cleaned,
            target_col=config.TARGET_COLUMN,
            save_dir=config.FIGURES_DIR / "eda",
            show=False
        )
        print(f"✅ EDA figures saved to: {config.FIGURES_DIR / 'eda'}")
    
    print("\n" + "=" * 70)
    print("✅ DATA PROCESSING COMPLETE")
    print("=" * 70)
    
    return X_train, y_train, X_test, y_test


def step_train(args):
    """
    Execute model training pipeline.
    
    Steps:
        1. Load train/test splits
        2. Train multiple models
        3. Perform hyperparameter tuning (optional)
        4. Save trained models
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    
    Returns
    -------
    dict
        Dictionary of trained models.
    """
    print("\n" + "=" * 70)
    print("STEP 2: MODEL TRAINING")
    print("=" * 70)
    
    # 1. Load data splits
    print("\n📂 Loading data splits...")
    try:
        X_train, y_train, X_test, y_test = load_splits(config.SPLITS_DIR)
    except FileNotFoundError:
        logger.error("Data splits not found. Please run --step=data-proc first.")
        sys.exit(1)
    
    print(f"   Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # 2. Train all models
    print("\n🤖 Training models...")
    models = train_all_models(
        X_train, y_train,
        include_baseline=True,
        include_linear=True,
        include_rf=True,
        include_gb=True
    )
    
    # 3. Hyperparameter tuning (if requested)
    if args.tune:
        print("\n🔧 Performing hyperparameter tuning...")
        
        # Tune Random Forest
        print("\n   Tuning Random Forest...")
        best_rf, best_params_rf, cv_results_rf = tune_hyperparameters(
            X_train, y_train,
            model_type='random_forest',
            search_type='random' if args.fast else 'grid',
            n_iter=10 if args.fast else 20,
            cv=3 if args.fast else 5,
            verbose=1 if args.verbose else 0
        )
        models['random_forest_tuned'] = best_rf
        print(f"   ✅ Best RF params: {best_params_rf}")
        
        # Tune Gradient Boosting
        print("\n   Tuning Gradient Boosting...")
        best_gb, best_params_gb, cv_results_gb = tune_hyperparameters(
            X_train, y_train,
            model_type='gradient_boosting',
            search_type='random' if args.fast else 'grid',
            n_iter=10 if args.fast else 20,
            cv=3 if args.fast else 5,
            verbose=1 if args.verbose else 0
        )
        models['gradient_boosting_tuned'] = best_gb
        print(f"   ✅ Best GB params: {best_params_gb}")
    
    # 4. Save models
    print("\n💾 Saving trained models...")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        model_path = save_model(model, model_name=name)
        print(f"   ✅ {name} saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print(f"✅ MODEL TRAINING COMPLETE - {len(models)} models trained")
    print("=" * 70)
    
    return models


def step_test(args):
    """
    Execute model evaluation pipeline.
    
    Steps:
        1. Load test data and trained models
        2. Evaluate all models on test set
        3. Generate comparison report
        4. Create visualizations
        5. Identify best model
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    
    Returns
    -------
    dict
        Evaluation results and best model info.
    """
    print("\n" + "=" * 70)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 70)
    
    # 1. Load test data
    print("\n📂 Loading test data...")
    try:
        X_train, y_train, X_test, y_test = load_splits(config.SPLITS_DIR)
    except FileNotFoundError:
        logger.error("Data splits not found. Please run --step=data-proc first.")
        sys.exit(1)
    
    print(f"   Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # 2. Load trained models
    print("\n📂 Loading trained models...")
    models = {}
    model_files = list(config.MODELS_DIR.glob("*.pkl"))
    
    if not model_files:
        logger.error("No trained models found. Please run --step=train first.")
        sys.exit(1)
    
    for model_path in model_files:
        model_name = model_path.stem
        models[model_name] = load_model(model_path)
        print(f"   ✅ Loaded: {model_name}")
    
    # 3. Evaluate all models
    print("\n📊 Evaluating models on test set...")
    
    # Store results for visualization
    models_results = {}
    
    for name, model in models.items():
        results = evaluate_model(
            model, X_test, y_test,
            model_name=name,
            verbose=args.verbose
        )
        models_results[name] = {
            'predictions': results['predictions'],
            'y_true': y_test,
            'residuals': results['residuals'],
            'metrics': results['metrics']
        }
    
    # 4. Create comparison report
    print("\n📋 Generating comparison report...")
    comparison_df = evaluate_multiple_models(models, X_test, y_test, verbose=True)
    
    # Save comparison to CSV
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = config.RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"   ✅ Comparison saved to: {comparison_path}")
    
    # 5. Generate visualizations
    if args.generate_plots:
        print("\n📊 Generating evaluation visualizations...")
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Model comparison plot
        fig_comparison = plot_model_comparison(
            comparison_df,
            save_path=config.FIGURES_DIR / "model_comparison.png",
            show=False
        )
        
        # Best model visualizations
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = models[best_model_name]
        best_results = models_results[best_model_name]
        
        # Predictions vs Actual
        fig_pred = plot_predictions_vs_actual(
            best_results['y_true'],
            best_results['predictions'],
            model_name=best_model_name,
            save_path=config.FIGURES_DIR / f"predictions_{best_model_name}.png",
            show=False
        )
        
        # Feature importance (if available)
        if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'coef_'):
            importance_df = get_feature_importance(best_model, X_train.columns.tolist())
            if not importance_df.empty:
                fig_importance = plot_feature_importance(
                    importance_df,
                    save_path=config.FIGURES_DIR / f"feature_importance_{best_model_name}.png",
                    show=False
                )
        
        print(f"   ✅ Figures saved to: {config.FIGURES_DIR}")
    
    # 6. Create full evaluation report
    print("\n📋 Creating full evaluation report...")
    report = create_evaluation_report(
        models, X_train, y_train, X_test, y_test,
        save_path=config.RESULTS_DIR / "evaluation_report.json"
    )
    
    # 7. Summary
    best_model_name = comparison_df.iloc[0]['Model']
    best_r2 = comparison_df.iloc[0]['R²']
    best_mae = comparison_df.iloc[0]['MAE']
    
    print("\n" + "=" * 70)
    print("✅ MODEL EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   R² Score: {best_r2:.4f}")
    print(f"   MAE: ${best_mae:,.0f}")
    print("=" * 70)
    
    return {
        'comparison': comparison_df,
        'best_model': best_model_name,
        'best_r2': best_r2,
        'report': report
    }


def run_all_steps(args):
    """
    Run the complete ML pipeline.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    """
    print("\n" + "=" * 70)
    print("🚀 RUNNING COMPLETE ML PIPELINE")
    print("=" * 70)
    
    # Step 1: Data Processing
    X_train, y_train, X_test, y_test = step_data_processing(args)
    
    # Step 2: Model Training
    models = step_train(args)
    
    # Step 3: Evaluation
    results = step_test(args)
    
    print("\n" + "=" * 70)
    print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Results saved in:")
    print(f"   - Processed data: {config.PROCESSED_DATA_DIR}")
    print(f"   - Data splits: {config.SPLITS_DIR}")
    print(f"   - Trained models: {config.MODELS_DIR}")
    print(f"   - Figures: {config.FIGURES_DIR}")
    print(f"   - Reports: {config.RESULTS_DIR}")
    print("=" * 70)


def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="House Price Prediction - ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --step=data-proc          # Process and prepare data
    python main.py --step=train              # Train all models
    python main.py --step=train --tune       # Train with hyperparameter tuning
    python main.py --step=test               # Evaluate models on test set
    python main.py --step=all                # Run complete pipeline
    python main.py --step=all --tune --plots # Full pipeline with tuning and plots
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["data-proc", "train", "test", "all"],
        help="Pipeline step to execute"
    )
    
    # Optional arguments
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning during training"
    )
    
    parser.add_argument(
        "--plots", "--generate-plots",
        dest="generate_plots",
        action="store_true",
        help="Generate and save visualization plots"
    )
    
    parser.add_argument(
        "--feature-engineering",
        action="store_true",
        help="Create additional feature interactions"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster settings (less CV folds, fewer iterations)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="House Price Prediction v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    config.create_directories()
    
    # Execute requested step
    try:
        if args.step == "data-proc":
            step_data_processing(args)
        elif args.step == "train":
            step_train(args)
        elif args.step == "test":
            step_test(args)
        elif args.step == "all":
            run_all_steps(args)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
