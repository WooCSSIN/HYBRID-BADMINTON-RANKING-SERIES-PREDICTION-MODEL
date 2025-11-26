"""
Validation Framework Module
============================

Validate models với time series best practices:
- Walk-forward validation
- Backtesting framework
- Comprehensive metrics calculation

Author: ML Improvements Project
Date: 2025-11-21
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE = Path(__file__).parent.parent
TEST_DIR = BASE / 'TEST'
OUTPUT_DIR = TEST_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class TimeSeriesValidator:
    """
    Time series validation framework for BWF forecasting models
    """
    
    def __init__(self, df, target_col='points'):
        self.df = df.copy()
        self.target_col = target_col
        self.results = []
        
    def walk_forward_validation(self, model_class, feature_cols, n_splits=5, draw='MS'):
        """
        Walk-forward validation (expanding window)
        
        Args:
            model_class: Model class to instantiate (e.g., LGBMRegressor)
            feature_cols: List of feature column names
            n_splits: Number of validation splits
            draw: Draw type to validate on
        
        Returns:
            DataFrame with validation results per split
        """
        print("\n" + "=" * 70)
        print(f"WALK-FORWARD VALIDATION - {draw}")
        print("=" * 70)
        
        # Filter data for this draw
        df_draw = self.df[self.df['draw'] == draw].copy()
        df_draw = df_draw.sort_values('date').reset_index(drop=True)
        
        if len(df_draw) < 100:
            print(f"  Warning: Insufficient data for {draw} ({len(df_draw)} rows)")
            return pd.DataFrame()
        
        # Calculate split points
        total_rows = len(df_draw)
        split_size = total_rows // (n_splits + 1)
        
        split_results = []
        
        for i in range(n_splits):
            print(f"\n Split {i+1}/{n_splits}")
            
            # Expanding window: train on all data up to this point
            train_end_idx = split_size * (i + 1)
            test_end_idx = split_size * (i + 2)
            
            train_df = df_draw.iloc[:train_end_idx]
            test_df = df_draw.iloc[train_end_idx:test_end_idx]
            
            if len(test_df) == 0:
                continue
            
            # Prepare features
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df[self.target_col]
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df[self.target_col]
            
            # Train model
            model = model_class()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Ranking correlation
            test_df_copy = test_df.copy()
            test_df_copy['predicted'] = y_pred
            corr, _ = spearmanr(test_df_copy[self.target_col], test_df_copy['predicted'])
            
            split_results.append({
                'split': i + 1,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'train_end_date': train_df['date'].max().strftime('%Y-%m-%d'),
                'test_end_date': test_df['date'].max().strftime('%Y-%m-%d'),
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'spearman_corr': corr
            })
            
            print(f"   Train: {len(train_df)} samples (up to {train_df['date'].max().date()})")
            print(f"   Test:  {len(test_df)} samples ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
            print(f"   MAE:   {mae:.2f}")
            print(f"   RMSE:  {rmse:.2f}")
            print(f"   MAPE:  {mape:.2f}%")
            print(f"   Corr:  {corr:.3f}")
        
        results_df = pd.DataFrame(split_results)
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Average MAE:  {results_df['mae'].mean():.2f} ± {results_df['mae'].std():.2f}")
        print(f"Average RMSE: {results_df['rmse'].mean():.2f} ± {results_df['rmse'].std():.2f}")
        print(f"Average MAPE: {results_df['mape'].mean():.2f}% ± {results_df['mape'].std():.2f}%")
        print(f"Average Corr: {results_df['spearman_corr'].mean():.3f}")
        
        return results_df
    
    def backtest(self, predictions_df, actual_df, player_id_col='player_id', date_col='date'):
        """
        Backtest predictions vs actual results
        
        Args:
            predictions_df: DataFrame with predictions
            actual_df: DataFrame with actual values
            player_id_col: Player ID column name
            date_col: Date column name
        
        Returns:
            DataFrame with comparison results
        """
        print("\n" + "=" * 70)
        print("BACKTESTING - PREDICTIONS VS ACTUAL")
        print("=" * 70)
        
        # Merge predictions with actuals
        comparison = predictions_df.merge(
            actual_df[[player_id_col, date_col, self.target_col]],
            on=[player_id_col, date_col],
            suffixes=('_pred', '_actual'),
            how='inner'
        )
        
        if len(comparison) == 0:
            print("  No matching records for backtesting")
            return pd.DataFrame()
        
        # Calculate errors
        comparison['error'] = comparison[f'{self.target_col}_pred'] - comparison[f'{self.target_col}_actual']
        comparison['abs_error'] = comparison['error'].abs()
        comparison['pct_error'] = (comparison['error'] / comparison[f'{self.target_col}_actual']) * 100
        comparison['abs_pct_error'] = comparison['pct_error'].abs()
        
        # Metrics
        mae = comparison['abs_error'].mean()
        rmse = np.sqrt((comparison['error'] ** 2).mean())
        mape = comparison['abs_pct_error'].mean()
        
        print(f"\n Backtest Results ({len(comparison)} predictions)")
        print(f"   MAE:  {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        
        # Error distribution
        print(f"\n Error Distribution:")
        print(f"   Min error:    {comparison['error'].min():.2f}")
        print(f"   25th percentile: {comparison['error'].quantile(0.25):.2f}")
        print(f"   Median error: {comparison['error'].median():.2f}")
        print(f"   75th percentile: {comparison['error'].quantile(0.75):.2f}")
        print(f"   Max error:    {comparison['error'].max():.2f}")
        
        return comparison
    
    def plot_validation_results(self, results_df, title="Walk-Forward Validation Results"):
        """Plot validation metrics over splits"""
        
        if len(results_df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # MAE over splits
        axes[0, 0].plot(results_df['split'], results_df['mae'], marker='o', color='blue')
        axes[0, 0].set_title('MAE per Split')
        axes[0, 0].set_xlabel('Split')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE over splits
        axes[0, 1].plot(results_df['split'], results_df['rmse'], marker='o', color='red')
        axes[0, 1].set_title('RMSE per Split')
        axes[0, 1].set_xlabel('Split')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAPE over splits
        axes[1, 0].plot(results_df['split'], results_df['mape'], marker='o', color='green')
        axes[1, 0].set_title('MAPE per Split')
        axes[1, 0].set_xlabel('Split')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation over splits
        axes[1, 1].plot(results_df['split'], results_df['spearman_corr'], marker='o', color='purple')
        axes[1, 1].set_title('Spearman Correlation per Split')
        axes[1, 1].set_xlabel('Split')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = OUTPUT_DIR / 'validation_results.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n Saved validation plot to: {plot_path}")
        
        plt.close()


def demo_validation():
    """
    Demo validation framework
    Note: This is a simplified demo. In production, use actual trained models.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    print("\n" + "=" * 70)
    print("VALIDATION FRAMEWORK DEMO")
    print("=" * 70)
    
    # Check if enhanced data exists
    enhanced_file = TEST_DIR / 'bwf_official_enhanced.csv'
    
    if not enhanced_file.exists():
        print(f"\n  Enhanced dataset not found: {enhanced_file}")
        print("   Please run 1_feature_engineering.py first")
        print("\n Running feature engineering now...")
        
        # Import and run feature engineering
        import sys
        sys.path.insert(0, str(TEST_DIR))
        from importlib import import_module
        fe_module = import_module('1_feature_engineering')
        fe_module.main()
        
        if not enhanced_file.exists():
            print("\n Failed to create enhanced dataset")
            return
    
    # Load enhanced data
    print(f"\n Loading enhanced dataset...")
    df = pd.read_csv(enhanced_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    print(f"   Loaded {len(df)} rows")
    
    # Select feature columns
    feature_cols = [
        'rank', 'tournaments_played',
        'points_lag_1', 'points_lag_3', 'points_lag_6',
        'rank_lag_1', 'avg_points_3m', 'avg_points_6m',
        'std_points_6m', 'win_rate_estimated', 'tournament_weight',
        'career_years', 'momentum_score'
    ]
    
    # Filter to only include rows with all features
    df = df.dropna(subset=feature_cols + ['points'])
    
    # Validate on Men's Singles
    validator = TimeSeriesValidator(df, target_col='points')
    
    results = validator.walk_forward_validation(
        model_class=GradientBoostingRegressor,
        feature_cols=feature_cols,
        n_splits=5,
        draw='MS'
    )
    
    if len(results) > 0:
        # Save results
        results_file = OUTPUT_DIR / 'validation_results_MS.csv'
        results.to_csv(results_file, index=False)
        print(f"\n Saved validation results to: {results_file}")
        
        # Plot results
        validator.plot_validation_results(results, title="Walk-Forward Validation - Men's Singles")
    
    print("\n Validation framework demo completed!")


if __name__ == "__main__":
    demo_validation()
