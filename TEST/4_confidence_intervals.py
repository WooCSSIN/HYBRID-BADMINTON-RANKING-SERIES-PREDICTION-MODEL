"""
Confidence Intervals Module
============================

Predict với confidence intervals using Quantile Regression.

Provides:
- Lower bound (10th percentile)
- Median prediction (50th percentile)
- Upper bound (90th percentile)
- Confidence width

Author: ML Improvements Project
Date: 2025-11-21
"""

from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE = Path(__file__).parent.parent
TEST_DIR = BASE / 'TEST'
MODELS_DIR = TEST_DIR / 'models'
OUTPUT_DIR = TEST_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class QuantileForecaster:
    """
    Forecaster with confidence intervals using quantile regression
    """
    
    def __init__(self):
        self.model_lower = None  # 10th percentile
        self.model_median = None  # 50th percentile  
        self.model_upper = None  # 90th percentile
        self.feature_cols = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train 3 quantile models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        print("\n" + "=" * 70)
        print("TRAINING QUANTILE MODELS FOR CONFIDENCE INTERVALS")
        print("=" * 70)
        
        # Common hyperparameters
        params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'num_leaves': 31,
            'random_state': 42,
            'verbose': -1
        }
        
        # Train lower bound (10th percentile)
        print("\n[1/3] Training lower bound model (10th percentile)...")
        self.model_lower = LGBMRegressor(objective='quantile', alpha=0.1, **params)
        self.model_lower.fit(X_train, y_train)
        print("    Lower bound model trained")
        
        # Train median (50th percentile)
        print("\n[2/3] Training median model (50th percentile)...")
        self.model_median = LGBMRegressor(objective='quantile', alpha=0.5, **params)
        self.model_median.fit(X_train, y_train)
        print("    Median model trained")
        
        # Train upper bound (90th percentile)
        print("\n[3/3] Training upper bound model (90th percentile)...")
        self.model_upper = LGBMRegressor(objective='quantile', alpha=0.9, **params)
        self.model_upper.fit(X_train, y_train)
        print("    Upper bound model trained")
        
        # Validation
        if X_val is not None and y_val is not None:
            self._validate(X_val, y_val)
    
    def _validate(self, X_val, y_val):
        """Validate quantile predictions"""
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        
        pred_lower = self.model_lower.predict(X_val)
        pred_median = self.model_median.predict(X_val)
        pred_upper = self.model_upper.predict(X_val)
        
        # Check coverage (what % of actual values fall within bounds)
        within_bounds = ((y_val >= pred_lower) & (y_val <= pred_upper)).mean()
        below_lower = (y_val < pred_lower).mean()
        above_upper = (y_val > pred_upper).mean()
        
        print(f"\n Coverage Analysis:")
        print(f"   Within [10%, 90%]: {within_bounds*100:.1f}% (target: ~80%)")
        print(f"   Below lower bound: {below_lower*100:.1f}% (target: ~10%)")
        print(f"   Above upper bound: {above_upper*100:.1f}% (target: ~10%)")
        
        # Average confidence width
        avg_width = (pred_upper - pred_lower).mean()
        avg_value = y_val.mean()
        relative_width = (avg_width / avg_value) * 100
        
        print(f"\n Confidence Width:")
        print(f"   Average width: {avg_width:.2f} points")
        print(f"   Relative width: {relative_width:.1f}% of average value")
        
        # Median accuracy
        from sklearn.metrics import mean_absolute_error
        mae_median = mean_absolute_error(y_val, pred_median)
        print(f"\n Median Prediction Accuracy:")
        print(f"   MAE: {mae_median:.2f}")
    
    def predict_with_ci(self, X):
        """
        Predict with confidence intervals
        
        Args:
            X: Features to predict on
        
        Returns:
            DataFrame with columns: lower_bound, predicted_points, upper_bound, confidence_width
        """
        pred_lower = self.model_lower.predict(X)
        pred_median = self.model_median.predict(X)
        pred_upper = self.model_upper.predict(X)
        
        results = pd.DataFrame({
            'lower_bound': pred_lower,
            'predicted_points': pred_median,
            'upper_bound': pred_upper,
            'confidence_width': pred_upper - pred_lower
        })
        
        return results
    
    def forecast_player(self, player_data, feature_cols, months_ahead=12):
        """
        Forecast for a single player with confidence intervals
        
        Args:
            player_data: DataFrame with player's historical data
            feature_cols: List of feature column names
            months_ahead: Number of months to forecast
        
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        forecasts = []
        
        # Get latest state
        latest = player_data.sort_values('date').iloc[-1].copy()
        current_state = latest[feature_cols].values.reshape(1, -1)
        
        for month in range(1, months_ahead + 1):
            # Predict with CI
            preds = self.predict_with_ci(current_state)
            
            forecast_date = latest['date'] + pd.DateOffset(months=month)
            
            forecasts.append({
                'date': forecast_date,
                'month_ahead': month,
                'lower_bound': preds['lower_bound'].values[0],
                'predicted_points': preds['predicted_points'].values[0],
                'upper_bound': preds['upper_bound'].values[0],
                'confidence_width': preds['confidence_width'].values[0]
            })
            
            # Update state for next iteration (simple recursive update)
            # In production, use more sophisticated state update
            current_state[0, feature_cols.index('points_lag_1')] = preds['predicted_points'].values[0]
        
        return pd.DataFrame(forecasts)


def demo_confidence_intervals():
    """Demo confidence intervals forecasting"""
    
    print("\n" + "=" * 70)
    print("CONFIDENCE INTERVALS FORECASTING DEMO")
    print("=" * 70)
    
    # Load enhanced data
    enhanced_file = TEST_DIR / 'bwf_official_enhanced.csv'
    
    if not enhanced_file.exists():
        print(f"\n Enhanced dataset not found: {enhanced_file}")
        print("   Please run 1_feature_engineering.py first")
        return
    
    print(f"\n Loading data...")
    df = pd.read_csv(enhanced_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to MS only for demo
    df = df[df['draw'] == 'MS'].copy()
    print(f"   Loaded {len(df)} rows (Men's Singles)")
    
    # Feature columns
    feature_cols = [
        'rank', 'tournaments_played',
        'points_lag_1', 'points_lag_3', 'points_lag_6',
        'rank_lag_1', 'avg_points_3m', 'avg_points_6m',
        'std_points_6m', 'win_rate_estimated', 'tournament_weight',
        'career_years', 'momentum_score'
    ]
    
    # Remove rows with missing features
    df = df.dropna(subset=feature_cols + ['points'])
    
    # Train/val split
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] <= split_date]
    val_df = df[df['date'] > split_date]
    
    print(f"\n Data split:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    
    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df['points'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['points'].values
    
    # Train quantile forecaster
    forecaster = QuantileForecaster()
    forecaster.feature_cols = feature_cols
    forecaster.train(X_train, y_train, X_val, y_val)
    
    # Predict on validation with CI
    print(f"\n Predicting on validation set with confidence intervals...")
    predictions = forecaster.predict_with_ci(X_val)
    
    # Add actual values
    predictions['actual_points'] = y_val
    predictions['player_id'] = val_df['player_id'].values
    predictions['player_name'] = val_df['player_name'].values
    predictions['date'] = val_df['date'].values
    
    # Save predictions
    output_file = OUTPUT_DIR / 'predictions_with_ci_MS.csv'
    predictions.to_csv(output_file, index=False)
    print(f"\n Saved predictions to: {output_file}")
    
    # Show sample
    print(f"\n Sample predictions (first 5):")
    sample_cols = ['player_name', 'date', 'actual_points', 'lower_bound', 
                   'predicted_points', 'upper_bound', 'confidence_width']
    print(predictions[sample_cols].head().to_string(index=False))
    
    # Summary statistics
    print(f"\n Confidence Interval Statistics:")
    print(f"   Average confidence width: {predictions['confidence_width'].mean():.2f} points")
    print(f"   Min confidence width: {predictions['confidence_width'].min():.2f}")
    print(f"   Max confidence width: {predictions['confidence_width'].max():.2f}")
    
    # Check if actuals within bounds
    within_bounds = (
        (predictions['actual_points'] >= predictions['lower_bound']) & 
        (predictions['actual_points'] <= predictions['upper_bound'])
    ).mean()
    print(f"   Actual within CI: {within_bounds*100:.1f}%")
    
    print("\n Confidence intervals demo completed!")


if __name__ == "__main__":
    demo_confidence_intervals()
