"""
Feature Engineering Module
===========================

Thêm advanced features vào BWF dataset để cải thiện model accuracy.

Features được thêm:
1. Win/Loss Ratio (estimated từ points trend)
2. Tournament Importance Weighting
3. Age & Career Features (estimated)
4. Enhanced Lag Features
5. Momentum & Trend Indicators

Author: ML Improvements Project
Date: 2025-11-21
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE = Path(__file__).parent.parent
INPUT_DIR = BASE / 'MACHINE LEARNING'
OUTPUT_DIR = BASE / 'TEST'
INPUT_FILE = INPUT_DIR / 'bwf_official.csv'
OUTPUT_FILE = OUTPUT_DIR / 'bwf_official_enhanced.csv'


class BWFFeatureEngineer:
    """
    Feature engineering pipeline for BWF badminton data
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_columns = list(df.columns)
        self.new_features = []
        
    def add_all_features(self):
        """Execute all feature engineering steps"""
        print("=" * 70)
        print("BWF FEATURE ENGINEERING - ADDING ADVANCED FEATURES")
        print("=" * 70)
        
        self.df = self._prepare_data()
        self.df = self._add_win_loss_features()
        self.df = self._add_tournament_features()
        self.df = self._add_career_features()
        self.df = self._add_enhanced_lag_features()
        self.df = self._add_momentum_features()
        
        print(f"\n Feature engineering complete!")
        print(f"   Original features: {len(self.original_columns)}")
        print(f"   New features added: {len(self.new_features)}")
        print(f"   Total features: {len(self.df.columns)}")
        print(f"\n New features: {', '.join(self.new_features)}")
        
        return self.df
    
    def _prepare_data(self):
        """Chuẩn bị dữ liệu cơ bản"""
        print("\n[1/6] Preparing data...")
        
        df = self.df.copy()
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Convert numeric columns
        for col in ['rank', 'points', 'tournaments_played']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by player and date
        df = df.sort_values(['player_id', 'draw', 'date']).reset_index(drop=True)
        
        # Filter out rows with missing critical data
        df = df.dropna(subset=['date', 'player_id', 'draw', 'points'])
        
        print(f"    Processed {len(df)} rows")
        return df
    
    def _add_win_loss_features(self):
        """
        Add win/loss ratio features (SIMULATED)
        
         TO USE REAL DATA:
        Replace this logic with actual match results if available:
        - Real columns: 'matches_won', 'matches_played', 'result'
        - Compute actual win_rate = matches_won / matches_played
        - Track win_streak from consecutive wins
        """
        print("\n[2/6] Adding win/loss features (simulated)...")
        
        df = self.df.copy()
        
        def estimate_wins(group):
            """Estimate wins from points trend"""
            group = group.sort_values('date')
            
            # Points increase → assume win, decrease → assume loss
            group['points_change_temp'] = group['points'].diff()
            
            # Estimate win probability based on points change
            # Positive change → higher win probability
            group['win_probability'] = (group['points_change_temp'] / 1000).clip(-1, 1)
            group['win_probability'] = (group['win_probability'] + 1) / 2  # Scale to [0, 1]
            group['win_probability'] = group['win_probability'].fillna(0.5)
            
            # Estimate win rate (rolling 10 weeks)
            group['win_rate_estimated'] = group['win_probability'].rolling(10, min_periods=1).mean()
            
            # Estimate loss rate
            group['loss_rate_estimated'] = 1 - group['win_rate_estimated']
            
            # Estimate win streak (consecutive positive point changes)
            group['is_positive_change'] = (group['points_change_temp'] > 0).astype(int)
            group['win_streak_estimated'] = (
                group['is_positive_change']
                * (group.groupby((group['is_positive_change'] != group['is_positive_change'].shift()).cumsum()).cumcount() + 1)
            )
            
            # Clean up temp columns
            group = group.drop(['points_change_temp', 'win_probability', 'is_positive_change'], axis=1)
            
            return group
        
        df = df.groupby(['player_id', 'draw'], group_keys=False).apply(estimate_wins)
        
        self.new_features.extend(['win_rate_estimated', 'loss_rate_estimated', 'win_streak_estimated'])
        
        print(f"    Added: win_rate_estimated, loss_rate_estimated, win_streak_estimated")
        return df
    
    def _add_tournament_features(self):
        """
        Add tournament importance weighting
        
         TO USE REAL DATA:
        Map actual tournament names to weights:
        - Parse 'event_name' column if available
        - Use official BWF tournament tier system
        - Weight: Olympic=5.0, World Champ=4.5, Super Series=3.0, etc.
        """
        print("\n[3/6] Adding tournament importance features...")
        
        df = self.df.copy()
        
        # SIMULATED: Estimate tournament importance from points distribution
        # Higher points concentration → more important tournament
        
        def estimate_tournament_weight(group):
            """Estimate tournament importance from points variance"""
            group = group.sort_values('date')
            
            # High variance in points → important tournaments
            group['points_std_6m'] = group['points'].rolling(6, min_periods=1).std()
            
            # Normalize to [1.0, 3.0] scale (conservative estimates)
            max_std = group['points_std_6m'].max()
            if max_std > 0:
                group['tournament_weight'] = 1.0 + 2.0 * (group['points_std_6m'] / max_std)
            else:
                group['tournament_weight'] = 1.5  # Default medium importance
            
            group['tournament_weight'] = group['tournament_weight'].fillna(1.5)
            
            # Weighted points
            group['weighted_points'] = group['points'] * group['tournament_weight']
            
            # Average weighted points (6 months rolling)
            group['avg_weighted_points_6m'] = group['weighted_points'].rolling(6, min_periods=1).mean()
            
            # Clean up temp
            group = group.drop(['points_std_6m'], axis=1)
            
            return group
        
        df = df.groupby(['player_id', 'draw'], group_keys=False).apply(estimate_tournament_weight)
        
        self.new_features.extend(['tournament_weight', 'weighted_points', 'avg_weighted_points_6m'])
        
        print(f"    Added: tournament_weight, weighted_points, avg_weighted_points_6m")
        return df
    
    def _add_career_features(self):
        """
        Add age & career features (SIMULATED)
        
         TO USE REAL DATA:
        Use actual birth dates and debut dates:
        - Real columns: 'birth_year', 'debut_year', 'birth_date'
        - Compute actual age = current_year - birth_year
        - Peak age typically 25-30 for badminton
        """
        print("\n[4/6] Adding career features (simulated)...")
        
        df = self.df.copy()
        
        def estimate_career_age(group):
            """Estimate career stage from first appearance"""
            group = group.sort_values('date')
            
            # First appearance date = estimated debut
            debut_date = group['date'].min()
            
            # Career duration in years and months
            group['career_years'] = ((group['date'] - debut_date).dt.days / 365.25).round(1)
            group['career_months'] = ((group['date'] - debut_date).dt.days / 30.44).round(0).astype(int)
            
            # Peak career indicator (2-5 years experience is often peak)
            group['in_peak_career'] = ((group['career_years'] >= 2) & (group['career_years'] <= 5)).astype(int)
            
            # Career stage
            def career_stage(years):
                if years < 1:
                    return 'rookie'
                elif years < 3:
                    return 'rising'
                elif years < 7:
                    return 'peak'
                elif years < 12:
                    return 'experienced'
                else:
                    return 'veteran'
            
            group['career_stage'] = group['career_years'].apply(career_stage)
            
            return group
        
        df = df.groupby(['player_id', 'draw'], group_keys=False).apply(estimate_career_age)
        
        self.new_features.extend(['career_years', 'career_months', 'in_peak_career', 'career_stage'])
        
        print(f"    Added: career_years, career_months, in_peak_career, career_stage")
        return df
    
    def _add_enhanced_lag_features(self):
        """
        Add enhanced lag features (similar to forecast_to_2035.py but more)
        """
        print("\n[5/6] Adding enhanced lag features...")
        
        df = self.df.copy()
        
        def add_lags(group):
            group = group.sort_values('date')
            
            # Points lags
            group['points_lag_1'] = group['points'].shift(1)
            group['points_lag_3'] = group['points'].shift(3)
            group['points_lag_6'] = group['points'].shift(6)
            group['points_lag_12'] = group['points'].shift(12)
            
            # Rank lags
            group['rank_lag_1'] = group['rank'].shift(1)
            group['rank_lag_3'] = group['rank'].shift(3)
            
            # Rolling averages
            group['avg_points_3m'] = group['points'].rolling(3, min_periods=1).mean()
            group['avg_points_6m'] = group['points'].rolling(6, min_periods=1).mean()
            group['avg_points_12m'] = group['points'].rolling(12, min_periods=1).mean()
            
            # Rolling std (volatility)
            group['std_points_6m'] = group['points'].rolling(6, min_periods=1).std().fillna(0)
            group['std_points_12m'] = group['points'].rolling(12, min_periods=1).std().fillna(0)
            
            # Changes
            group['points_change_1m'] = group['points'] - group['points_lag_1']
            group['points_change_6m'] = group['points'] - group['points_lag_6']
            group['rank_change_1m'] = group['rank'] - group['rank_lag_1']
            
            # Fill NaN with group medians
            for col in group.columns:
                if col.startswith(('points_lag_', 'rank_lag_', 'avg_', 'std_', 'points_change', 'rank_change')):
                    group[col] = group[col].fillna(group[col].median()).fillna(0)
            
            return group
        
        df = df.groupby(['player_id', 'draw'], group_keys=False).apply(add_lags)
        
        lag_features = [
            'points_lag_1', 'points_lag_3', 'points_lag_6', 'points_lag_12',
            'rank_lag_1', 'rank_lag_3',
            'avg_points_3m', 'avg_points_6m', 'avg_points_12m',
            'std_points_6m', 'std_points_12m',
            'points_change_1m', 'points_change_6m', 'rank_change_1m'
        ]
        
        self.new_features.extend(lag_features)
        
        print(f"    Added {len(lag_features)} lag features")
        return df
    
    def _add_momentum_features(self):
        """Add momentum and trend indicators"""
        print("\n[6/6] Adding momentum features...")
        
        df = self.df.copy()
        
        def add_momentum(group):
            group = group.sort_values('date')
            
            # Momentum score (weighted recent performance)
            # Recent months weighted higher
            if len(group) >= 6:
                recent_changes = group['points'].diff().tail(6)
                weights = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.25])[:len(recent_changes)]
                group['momentum_score'] = (recent_changes * weights).sum() / weights.sum()
            else:
                group['momentum_score'] = group['points'].diff().mean()
            
            group['momentum_score'] = group['momentum_score'].fillna(0)
            
            # Trend indicator: rising (1), stable (0), falling (-1)
            group['trend_3m'] = np.sign(group['points'].diff(3)).fillna(0)
            
            # Consistency score (inverse of volatility)
            group['consistency_score'] = 1 / (1 + group['std_points_6m'])
            
            return group
        
        df = df.groupby(['player_id', 'draw'], group_keys=False).apply(add_momentum)
        
        self.new_features.extend(['momentum_score', 'trend_3m', 'consistency_score'])
        
        print(f"    Added: momentum_score, trend_3m, consistency_score")
        return df


def main():
    """Main execution"""
    
    # Check input file exists
    if not INPUT_FILE.exists():
        print(f" Error: Input file not found: {INPUT_FILE}")
        print(f"   Please ensure bwf_official.csv exists in MACHINE LEARNING folder")
        return
    
    # Load data
    print(f"\n Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Feature engineering
    engineer = BWFFeatureEngineer(df)
    df_enhanced = engineer.add_all_features()
    
    # Save enhanced dataset
    OUTPUT_DIR.mkdir(exist_ok=True)
    df_enhanced.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n Saved enhanced dataset to: {OUTPUT_FILE}")
    print(f"   Shape: {df_enhanced.shape}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)
    
    # Sample of new features
    print("\n Sample values (first player, first 3 records):")
    sample_cols = ['player_name', 'date', 'points', 'win_rate_estimated', 
                   'tournament_weight', 'career_years', 'momentum_score']
    available_cols = [c for c in sample_cols if c in df_enhanced.columns]
    print(df_enhanced[available_cols].head(3).to_string(index=False))
    
    # Feature statistics
    print("\n New feature statistics:")
    new_cols = engineer.new_features[:5]  # Show first 5
    print(df_enhanced[new_cols].describe())
    
    print("\n Feature engineering completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
