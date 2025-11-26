"""
Ensemble Models Module
======================

Ensemble LightGBM + LSTM for improved forecasting accuracy.

Components:
1. LightGBM for tabular features
2. LSTM (PyTorch) for sequential patterns  
3. Weighted ensemble combination

Author: ML Improvements Project
Date: 2025-11-21
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
import warnings

# ML libraries
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# PyTorch for LSTM
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("  PyTorch not available. LSTM model will be skipped.")
    # Create dummy classes to avoid NameError
    class nn:
        class Module:
            pass

warnings.filterwarnings('ignore')

# Paths
BASE = Path(__file__).parent.parent
TEST_DIR = BASE / 'TEST'
MODELS_DIR = TEST_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True, parents=True)


# Only define LSTMModel if PyTorch is available
if PYTORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """LSTM model for time series forecasting"""
        
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            # x shape: (batch, sequence_length, input_size)
            lstm_out, _ = self.lstm(x)
            
            # Take last timestep output
            last_output = lstm_out[:, -1, :]
            
            # Fully connected layer
            output = self.fc(last_output)
            
            return output.squeeze()
else:
    # Dummy LSTMModel class when PyTorch not available
    class LSTMModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available. Cannot use LSTM model.")


class EnsembleForecaster:
    """
    Ensemble model combining LightGBM and LSTM
    """
    
    def __init__(self, sequence_length=12):
        self.sequence_length = sequence_length
        self.lgbm_model = None
        self.lstm_model = None
        self.lgbm_weight = 0.6
        self.lstm_weight = 0.4
        self.feature_cols = None
        self.scaler_mean = None
        self.scaler_std = None
        
    def prepare_sequences(self, df, feature_cols, target_col='points'):
        """
        Prepare sequences for LSTM
        
        Returns:
            X_seq: (n_samples, sequence_length, n_features)
            y: (n_samples,)
        """
        df = df.sort_values(['player_id', 'draw', 'date']).reset_index(drop=True)
        
        sequences = []
        targets = []
        
        for (player_id, draw), group in df.groupby(['player_id', 'draw']):
            group = group.sort_values('date')
            
            if len(group) < self.sequence_length + 1:
                continue
            
            # Create sequences
            for i in range(len(group) - self.sequence_length):
                seq = group.iloc[i:i+self.sequence_length][feature_cols].values
                target = group.iloc[i+self.sequence_length][target_col]
                
                sequences.append(seq)
                targets.append(target)
        
        X_seq = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)
        
        # Check if we have valid sequences
        if len(X_seq) == 0 or X_seq.ndim < 3:
            print(f"     Warning: No valid sequences created (need at least {self.sequence_length} consecutive months)")
            return np.array([]), np.array([])
        
        # Normalize
        self.scaler_mean = X_seq.mean(axis=(0, 1))
        self.scaler_std = X_seq.std(axis=(0, 1)) + 1e-8
        
        X_seq = (X_seq - self.scaler_mean) / self.scaler_std
        
        return X_seq, y
    
    def train_lgbm(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM model"""
        print("\n Training LightGBM model...")
        
        self.lgbm_model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        if X_val is not None and y_val is not None:
            self.lgbm_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                # early_stopping_rounds=50,  # Commented out due to deprecation
                # verbose=False  # Already set via verbose=-1
            )
        else:
            self.lgbm_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.lgbm_model.predict(X_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        
        print(f"    LightGBM trained")
        print(f"   Train MAE: {mae_train:.2f}")
        
        if X_val is not None:
            y_pred_val = self.lgbm_model.predict(X_val)
            mae_val = mean_absolute_error(y_val, y_pred_val)
            print(f"   Val MAE:   {mae_val:.2f}")
            return mae_val
        
        return mae_train
    
    def train_lstm(self, X_seq_train, y_train, X_seq_val=None, y_val=None, epochs=50):
        """Train LSTM model"""
        
        if not PYTORCH_AVAILABLE:
            print("  PyTorch not available, skipping LSTM training")
            return None
        
        print("\n Training LSTM model...")
        
        input_size = X_seq_train.shape[2]
        
        self.lstm_model = LSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_seq_train)
        y_train_t = torch.FloatTensor(y_train)
        
        # Training loop
        self.lstm_model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.lstm_model(X_train_t)
            loss = criterion(outputs, y_train_t)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        # Evaluate
        self.lstm_model.eval()
        with torch.no_grad():
            y_pred_train = self.lstm_model(X_train_t).numpy()
            mae_train = mean_absolute_error(y_train, y_pred_train)
        
        print(f"    LSTM trained")
        print(f"   Train MAE: {mae_train:.2f}")
        
        if X_seq_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_seq_val)
            with torch.no_grad():
                y_pred_val = self.lstm_model(X_val_t).numpy()
                mae_val = mean_absolute_error(y_val, y_pred_val)
            print(f"   Val MAE:   {mae_val:.2f}")
            return mae_val
        
        return mae_train
    
    def predict_ensemble(self, X_tabular, X_seq=None):
        """
        Predict using ensemble
        
        Args:
            X_tabular: Tabular features for LightGBM
            X_seq: Sequential features for LSTM (optional)
        
        Returns:
            Ensemble predictions
        """
        # LightGBM prediction
        lgbm_pred = self.lgbm_model.predict(X_tabular)
        
        # If LSTM available and sequences provided
        if PYTORCH_AVAILABLE and self.lstm_model is not None and X_seq is not None:
            # Normalize sequences
            X_seq_norm = (X_seq - self.scaler_mean) / self.scaler_std
            X_seq_t = torch.FloatTensor(X_seq_norm)
            
            self.lstm_model.eval()
            with torch.no_grad():
                lstm_pred = self.lstm_model(X_seq_t).numpy()
            
            # Weighted ensemble
            ensemble_pred = (self.lgbm_weight * lgbm_pred + 
                           self.lstm_weight * lstm_pred)
        else:
            # Only LightGBM if LSTM not available
            ensemble_pred = lgbm_pred
        
        return ensemble_pred
    
    def save_models(self, prefix='ensemble'):
        """Save trained models"""
        # Save LightGBM
        lgbm_path = MODELS_DIR / f'{prefix}_lgbm.pkl'
        with open(lgbm_path, 'wb') as f:
            pickle.dump(self.lgbm_model, f)
        print(f"    Saved LightGBM to {lgbm_path}")
        
        # Save LSTM if available
        if PYTORCH_AVAILABLE and self.lstm_model is not None:
            lstm_path = MODELS_DIR / f'{prefix}_lstm.pt'
            torch.save(self.lstm_model.state_dict(), lstm_path)
            print(f"    Saved LSTM to {lstm_path}")
        
        # Save ensemble config
        config = {
            'lgbm_weight': self.lgbm_weight,
            'lstm_weight': self.lstm_weight,
            'sequence_length': self.sequence_length,
            'feature_cols': self.feature_cols,
            'scaler_mean': self.scaler_mean.tolist() if self.scaler_mean is not None else None,
            'scaler_std': self.scaler_std.tolist() if self.scaler_std is not None else None
        }
        
        config_path = MODELS_DIR / f'{prefix}_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"    Saved config to {config_path}")


def demo_ensemble():
    """Demo ensemble model training"""
    
    print("\n" + "=" * 70)
    print("ENSEMBLE MODEL TRAINING DEMO")
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
    
    # Filter to MS draw only for demo
    df = df[df['draw'] == 'MS'].copy()
    print(f"   Loaded {len(df)} rows (Men's Singles only)")
    
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
    
    # Train/val split (temporal)
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] <= split_date]
    val_df = df[df['date'] > split_date]
    
    print(f"\n Data split:")
    print(f"   Train: {len(train_df)} samples (up to {split_date.date()})")
    print(f"   Val:   {len(val_df)} samples")
    
    # Initialize ensemble
    ensemble = EnsembleForecaster(sequence_length=12)
    ensemble.feature_cols = feature_cols
    
    # Prepare tabular data
    X_train = train_df[feature_cols].values
    y_train = train_df['points'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['points'].values
    
    # Train LightGBM
    lgbm_mae = ensemble.train_lgbm(X_train, y_train, X_val, y_val)
    
    # Prepare sequential data for LSTM
    if PYTORCH_AVAILABLE:
        X_seq_train, y_seq_train = ensemble.prepare_sequences(train_df, feature_cols)
        X_seq_val, y_seq_val = ensemble.prepare_sequences(val_df, feature_cols)
        
        # Check if we have valid sequences
        if len(X_seq_train) == 0:
            print(f"\n  Warning: Not enough training data for sequences. Skipping LSTM training.")
            print(f"   Need at least {ensemble.sequence_length} consecutive months per player.")
        elif len(X_seq_val) == 0:
            print(f"\n  Warning: Not enough validation data for sequences. Training LSTM without validation.")
            
            print(f"\n Sequence data:")
            print(f"   Train sequences: {X_seq_train.shape}")
            
            # Train LSTM without validation
            lstm_mae = ensemble.train_lstm(X_seq_train, y_seq_train, epochs=30)
        else:
            print(f"\n Sequence data:")
            print(f"   Train sequences: {X_seq_train.shape}")
            print(f"   Val sequences:   {X_seq_val.shape}")
            
            # Train LSTM
            lstm_mae = ensemble.train_lstm(X_seq_train, y_seq_train, X_seq_val, y_seq_val, epochs=30)
            
            # Ensemble prediction on validation
            print(f"\n Ensemble prediction on validation set...")
            ensemble_pred = ensemble.predict_ensemble(X_val, X_seq_val)
            ensemble_mae = mean_absolute_error(y_seq_val, ensemble_pred[:len(y_seq_val)])
            
            print(f"\n Final Results:")
            print(f"   LightGBM MAE:  {lgbm_mae:.2f}")
            print(f"   LSTM MAE:      {lstm_mae:.2f}")
            print(f"   Ensemble MAE:  {ensemble_mae:.2f}")
            
            if ensemble_mae < min(lgbm_mae, lstm_mae):
                print(f"    Ensemble outperforms individual models!")
    
    # Save models
    print(f"\n Saving models...")
    ensemble.save_models(prefix='ensemble_MS')
    
    print("\n Ensemble training completed!")


if __name__ == "__main__":
    demo_ensemble()
