"""
Dự báo ML đến năm 2035 - Tách riêng các tệp Top10 theo từng đợt rút thăm và châu lục
Đầu vào: bwf_official.csv
Đầu ra: Top10_{Global|Asia|Europe}_{MS|WS|MD|WD|XD}_2035.csv (15 tệp)
"""


from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Try to import XGBoost and LightGBM (optional)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent
INPUT = BASE.parent / 'MACHINE LEARNING' / 'bwf_official.csv'
OUTPUT_DIR = BASE

# Country to Continent mapping
CONTINENT_MAP = {
    'CHN': 'Asia', 'JPN': 'Asia', 'KOR': 'Asia', 'IND': 'Asia', 'MAS': 'Asia', 'INA': 'Asia', 'THA': 'Asia', 'HKG': 'Asia', 'TPE': 'Asia', 'PAK': 'Asia', 'BAN': 'Asia', 'SGP': 'Asia', 'PHL': 'Asia', 'LAO': 'Asia',
    'GBR': 'Europe', 'FRA': 'Europe', 'GER': 'Europe', 'NED': 'Europe', 'DEN': 'Europe', 'SWE': 'Europe', 'ESP': 'Europe', 'RUS': 'Europe', 'POL': 'Europe', 'UKR': 'Europe', 'SUI': 'Europe', 'ITA': 'Europe', 'BEL': 'Europe', 'NOR': 'Europe', 'GRE': 'Europe', 'POR': 'Europe', 'ROM': 'Europe', 'BUL': 'Europe', 'HUN': 'Europe', 'CRO': 'Europe', 'SLO': 'Europe', 'FIN': 'Europe',
    'USA': 'Global', 'CAN': 'Global', 'MEX': 'Global', 'BRA': 'Global', 'ARG': 'Global', 'ECU': 'Global', 'COL': 'Global',
    'AUS': 'Global', 'NZL': 'Global',
}

DRAWS = ["MS", "WS", "MD", "WD", "XD"]
CONTINENTS = ["Global", "Asia", "Europe"]
FORECAST_MONTHS = 120
TARGET_YEAR = 2035

print("=" * 70)
print("ML FORECASTING TO 2035 - SEPARATE TOP10 PER DRAW & CONTINENT")
print("=" * 70)

# 1. Load data
print(f"\n[1] Loading data from {INPUT}...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"    Loaded {len(df)} rows")

# 2. Data preparation
print("\n[2] Preparing data...")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date', 'draw'])
df['draw'] = df['draw'].astype(str).str.upper()

# Add continent
df['continent'] = df['country_code'].map(CONTINENT_MAP).fillna('Global')

# Convert numeric columns
for col in ['rank', 'points', 'tournaments_played']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.sort_values(['player_id', 'draw', 'date']).reset_index(drop=True)
print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"    Draws: {df['draw'].unique()}")
print(f"    Unique players: {df['player_id'].nunique()}")

# 3. Feature engineering per draw
print("\n[3] Engineering features per draw...")

def add_features(group_df):
    group_df = group_df.sort_values('date')
    group_df['points_lag_1'] = group_df['points'].shift(1)
    group_df['points_lag_3'] = group_df['points'].shift(3)
    group_df['points_lag_6'] = group_df['points'].shift(6)
    group_df['rank_lag_1'] = group_df['rank'].shift(1)
    group_df['avg_points_3m'] = group_df['points'].rolling(3, min_periods=1).mean()
    group_df['avg_points_6m'] = group_df['points'].rolling(6, min_periods=1).mean()
    group_df['std_points_6m'] = group_df['points'].rolling(6, min_periods=1).std()
    return group_df

df = df.groupby(['player_id', 'draw'], group_keys=False).apply(add_features)

# Fill missing lags
lag_cols = ['points_lag_1', 'points_lag_3', 'points_lag_6', 'rank_lag_1', 'avg_points_3m', 'avg_points_6m', 'std_points_6m']
for col in lag_cols:
    df[col] = df.groupby(['player_id', 'draw'])[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(0)

df['month'] = df['date'].dt.month
print(f"    Features created: {lag_cols}")

# 4. Train and compare multiple models per draw
print("\n[4] Training and comparing models per draw...")
print("    Models: GradientBoosting, RandomForest" + (", XGBoost" if HAS_XGB else "") + (", LightGBM" if HAS_LGB else ""))

best_models = {}
model_comparison_results = []
feature_cols = lag_cols + ['tournaments_played', 'month']

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # Avoid division by zero
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

for draw in DRAWS:
    draw_df = df[df['draw'] == draw].copy()
    if len(draw_df) < 100:
        print(f"\n    SKIP {draw}: insufficient data")
        continue
    
    print(f"\n    {'='*60}")
    print(f"    {draw} - Comparing Models")
    print(f"    {'='*60}")
    
    # Split data: 70% train, 30% test
    cutoff_train = draw_df['date'].quantile(0.7)
    train_df = draw_df[draw_df['date'] <= cutoff_train]
    test_df = draw_df[draw_df['date'] > cutoff_train]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['points']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['points']
    
    print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Define models to compare
    models_to_test = {
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        )
    }
    
    if HAS_XGB:
        models_to_test['XGBoost'] = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbosity=0
        )
    
    if HAS_LGB:
        models_to_test['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbosity=-1
        )
    
    # Train and evaluate each model
    draw_results = []
    for model_name, model in models_to_test.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)
        
        draw_results.append({
            'draw': draw,
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'model_object': model
        })
        
        model_comparison_results.append({
            'Draw': draw,
            'Model': model_name,
            'MAE': f'{mae:.2f}',
            'RMSE': f'{rmse:.2f}',
            'R²': f'{r2:.4f}',
            'MAPE': f'{mape:.2f}%' if not np.isnan(mape) else 'N/A'
        })
    
    # Select best model (lowest RMSE)
    best_result = min(draw_results, key=lambda x: x['rmse'])
    best_models[draw] = best_result['model_object']
    
    # Display results table
    print(f"\n    {'Model':<20} {'MAE':<12} {'RMSE':<12} {'R²':<10} {'MAPE':<10}")
    print(f"    {'-'*64}")
    for result in draw_results:
        marker = ' ★ BEST' if result['model'] == best_result['model'] else ''
        mape_str = f"{result['mape']:.2f}%" if not np.isnan(result['mape']) else 'N/A'
        print(f"    {result['model']:<20} {result['mae']:<12.2f} {result['rmse']:<12.2f} {result['r2']:<10.4f} {mape_str:<10}{marker}")

# Display summary comparison table
print(f"\n\n{'='*70}")
print("MODEL COMPARISON SUMMARY")
print(f"{'='*70}")
comparison_df = pd.DataFrame(model_comparison_results)
print(comparison_df.to_string(index=False))
print(f"\n✓ Best models selected for each draw (based on lowest RMSE)")
print(f"{'='*70}")

# 5. Recursive forecast per draw
print(f"\n[5] Forecasting {FORECAST_MONTHS} months ahead per draw...")

all_forecasts = []

for draw in DRAWS:
    if draw not in best_models:
        continue
    
    model = best_models[draw]
    draw_df = df[df['draw'] == draw].copy()
    
    # Latest per player
    latest = draw_df.sort_values('date').drop_duplicates('player_id', keep='last')
    
    print(f"    Forecasting {draw}: {len(latest)} players...")
    
    for idx, (_, row) in enumerate(latest.iterrows()):
        if (idx + 1) % 100 == 0:
            print(f"      {draw}: {idx + 1}/{len(latest)}")
        
        player_id = row['player_id']
        current_state = row.copy()
        
        for month_step in range(1, FORECAST_MONTHS + 1):
            # Prepare features
            features_dict = {col: current_state.get(col, 0) for col in feature_cols}
            X_pred = pd.DataFrame([features_dict])
            pred_points = model.predict(X_pred)[0]
            pred_points = max(0, pred_points)
            
            # Forecast date
            forecast_date = row['date'] + timedelta(days=30 * month_step)
            
            all_forecasts.append({
                'player_id': player_id,
                'player_name': row.get('player_name', ''),
                'player2_id': row.get('player2_id', ''),
                'player2_name': row.get('player2_name', ''),
                'country_code': row.get('country_code', ''),
                'continent': row['continent'],
                'draw': draw,
                'date': forecast_date,
                'predicted_points': pred_points,
                'horizon': month_step
            })
            
            # Update state
            current_state['points_lag_1'] = pred_points
            current_state['points_lag_3'] = current_state.get('points_lag_3', 0) * 0.67 + pred_points * 0.33
            current_state['points_lag_6'] = current_state.get('points_lag_6', 0) * 0.83 + pred_points * 0.17
            current_state['avg_points_3m'] = current_state['points_lag_3']
            current_state['avg_points_6m'] = current_state['points_lag_6']
            current_state['month'] = ((current_state['month'] - 1 + 1) % 12) + 1

forecast_df = pd.DataFrame(all_forecasts)
print(f"    Total forecasts: {len(forecast_df)}")

# 6. Generate Top10 per draw & continent
print(f"\n[6] Generating Top10 files...")

for draw in DRAWS:
    draw_forecast = forecast_df[forecast_df['draw'] == draw]
    if len(draw_forecast) == 0:
        print(f"    SKIP Top10_{draw}: no forecasts")
        continue
    
    # Get last month per player
    latest_forecast = draw_forecast.sort_values('date').drop_duplicates('player_id', keep='last')
    
    for continent in CONTINENTS:
        if continent == 'Global':
            cont_forecast = latest_forecast
        else:
            cont_forecast = latest_forecast[latest_forecast['continent'] == continent]
        
        if len(cont_forecast) == 0:
            print(f"    SKIP Top10_{continent}_{draw}: no data")
            continue
        
        # Top 10
        top10 = cont_forecast.nlargest(10, 'predicted_points')
        top10['predicted_rank'] = range(1, len(top10) + 1)
        
        # Reorder columns based on draw type
        if draw in ['MD', 'WD', 'XD']:  # Doubles - show both players
            top10_out = top10[['predicted_rank', 'player_id', 'player_name', 'player2_id', 'player2_name', 'country_code', 'continent', 'draw', 'predicted_points', 'date']]
        else:  # Singles - show only main player
            top10_out = top10[['predicted_rank', 'player_id', 'player_name', 'country_code', 'continent', 'draw', 'predicted_points', 'date']]
        
        # Save
        fname = OUTPUT_DIR / f"Top10_{continent}_{draw}_{TARGET_YEAR}.csv"
        top10_out.to_csv(fname, index=False)
        print(f"     {fname.name}")

print("\n" + "=" * 70)
print("COMPLETE - All Top10 files generated")
print("=" * 70)
