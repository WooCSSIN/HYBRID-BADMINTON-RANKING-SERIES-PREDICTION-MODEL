from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
IN = BASE.parent / 'MACHINE LEARNING' / 'bwf_cleaned_full.csv'
OUT = BASE / 'bwf_for_sql_pairs.csv'  # overwrite existing simplified pairs file

print('Read original cleaned file:', IN)
df = pd.read_csv(IN, low_memory=False)

if 'id_main' in df.columns:
    df['player_id'] = pd.to_numeric(df['id_main'], errors='coerce').astype('Int64')
elif 'id_main_int' in df.columns:
    df['player_id'] = pd.to_numeric(df['id_main_int'], errors='coerce').astype('Int64')
else:
    df['player_id'] = df['uid'].str.split('_').str[0].astype('Int64')

if 'player_name' in df.columns:
    df['player_name'] = df['player_name']
else:
    df['player_name'] = ''

if 'id_aux' in df.columns:
    df['player2_id'] = pd.to_numeric(df['id_aux'], errors='coerce').astype('Int64')
elif df['uid'].str.count('_').ge(2).any():
    df['player2_id'] = df['uid'].str.split('_').str[1].astype('Int64')
else:
    df['player2_id'] = pd.NA

if 'partner_name' in df.columns:
    df['player2_name'] = df['partner_name']
else:
    df['player2_name'] = pd.NA

# Ensure date, draw, country
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
else:
    df['date'] = pd.NA

if 'draw' in df.columns:
    df['draw'] = df['draw']
else:
    df['draw'] = pd.NA

if 'country_code' in df.columns:
    df['country_code'] = df['country_code']
else:
    df['country_code'] = pd.NA

# Keep selected columns and remove team_id
cols = ['uid','date','draw','country_code','player_id','player_name','player2_id','player2_name','rank','points','tournaments_played']
cols = [c for c in cols if c in df.columns]
df_out = df[cols].copy()

# Cast numeric columns safely
for c in ['rank','points','tournaments_played']:
    if c in df_out.columns:
        df_out[c] = pd.to_numeric(df_out[c], errors='coerce').astype('Int64')

# Save (overwrite existing simplified file)
print('Write updated pairs CSV to:', OUT)
df_out.to_csv(OUT, index=False, encoding='utf-8-sig', na_rep='')
print('Done. Columns in output:', df_out.columns.tolist())

