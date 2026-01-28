import pandas as pd
from pathlib import Path

# Path back to the cleaned data
DATA_DIR = Path("../../final_station_data_clean")
OUTPUT_FILE = "diurnal_pm2.5_seasonwise.csv"

all_data = []

print("Reading station files...")

# Loop through the 26 station folders
for station_folder in DATA_DIR.iterdir():
    if not station_folder.is_dir(): continue
    
    for file in station_folder.glob("*.csv"):
        df = pd.read_csv(file)
        
        # Ensure timestamp is datetime and extract hour
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Note: Your friend added 'season' and 'PM 2.5' columns already
        # We select only what we need to save memory
        all_data.append(df[['hour', 'season', 'PM 2.5']])

# Combine all stations into one big DataFrame
full_df = pd.concat(all_data)

# Calculate the mean PM 2.5 for every hour of every season
# This creates the 'Typical Day' for Winter, Monsoon, etc.
diurnal_stats = full_df.groupby(['season', 'hour'])['PM 2.5'].mean().reset_index()

# Save the numeric results
diurnal_stats.to_csv(OUTPUT_FILE, index=False)
print(f"Success! Diurnal averages saved to {OUTPUT_FILE}")