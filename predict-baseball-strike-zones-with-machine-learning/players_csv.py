import pandas as pd
import os

# Try to load from CSV files first (more compatible)
if os.path.exists('aaron_judge.csv'):
    aaron_judge = pd.read_csv('aaron_judge.csv')
    jose_altuve = pd.read_csv('jose_altuve.csv')
    david_ortiz = pd.read_csv('david_ortiz.csv')
    print("Loaded player data from CSV files")
else:
    # If CSV files don't exist, create empty dataframes
    print("Warning: Player data files not found. Please run:")
    print("1. python download_player_data.py")
    print("2. python fix_pickle_files.py")
    aaron_judge = pd.DataFrame()
    jose_altuve = pd.DataFrame()
    david_ortiz = pd.DataFrame()