import pickle
import os
import pandas as pd

def load_player_data():
    """Load player data from pickle files if they exist"""
    players = {}
    
    files = {
        'aaron_judge': 'aaron_judge.p',
        'jose_altuve': 'jose_altuve.p',
        'david_ortiz': 'david_ortiz.p'
    }
    
    for player_name, filename in files.items():
        if os.path.exists(filename):
            try:
                players[player_name] = pickle.load(open(filename, "rb"))
                print(f"Loaded {player_name} data successfully")
            except Exception as e:
                print(f"Error loading {player_name}: {e}")
                players[player_name] = pd.DataFrame()
        else:
            print(f"Warning: {filename} not found. Run setup_and_download.py first.")
            players[player_name] = pd.DataFrame()
    
    return players['aaron_judge'], players['jose_altuve'], players['david_ortiz']

# Try to load the data
aaron_judge, jose_altuve, david_ortiz = load_player_data()