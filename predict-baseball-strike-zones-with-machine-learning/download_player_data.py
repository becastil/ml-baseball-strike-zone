try:
    from pybaseball import statcast_batter
    import pandas as pd
    import pickle
    print("✓ All required packages imported successfully")
except ImportError as e:
    print(f"Error importing packages: {e}")
    print("\nPlease run: python simple_install.py")
    exit(1)

def download_player(start_date, end_date, player_id, player_name):
    """Download data for a single player with error handling"""
    try:
        print(f"\nDownloading {player_name} data ({start_date[:4]})...")
        data = statcast_batter(start_date, end_date, player_id)
        print(f"✓ Downloaded {len(data)} pitches for {player_name}")
        return data
    except Exception as e:
        print(f"✗ Error downloading {player_name}: {e}")
        return pd.DataFrame()

# Download player data
players_data = {
    'aaron_judge': download_player('2017-01-01', '2017-12-31', 592450, 'Aaron Judge'),
    'jose_altuve': download_player('2017-01-01', '2017-12-31', 514888, 'Jose Altuve'),
    'david_ortiz': download_player('2016-01-01', '2016-12-31', 120074, 'David Ortiz')
}

# Save data to pickle files
print("\nSaving data to pickle files...")
saved_files = []

for player_key, data in players_data.items():
    if not data.empty:
        try:
            filename = f"{player_key}.p"
            data.to_pickle(filename)
            saved_files.append(filename)
            print(f"✓ Saved {filename}")
        except Exception as e:
            print(f"✗ Error saving {player_key}: {e}")
    else:
        print(f"✗ No data to save for {player_key}")

print("\n" + "=" * 50)
print("Summary:")
print(f"✓ Successfully saved {len(saved_files)} files:")
for file in saved_files:
    print(f"  - {file}")

if len(saved_files) < 3:
    print("\n⚠ Some files were not created. Check errors above.")