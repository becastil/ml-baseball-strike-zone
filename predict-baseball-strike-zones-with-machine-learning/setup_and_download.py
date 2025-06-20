import subprocess
import sys
import os

def install_packages():
    """Install required packages from requirements.txt"""
    print("Installing required packages from requirements.txt...")
    
    # First, ensure pip is up to date
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install from requirements.txt
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    
    print("\nAll packages installed successfully!")

def download_data():
    """Download player data"""
    try:
        from pybaseball import statcast_batter
        import pandas as pd
        import pickle
        
        print("\nDownloading Aaron Judge data (2017)...")
        aaron_judge = statcast_batter('2017-01-01', '2017-12-31', 592450)
        print(f"Downloaded {len(aaron_judge)} pitches for Aaron Judge")
        
        print("\nDownloading Jose Altuve data (2017)...")
        jose_altuve = statcast_batter('2017-01-01', '2017-12-31', 514888)
        print(f"Downloaded {len(jose_altuve)} pitches for Jose Altuve")
        
        print("\nDownloading David Ortiz data (2016)...")
        david_ortiz = statcast_batter('2016-01-01', '2016-12-31', 120074)
        print(f"Downloaded {len(david_ortiz)} pitches for David Ortiz")
        
        print("\nSaving data to pickle files...")
        aaron_judge.to_pickle('aaron_judge.p')
        jose_altuve.to_pickle('jose_altuve.p')
        david_ortiz.to_pickle('david_ortiz.p')
        
        print("\nData saved successfully!")
        print("Files created:")
        print("- aaron_judge.p")
        print("- jose_altuve.p")
        print("- david_ortiz.p")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please make sure you have an internet connection.")

if __name__ == "__main__":
    print("Setting up environment for baseball strike zone prediction...")
    print("=" * 50)
    
    # Install packages
    install_packages()
    
    # Download data
    print("\n" + "=" * 50)
    print("Starting data download...")
    download_data()