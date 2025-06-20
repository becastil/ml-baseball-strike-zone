import pandas as pd
import pickle
import os

def convert_pickle_to_csv():
    """Convert pickle files to CSV to avoid numpy version issues"""
    
    files_to_convert = [
        ('aaron_judge.p', 'aaron_judge.csv'),
        ('jose_altuve.p', 'jose_altuve.csv'),
        ('david_ortiz.p', 'david_ortiz.csv')
    ]
    
    for pickle_file, csv_file in files_to_convert:
        if os.path.exists(pickle_file):
            try:
                # Try to load with pandas read_pickle (more robust)
                print(f"Converting {pickle_file} to {csv_file}...")
                df = pd.read_pickle(pickle_file)
                df.to_csv(csv_file, index=False)
                print(f"✓ Successfully converted to {csv_file}")
            except Exception as e:
                print(f"✗ Error converting {pickle_file}: {e}")
        else:
            print(f"✗ {pickle_file} not found")

def load_from_csv():
    """Load data from CSV files"""
    aaron_judge = pd.read_csv('aaron_judge.csv') if os.path.exists('aaron_judge.csv') else pd.DataFrame()
    jose_altuve = pd.read_csv('jose_altuve.csv') if os.path.exists('jose_altuve.csv') else pd.DataFrame()
    david_ortiz = pd.read_csv('david_ortiz.csv') if os.path.exists('david_ortiz.csv') else pd.DataFrame()
    
    return aaron_judge, jose_altuve, david_ortiz

if __name__ == "__main__":
    print("Fixing pickle file compatibility issues...")
    print("=" * 50)
    convert_pickle_to_csv()
    print("\nNow you can use load_from_csv() in your notebook!")