# Baseball Strike Zone Prediction Setup Guide

## Quick Start

1. **Install all dependencies and download data:**
   ```bash
   python3 setup_and_download.py
   ```

2. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook predict_baseball_strike_zones_with_svm.ipynb
   ```

## Manual Setup (if needed)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download player data:**
   ```bash
   python3 download_player_data.py
   ```

## Troubleshooting

### NumPy Compatibility Error
If you see "No module named 'numpy._core.numeric'", run:
```bash
pip uninstall numpy -y
pip install numpy==1.26.4
```

### Missing Pickle Files
If you get "FileNotFoundError" for .p files, make sure to run the download script first.

### Alternative: Use Safe Player Loading
Replace the import in your notebook:
```python
# Instead of:
# from players import aaron_judge, jose_altuve, david_ortiz

# Use:
from players_safe import aaron_judge, jose_altuve, david_ortiz
```

## Files Created
- `aaron_judge.p` - 2017 season pitch data
- `jose_altuve.p` - 2017 season pitch data  
- `david_ortiz.p` - 2016 season pitch data