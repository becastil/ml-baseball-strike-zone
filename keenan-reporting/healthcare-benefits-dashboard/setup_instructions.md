# Setup Instructions for Python 3.13 Users

## Issue
Python 3.13 is very new and some packages don't have pre-built wheels yet, causing installation failures.

## Solutions

### Option 1: Use Minimal Requirements (Recommended)
```bash
# Install only essential packages
pip install pandas openpyxl

# Generate sample data without pandas
python src/generate_sample_data_minimal.py
```

### Option 2: Use Conda
```bash
# Install Anaconda or Miniconda, then:
conda install pandas numpy openpyxl pyyaml
```

### Option 3: Use Python 3.12
```bash
# Install Python 3.12 alongside 3.13
# Then use pip normally:
pip install -r requirements.txt
```

## Testing Without Full Dependencies

1. **Check setup:**
   ```bash
   python quickstart_simple.py
   ```

2. **Generate minimal sample data:**
   ```bash
   python src/generate_sample_data_minimal.py
   ```

3. **View sample data:**
   The CSV file will be in `data/input/sample_healthcare_data_minimal.csv`

## Next Steps

Once you have pandas installed, you can:
1. Run the full data processor
2. Connect Power BI to the processed Excel file
3. Import the theme and watch the dashboard populate

The dashboard will work with the minimal sample data, giving you a feel for the functionality while you resolve the dependency issues.