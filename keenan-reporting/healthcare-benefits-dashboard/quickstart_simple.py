#!/usr/bin/env python3
"""
Simple quick start script with minimal dependencies.
"""

import os
import sys
from pathlib import Path

def main():
    """Run quick start setup."""
    print("=" * 60)
    print("Healthcare Benefits Dashboard - Setup Check")
    print("=" * 60)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Check directory structure
    print("Project structure created successfully:")
    base_dir = Path(__file__).parent
    for item in sorted(base_dir.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")
    print()
    
    # Installation instructions
    print("To complete setup:")
    print()
    print("1. Install dependencies (try one of these):")
    print("   Option A: pip install pandas openpyxl")
    print("   Option B: conda install pandas openpyxl pyyaml")
    print()
    print("2. For full functionality, install all requirements:")
    print("   pip install -r requirements.txt")
    print()
    print("3. Generate sample data:")
    print("   python src/generate_sample_data.py")
    print()
    print("4. Process the data:")
    print("   python src/process_benefits_data.py data/input/sample_healthcare_data.csv")
    print()
    print("Note: If you have Python 3.13, some packages may need to be")
    print("      installed from conda-forge or wait for updated wheels.")


if __name__ == '__main__':
    main()