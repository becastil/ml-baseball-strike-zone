#!/usr/bin/env python3
"""
Quick start script for Healthcare Benefits Dashboard.
Generates sample data and runs initial processing.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from generate_sample_data import SampleDataGenerator
from process_benefits_data import BenefitsDataProcessor


def main():
    """Run quick start setup."""
    print("=" * 60)
    print("Healthcare Benefits Dashboard - Quick Start")
    print("=" * 60)
    print()
    
    # Check if sample data exists
    input_dir = Path(__file__).parent / 'data' / 'input'
    sample_file = input_dir / 'sample_healthcare_data.csv'
    
    if not sample_file.exists():
        print("Generating sample data...")
        generator = SampleDataGenerator(num_employees=500, months=12)
        generator.save_sample_data()
        print()
    else:
        print(f"Sample data already exists: {sample_file}")
        print()
    
    # Process the sample data
    print("Processing sample data...")
    config_file = Path(__file__).parent / 'config' / 'default_config.json'
    
    processor = BenefitsDataProcessor(str(config_file))
    output_file = processor.process_monthly_data(str(sample_file))
    
    print()
    print("=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Open Power BI Desktop")
    print("2. Import the theme file from: templates/healthcare_dashboard_theme.json")
    print(f"3. Connect to the processed data file: {output_file}")
    print("4. The dashboard will automatically populate with sample data")
    print()
    print("To process your own data:")
    print("1. Copy config/client_template.json to config/your_client.json")
    print("2. Customize the configuration for your client")
    print("3. Run: python src/process_benefits_data.py your_data.csv --config config/your_client.json")
    print()


if __name__ == '__main__':
    main()