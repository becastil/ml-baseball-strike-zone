# Healthcare Benefits Dashboard

A comprehensive, automated dashboard solution for healthcare and employee benefits reporting, designed for employer clients to track medical claims, pharmacy costs, enrollment trends, and budget performance.

## Features

- **Automated Data Processing**: Python-based ETL pipeline for complex calculations
- **Plug-and-Play Design**: Drop in monthly data files and everything auto-refreshes
- **Modern Visualizations**: Clean, professional Power BI dashboards with KPI cards
- **Comprehensive Analytics**: Track claims, enrollment, high-cost members, and utilization
- **Multi-Client Support**: Configurable for different employers with custom branding
- **Data Validation**: Built-in quality checks and error reporting

## Project Structure

```
healthcare-benefits-dashboard/
├── src/                        # Python source code
│   ├── process_benefits_data.py    # Main data processor
│   ├── data_validator.py           # Data validation module
│   ├── config_manager.py           # Configuration management
│   └── generate_sample_data.py     # Sample data generator
├── config/                     # Configuration files
│   ├── client_template.json       # Template for new clients
│   └── default_config.json        # Default configuration
├── data/                       # Data directories
│   ├── input/                     # Raw data files
│   ├── output/                    # Processed data
│   └── archive/                   # Historical data
├── templates/                  # Power BI templates
│   └── healthcare_dashboard_theme.json
├── logs/                       # Processing logs
├── docs/                       # Documentation
└── requirements.txt           # Python dependencies
```

## Installation

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd healthcare-benefits-dashboard
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data (optional)**
   ```bash
   python src/generate_sample_data.py
   ```

## Usage

### 1. Configure Client Settings

Copy the client template and customize:
```bash
cp config/client_template.json config/your_client.json
```

Edit the configuration with:
- Client ID and name
- Pharmacy rebate percentages
- Stop-loss attachment points
- Fixed costs (admin fees, premiums)
- Budget targets
- Brand colors and logos

### 2. Process Monthly Data

Run the data processor:
```bash
python src/process_benefits_data.py data/input/monthly_claims.csv --config config/your_client.json
```

The processor will:
- Validate data structure and quality
- Calculate pharmacy rebates
- Process stop-loss reimbursements
- Calculate PMPM/PEPM metrics
- Flag high-cost claimants
- Generate Power BI-ready output

### 3. Connect Power BI

1. Open Power BI Desktop
2. Import the theme file: `templates/healthcare_dashboard_theme.json`
3. Get Data → Navigate to `data/output/[client]_processed_[timestamp].xlsx`
4. The dashboard will auto-populate with your data

## Data Requirements

### Input File Format

The system expects CSV or Excel files with these columns:

**Required Fields:**
- `ClientID`: Client identifier
- `ReportMonth`: YYYY-MM-DD format
- `EmployeeID`: Unique employee ID (will be anonymized)
- `PlanType`: Medical, Rx, Dental, or Vision
- `ClaimType`: Medical, Pharmacy, Fixed, or Admin
- `PaidAmount`: Amount paid by plan
- `AllowedAmount`: Contracted amount

**Additional Fields** (see `config/client_template.json` for full list)

### Output Format

The processor generates an Excel file with multiple sheets:
- **Claims_Detail**: All processed claims with calculations
- **Monthly_Summary**: Aggregated metrics by month
- **High_Cost_Members**: Members exceeding threshold

## Dashboard Pages

1. **Executive Summary**
   - Key KPIs: Total cost, PMPM, enrollment, budget variance
   - Cost trend charts with budget overlay
   - Current month highlights

2. **Cost Analysis**
   - Medical vs pharmacy split
   - Stop-loss impact visualization
   - Pharmacy rebate analysis
   - Fixed costs breakdown

3. **Utilization & Claims**
   - Service utilization patterns
   - High-cost claimant analysis
   - ER vs urgent care usage
   - Claims distribution

4. **Enrollment & Demographics**
   - Enrollment trends
   - Coverage tier distribution
   - Employee vs dependent metrics

5. **Data Details**
   - Searchable raw data table
   - Export capabilities

## Calculations

### Pharmacy Rebates
- Configurable by drug tier (generic, brand, specialty)
- Applied as reduction to gross pharmacy cost
- Tracked separately for transparency

### Stop-Loss
- Claims above attachment point eligible for reimbursement
- Configurable lag period (typically 60-90 days)
- Supports specific and aggregate stop-loss

### PMPM/PEPM Metrics
- **PMPM**: Per Member Per Month
- **PEPM**: Per Employee Per Month
- Calculated on both gross and net basis

## Validation & Quality Checks

The system performs automatic validation:
- Missing required columns
- Data type verification
- Outlier detection (configurable z-score)
- Date continuity checks
- Business rule validation
- Budget variance alerts

## Security & Compliance

- Employee IDs are anonymized by default
- No PHI stored in output files
- Configurable data retention policies
- Audit trail logging

## Troubleshooting

### Common Issues

1. **"Missing required columns" error**
   - Check your input file has all required fields
   - Use column mapping in config file if names differ

2. **High outlier warnings**
   - Review high-cost claims for accuracy
   - Adjust outlier threshold in configuration

3. **Power BI refresh fails**
   - Ensure file path hasn't changed
   - Check for Excel file corruption
   - Verify all sheets are present

### Logs

Check `logs/` directory for detailed processing logs with timestamps.

## Support

For issues or questions:
- Review configuration documentation
- Check validation reports
- Contact support team

## License

[Your License Here]