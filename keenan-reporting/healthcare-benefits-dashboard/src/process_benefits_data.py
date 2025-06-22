#!/usr/bin/env python3
"""
Healthcare Benefits Data Processing Pipeline
Handles complex calculations for employer benefits reporting including:
- Pharmacy rebate netting
- Stop-loss accrual and reimbursement matching
- Retroactive enrollment adjustments
- PMPM/PEPM calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from data_validator import DataValidator
    from config_manager import ConfigManager
except ImportError:
    from src.data_validator import DataValidator
    from src.config_manager import ConfigManager


class BenefitsDataProcessor:
    """Main class for processing healthcare benefits data."""
    
    def __init__(self, config_path: str):
        """Initialize processor with client configuration."""
        self.config = ConfigManager(config_path)
        self.validator = DataValidator(self.config)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the processor."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path(__file__).parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(
            log_dir / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def process_monthly_data(self, input_file: str) -> pd.DataFrame:
        """
        Main processing function for monthly benefits data.
        
        Args:
            input_file: Path to input CSV/Excel file
            
        Returns:
            Processed DataFrame ready for Power BI
        """
        self.logger.info(f"Starting processing for: {input_file}")
        
        # Load data
        df = self._load_data(input_file)
        
        # Validate data structure
        validation_results = self.validator.validate_structure(df)
        if not validation_results['is_valid']:
            self.logger.error(f"Data validation failed: {validation_results['errors']}")
            raise ValueError(f"Data validation failed: {validation_results['errors']}")
        
        # Core processing steps
        df = self._standardize_dates(df)
        df = self._process_enrollment(df)
        df = self._calculate_pharmacy_rebates(df)
        df = self._process_stop_loss(df)
        df = self._calculate_pmpm_metrics(df)
        df = self._add_budget_variance(df)
        df = self._flag_high_cost_claimants(df)
        
        # Data quality checks
        quality_results = self.validator.validate_quality(df)
        if quality_results['warnings']:
            self.logger.warning(f"Data quality warnings: {quality_results['warnings']}")
        
        # Save processed data
        output_path = self._save_output(df)
        self.logger.info(f"Processing complete. Output saved to: {output_path}")
        
        return df
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV or Excel file."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        self.logger.info(f"Loaded {len(df)} rows from {file_path.name}")
        return df
    
    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert and standardize all date columns."""
        date_columns = ['ReportMonth', 'ServiceMonth', 'PaidMonth', 
                       'EnrollmentStartDate', 'EnrollmentEndDate']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Add derived date fields
        df['ReportYear'] = df['ReportMonth'].dt.year
        df['ReportMonthName'] = df['ReportMonth'].dt.strftime('%B')
        df['ReportQuarter'] = df['ReportMonth'].dt.quarter
        
        return df
    
    def _process_enrollment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process enrollment data including retro adjustments."""
        # Calculate member months
        df['MemberMonths'] = df.groupby(['EmployeeID', 'ReportMonth'])['EmployeeID'].transform('count')
        
        # Handle retroactive adjustments
        retro_mask = df['RetroAdjustmentFlag'] == 'Y'
        df.loc[retro_mask, 'AdjustmentType'] = 'Retroactive'
        
        # Calculate enrollment by tier
        tier_counts = df.groupby(['ReportMonth', 'CoverageTier'])['EmployeeID'].nunique()
        
        # Add enrollment metrics
        df['TotalEnrolled'] = df.groupby('ReportMonth')['EmployeeID'].transform('nunique')
        df['EmployeeCount'] = df[df['RelationshipType'] == 'Employee'].groupby('ReportMonth')['EmployeeID'].transform('nunique')
        
        return df
    
    def _calculate_pharmacy_rebates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate and apply pharmacy rebates based on configuration."""
        # Get rebate configuration
        rebate_config = self.config.get('pharmacy_rebates', {})
        
        # Apply rebates to pharmacy claims
        pharmacy_mask = df['ClaimType'] == 'Pharmacy'
        
        if 'flat_percentage' in rebate_config:
            # Simple percentage rebate
            rebate_pct = rebate_config['flat_percentage']
            df.loc[pharmacy_mask, 'PharmacyRebateAmount'] = (
                df.loc[pharmacy_mask, 'PaidAmount'] * rebate_pct
            )
        else:
            # Tiered rebates based on drug type
            df['PharmacyRebateAmount'] = 0
            for tier, pct in rebate_config.get('tiers', {}).items():
                tier_mask = pharmacy_mask & (df['DrugTier'] == tier)
                df.loc[tier_mask, 'PharmacyRebateAmount'] = (
                    df.loc[tier_mask, 'PaidAmount'] * pct
                )
        
        # Calculate net pharmacy cost
        df['NetPharmacyCost'] = df['PaidAmount'] - df['PharmacyRebateAmount'].fillna(0)
        
        # Log rebate impact
        total_rebates = df['PharmacyRebateAmount'].sum()
        total_rx_cost = df.loc[pharmacy_mask, 'PaidAmount'].sum()
        if total_rx_cost > 0:
            rebate_pct = (total_rebates / total_rx_cost) * 100
            self.logger.info(f"Total pharmacy rebates: ${total_rebates:,.2f} ({rebate_pct:.1f}% of Rx cost)")
        
        return df
    
    def _process_stop_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process stop-loss calculations and reimbursements."""
        sl_config = self.config.get('stop_loss', {})
        attachment_point = sl_config.get('attachment_point', 250000)
        
        # Calculate stop-loss eligible amounts by member
        member_claims = df.groupby(['EmployeeID', 'ReportMonth'])['PaidAmount'].sum().reset_index()
        member_claims['StopLossEligible'] = np.maximum(
            member_claims['PaidAmount'] - attachment_point, 0
        )
        
        # Merge back to main dataframe
        df = df.merge(
            member_claims[['EmployeeID', 'ReportMonth', 'StopLossEligible']],
            on=['EmployeeID', 'ReportMonth'],
            how='left',
            suffixes=('', '_calc')
        )
        
        # Apply reimbursement lag (typically 60-90 days)
        lag_months = sl_config.get('reimbursement_lag_months', 2)
        df['ExpectedReimbursementMonth'] = df['ReportMonth'] + pd.DateOffset(months=lag_months)
        
        # Calculate accrued vs received
        df['StopLossAccrued'] = df['StopLossEligible']
        
        return df
    
    def _calculate_pmpm_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Per Member Per Month (PMPM) and Per Employee Per Month (PEPM) metrics."""
        # Aggregate by month
        monthly_summary = df.groupby('ReportMonth').agg({
            'PaidAmount': 'sum',
            'AllowedAmount': 'sum',
            'PharmacyRebateAmount': 'sum',
            'StopLossReimbursement': 'sum',
            'MemberMonths': 'sum',
            'EmployeeCount': 'first',
            'TotalEnrolled': 'first'
        }).reset_index()
        
        # Calculate net cost
        monthly_summary['NetPlanCost'] = (
            monthly_summary['PaidAmount'] - 
            monthly_summary['PharmacyRebateAmount'].fillna(0) - 
            monthly_summary['StopLossReimbursement'].fillna(0)
        )
        
        # Calculate PMPM and PEPM
        monthly_summary['GrossPMPM'] = monthly_summary['PaidAmount'] / monthly_summary['MemberMonths']
        monthly_summary['NetPMPM'] = monthly_summary['NetPlanCost'] / monthly_summary['MemberMonths']
        monthly_summary['GrossPEPM'] = monthly_summary['PaidAmount'] / monthly_summary['EmployeeCount']
        monthly_summary['NetPEPM'] = monthly_summary['NetPlanCost'] / monthly_summary['EmployeeCount']
        
        # Add fixed costs
        fixed_costs = self.config.get('fixed_costs', {})
        monthly_summary['AdminFeesTotal'] = (
            monthly_summary['EmployeeCount'] * fixed_costs.get('admin_fee_pepm', 0)
        )
        monthly_summary['StopLossPremiumTotal'] = (
            monthly_summary['EmployeeCount'] * fixed_costs.get('stop_loss_premium_pepm', 0)
        )
        
        # Merge back to main dataframe
        df = df.merge(
            monthly_summary[['ReportMonth', 'NetPMPM', 'NetPEPM', 'NetPlanCost']],
            on='ReportMonth',
            how='left'
        )
        
        return df
    
    def _add_budget_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate budget variance metrics."""
        budget_config = self.config.get('budget', {})
        
        # Add budget fields
        df['BudgetedPMPM'] = budget_config.get('pmpm', 500)
        df['BudgetedCost'] = df['BudgetedPMPM'] * df['MemberMonths']
        
        # Calculate variances
        df['BudgetVarianceAmount'] = df['NetPlanCost'] - df['BudgetedCost']
        df['BudgetVariancePercent'] = (df['BudgetVarianceAmount'] / df['BudgetedCost']) * 100
        
        # Add variance categories
        df['VarianceCategory'] = pd.cut(
            df['BudgetVariancePercent'],
            bins=[-np.inf, -10, -5, 5, 10, np.inf],
            labels=['Significantly Under', 'Under Budget', 'On Target', 'Over Budget', 'Significantly Over']
        )
        
        return df
    
    def _flag_high_cost_claimants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify and flag high-cost claimants."""
        # Calculate total claims by member
        member_totals = df.groupby(['EmployeeID', 'ReportMonth'])['PaidAmount'].sum().reset_index()
        
        # Flag high-cost claimants (top 1% or above threshold)
        threshold = self.config.get('high_cost_threshold', 50000)
        member_totals['HighCostFlag'] = member_totals['PaidAmount'] > threshold
        
        # Calculate percentile
        member_totals['CostPercentile'] = member_totals.groupby('ReportMonth')['PaidAmount'].rank(pct=True)
        
        # Merge flags back
        df = df.merge(
            member_totals[['EmployeeID', 'ReportMonth', 'HighCostFlag', 'CostPercentile']],
            on=['EmployeeID', 'ReportMonth'],
            how='left'
        )
        
        return df
    
    def _save_output(self, df: pd.DataFrame) -> str:
        """Save processed data to output directory."""
        output_dir = Path(__file__).parent.parent / 'data' / 'output'
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        client_id = self.config.get('client_id', 'default')
        output_file = output_dir / f"{client_id}_processed_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        # Also save as Excel with multiple sheets for Power BI
        excel_file = output_file.with_suffix('.xlsx')
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Main data
            df.to_excel(writer, sheet_name='Claims_Detail', index=False)
            
            # Monthly summary
            monthly_summary = df.groupby('ReportMonth').agg({
                'PaidAmount': 'sum',
                'NetPlanCost': 'first',
                'NetPMPM': 'first',
                'TotalEnrolled': 'first',
                'BudgetVariancePercent': 'first'
            }).reset_index()
            monthly_summary.to_excel(writer, sheet_name='Monthly_Summary', index=False)
            
            # High cost summary
            high_cost = df[df['HighCostFlag'] == True].groupby(
                ['ReportMonth', 'EmployeeID']
            )['PaidAmount'].sum().reset_index()
            high_cost.to_excel(writer, sheet_name='High_Cost_Members', index=False)
        
        return str(excel_file)


def main():
    """Main entry point for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process healthcare benefits data')
    parser.add_argument('input_file', help='Path to input data file')
    parser.add_argument('--config', default='config/default_config.json', 
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Process data
    processor = BenefitsDataProcessor(args.config)
    processor.process_monthly_data(args.input_file)


if __name__ == '__main__':
    main()