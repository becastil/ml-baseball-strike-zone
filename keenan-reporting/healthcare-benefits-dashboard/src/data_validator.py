"""
Data validation module for healthcare benefits data.
Provides comprehensive validation of data structure, quality, and business rules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging


class DataValidator:
    """Validates healthcare benefits data against defined rules."""
    
    def __init__(self, config_manager):
        """Initialize validator with configuration."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Define required columns
        self.required_columns = [
            'ClientID', 'ReportMonth', 'PlanType', 'EmployeeID',
            'ClaimType', 'PaidAmount', 'AllowedAmount'
        ]
        
        # Define column data types
        self.column_types = {
            'ClientID': 'object',
            'ReportMonth': 'datetime64[ns]',
            'PlanType': 'object',
            'EmployeeID': 'object',
            'ClaimType': 'object',
            'PaidAmount': 'float64',
            'AllowedAmount': 'float64',
            'PharmacyRebateAmount': 'float64',
            'StopLossReimbursement': 'float64'
        }
        
        # Valid values for categorical fields
        self.valid_values = {
            'PlanType': ['Medical', 'Rx', 'Dental', 'Vision'],
            'ClaimType': ['Medical', 'Pharmacy', 'Fixed', 'Admin'],
            'RelationshipType': ['Employee', 'Spouse', 'Child', 'Dependent'],
            'CoverageTier': ['EE', 'ES', 'EC', 'FAM'],
            'EnrollmentStatus': ['Active', 'Termed', 'COBRA'],
            'RetroAdjustmentFlag': ['Y', 'N']
        }
    
    def validate_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the structure of the input dataframe.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check for required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            results['is_valid'] = False
            results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        for col, expected_type in self.column_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type and not self._compatible_types(actual_type, expected_type):
                    results['warnings'].append(
                        f"Column '{col}' has type '{actual_type}', expected '{expected_type}'"
                    )
        
        # Check for empty dataframe
        if len(df) == 0:
            results['is_valid'] = False
            results['errors'].append("Input data is empty")
        
        return results
    
    def validate_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform data quality checks on the dataframe.
        
        Returns:
            Dictionary with quality check results
        """
        results = {
            'warnings': [],
            'info': [],
            'quality_score': 100.0
        }
        
        # Check for null values in critical fields
        critical_fields = ['EmployeeID', 'ReportMonth', 'PaidAmount']
        for field in critical_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    results['warnings'].append(f"{field} has {null_count} null values")
                    results['quality_score'] -= 5
        
        # Check for duplicate records
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            results['warnings'].append(f"Found {duplicate_count} duplicate records")
            results['quality_score'] -= 10
        
        # Validate categorical values
        for field, valid_vals in self.valid_values.items():
            if field in df.columns:
                invalid_vals = df[~df[field].isin(valid_vals + [None, np.nan])][field].unique()
                if len(invalid_vals) > 0:
                    results['warnings'].append(
                        f"{field} contains invalid values: {invalid_vals[:5]}"
                    )
                    results['quality_score'] -= 3
        
        # Check for outliers in paid amounts
        outliers = self._detect_outliers(df, 'PaidAmount')
        if len(outliers) > 0:
            results['info'].append(f"Found {len(outliers)} potential outliers in PaidAmount")
        
        # Check date continuity
        date_gaps = self._check_date_continuity(df)
        if date_gaps:
            results['warnings'].append(f"Missing data for months: {date_gaps}")
            results['quality_score'] -= 5
        
        # Validate business rules
        business_rules_results = self._validate_business_rules(df)
        results['warnings'].extend(business_rules_results['warnings'])
        results['quality_score'] -= len(business_rules_results['warnings']) * 2
        
        # Ensure quality score doesn't go below 0
        results['quality_score'] = max(0, results['quality_score'])
        
        return results
    
    def _compatible_types(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible."""
        # Allow object type for string fields
        if expected == 'object' and actual in ['object', 'string']:
            return True
        
        # Allow numeric compatibility
        numeric_types = ['int64', 'float64', 'int32', 'float32']
        if expected in numeric_types and actual in numeric_types:
            return True
        
        return False
    
    def _detect_outliers(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using z-score method."""
        if column not in df.columns:
            return pd.Series()
        
        # Calculate z-scores
        mean = df[column].mean()
        std = df[column].std()
        
        if std == 0:
            return pd.Series()
        
        z_scores = np.abs((df[column] - mean) / std)
        return df[z_scores > threshold]
    
    def _check_date_continuity(self, df: pd.DataFrame) -> List[str]:
        """Check for missing months in the data."""
        if 'ReportMonth' not in df.columns:
            return []
        
        # Get date range
        min_date = df['ReportMonth'].min()
        max_date = df['ReportMonth'].max()
        
        # Generate expected date range
        expected_dates = pd.date_range(start=min_date, end=max_date, freq='MS')
        actual_dates = df['ReportMonth'].unique()
        
        # Find missing dates
        missing_dates = set(expected_dates) - set(actual_dates)
        
        return [d.strftime('%Y-%m') for d in sorted(missing_dates)]
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate specific business rules."""
        results = {'warnings': []}
        
        # Rule 1: Allowed amount should be >= Paid amount
        if 'AllowedAmount' in df.columns and 'PaidAmount' in df.columns:
            invalid_claims = df[df['AllowedAmount'] < df['PaidAmount']]
            if len(invalid_claims) > 0:
                results['warnings'].append(
                    f"{len(invalid_claims)} claims have AllowedAmount < PaidAmount"
                )
        
        # Rule 2: Pharmacy rebates should be reasonable (10-30% of gross cost)
        if 'PharmacyRebateAmount' in df.columns and 'ClaimType' in df.columns:
            pharmacy_claims = df[df['ClaimType'] == 'Pharmacy']
            if len(pharmacy_claims) > 0:
                rebate_pct = (pharmacy_claims['PharmacyRebateAmount'].sum() / 
                             pharmacy_claims['PaidAmount'].sum() * 100)
                if rebate_pct < 10 or rebate_pct > 30:
                    results['warnings'].append(
                        f"Pharmacy rebate percentage ({rebate_pct:.1f}%) outside expected range (10-30%)"
                    )
        
        # Rule 3: Stop-loss reimbursements should have corresponding high claims
        if 'StopLossReimbursement' in df.columns:
            sl_config = self.config.get('stop_loss', {})
            attachment_point = sl_config.get('attachment_point', 250000)
            
            # Check if we have members with stop-loss reimbursements
            sl_members = df[df['StopLossReimbursement'] > 0]['EmployeeID'].unique()
            for member in sl_members:
                member_claims = df[df['EmployeeID'] == member]['PaidAmount'].sum()
                if member_claims < attachment_point:
                    results['warnings'].append(
                        f"Member {member} has stop-loss reimbursement but claims below attachment point"
                    )
        
        # Rule 4: Check for reasonable enrollment changes
        if 'TotalEnrolled' in df.columns and 'ReportMonth' in df.columns:
            monthly_enrollment = df.groupby('ReportMonth')['TotalEnrolled'].first().sort_index()
            pct_changes = monthly_enrollment.pct_change().abs()
            large_changes = pct_changes[pct_changes > 0.1]
            if len(large_changes) > 0:
                results['warnings'].append(
                    f"Large enrollment changes (>10%) detected in {len(large_changes)} months"
                )
        
        return results
    
    def generate_validation_report(self, structure_results: Dict, quality_results: Dict) -> str:
        """Generate a formatted validation report."""
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Structure validation
        report.append("STRUCTURE VALIDATION")
        report.append("-" * 30)
        report.append(f"Valid Structure: {'YES' if structure_results['is_valid'] else 'NO'}")
        
        if structure_results['errors']:
            report.append("\nErrors:")
            for error in structure_results['errors']:
                report.append(f"  - {error}")
        
        if structure_results['warnings']:
            report.append("\nWarnings:")
            for warning in structure_results['warnings']:
                report.append(f"  - {warning}")
        
        # Quality validation
        report.append("\n\nDATA QUALITY VALIDATION")
        report.append("-" * 30)
        report.append(f"Quality Score: {quality_results['quality_score']:.1f}/100")
        
        if quality_results['warnings']:
            report.append("\nWarnings:")
            for warning in quality_results['warnings']:
                report.append(f"  - {warning}")
        
        if quality_results['info']:
            report.append("\nInformation:")
            for info in quality_results['info']:
                report.append(f"  - {info}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)