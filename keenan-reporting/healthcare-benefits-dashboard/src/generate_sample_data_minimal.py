#!/usr/bin/env python3
"""
Minimal sample data generator using only standard library.
Creates a simple CSV file for testing without pandas dependency.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_minimal_sample_data():
    """Generate basic sample data without external dependencies."""
    
    # Setup
    output_dir = Path(__file__).parent.parent / 'data' / 'input'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / 'sample_healthcare_data_minimal.csv'
    
    # Define headers
    headers = [
        'ClientID', 'ReportMonth', 'EmployeeID', 'RelationshipType',
        'PlanType', 'ClaimType', 'ServiceMonth', 'PaidMonth',
        'CoverageTier', 'Division', 'Location', 'EnrollmentStatus',
        'BilledAmount', 'AllowedAmount', 'PaidAmount', 'MemberCopay',
        'PharmacyRebateAmount', 'StopLossReimbursement', 'RetroAdjustmentFlag',
        'MemberMonths', 'TotalEnrolled', 'EmployeeCount', 'BudgetedPMPM'
    ]
    
    # Generate data
    rows = []
    start_date = datetime(2024, 1, 1)
    
    for month in range(12):
        report_month = start_date + timedelta(days=30 * month)
        report_month_str = report_month.strftime('%Y-%m-%d')
        
        # Generate 50 employees with claims for this month
        for emp_num in range(1, 51):
            emp_id = f'EMP{emp_num:05d}'
            
            # Employee medical claim
            if random.random() > 0.3:  # 70% have claims
                paid_amount = round(random.uniform(100, 5000), 2)
                allowed_amount = round(paid_amount * 1.2, 2)
                billed_amount = round(allowed_amount * 1.5, 2)
                
                rows.append([
                    'CLIENT_001',  # ClientID
                    report_month_str,  # ReportMonth
                    emp_id,  # EmployeeID
                    'Employee',  # RelationshipType
                    'Medical',  # PlanType
                    'Medical',  # ClaimType
                    report_month_str,  # ServiceMonth
                    report_month_str,  # PaidMonth
                    random.choice(['EE', 'ES', 'EC', 'FAM']),  # CoverageTier
                    f'Division {random.randint(1, 5)}',  # Division
                    random.choice(['NY', 'CA', 'TX', 'FL']),  # Location
                    'Active',  # EnrollmentStatus
                    billed_amount,  # BilledAmount
                    allowed_amount,  # AllowedAmount
                    paid_amount,  # PaidAmount
                    round(allowed_amount - paid_amount, 2),  # MemberCopay
                    0,  # PharmacyRebateAmount
                    0,  # StopLossReimbursement
                    'N',  # RetroAdjustmentFlag
                    1,  # MemberMonths
                    500,  # TotalEnrolled
                    250,  # EmployeeCount
                    500.00  # BudgetedPMPM
                ])
            
            # Pharmacy claim
            if random.random() > 0.5:  # 50% have pharmacy claims
                paid_amount = round(random.uniform(50, 500), 2)
                
                rows.append([
                    'CLIENT_001',  # ClientID
                    report_month_str,  # ReportMonth
                    emp_id,  # EmployeeID
                    'Employee',  # RelationshipType
                    'Rx',  # PlanType
                    'Pharmacy',  # ClaimType
                    report_month_str,  # ServiceMonth
                    report_month_str,  # PaidMonth
                    random.choice(['EE', 'ES', 'EC', 'FAM']),  # CoverageTier
                    f'Division {random.randint(1, 5)}',  # Division
                    random.choice(['NY', 'CA', 'TX', 'FL']),  # Location
                    'Active',  # EnrollmentStatus
                    paid_amount,  # BilledAmount
                    paid_amount,  # AllowedAmount
                    paid_amount * 0.8,  # PaidAmount (80% coverage)
                    paid_amount * 0.2,  # MemberCopay
                    0,  # PharmacyRebateAmount (will be calculated)
                    0,  # StopLossReimbursement
                    'N',  # RetroAdjustmentFlag
                    1,  # MemberMonths
                    500,  # TotalEnrolled
                    250,  # EmployeeCount
                    500.00  # BudgetedPMPM
                ])
    
    # Add one high-cost claimant
    rows.append([
        'CLIENT_001', '2024-06-01', 'EMP00001', 'Employee', 'Medical', 'Medical',
        '2024-06-01', '2024-06-01', 'FAM', 'Division 1', 'NY', 'Active',
        350000, 300000, 280000, 5000, 0, 0, 'N', 1, 500, 250, 500.00
    ])
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"Sample data generated: {output_file}")
    print(f"Total rows: {len(rows)}")
    print(f"Date range: 2024-01-01 to 2024-12-01")
    print("\nYou can now process this file with the main script once dependencies are installed.")


if __name__ == '__main__':
    generate_minimal_sample_data()