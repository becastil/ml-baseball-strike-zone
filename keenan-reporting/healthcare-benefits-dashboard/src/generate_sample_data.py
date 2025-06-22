"""
Generate sample healthcare benefits data for testing the dashboard.
Creates realistic claims, enrollment, and cost data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path


class SampleDataGenerator:
    """Generate realistic healthcare benefits sample data."""
    
    def __init__(self, num_employees=500, months=12):
        """Initialize generator with parameters."""
        self.num_employees = num_employees
        self.months = months
        self.start_date = datetime(2024, 1, 1)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Define distributions
        self.coverage_tiers = {
            'EE': 0.40,      # Employee only
            'ES': 0.15,      # Employee + Spouse
            'EC': 0.20,      # Employee + Children
            'FAM': 0.25      # Family
        }
        
        self.plan_types = {
            'Medical': 0.95,  # Most have medical
            'Rx': 0.95,       # Most have pharmacy
            'Dental': 0.80,   # 80% have dental
            'Vision': 0.70    # 70% have vision
        }
        
        self.claim_types = {
            'Medical': 0.60,
            'Pharmacy': 0.35,
            'Fixed': 0.03,
            'Admin': 0.02
        }
        
        # Medical service categories
        self.medical_services = {
            'Office Visit': (100, 300, 0.30),
            'Lab/Diagnostic': (50, 500, 0.25),
            'Emergency Room': (1000, 5000, 0.10),
            'Hospital Inpatient': (5000, 50000, 0.05),
            'Hospital Outpatient': (500, 5000, 0.15),
            'Specialist Visit': (200, 500, 0.15)
        }
        
        # Drug tiers
        self.drug_tiers = {
            'Generic': (10, 50, 0.60),
            'Preferred Brand': (50, 200, 0.25),
            'Non-Preferred Brand': (100, 500, 0.10),
            'Specialty': (1000, 10000, 0.05)
        }
    
    def generate_employees(self):
        """Generate employee master data."""
        employees = []
        
        for i in range(self.num_employees):
            emp_id = f"EMP{i+1:05d}"
            
            # Determine coverage tier
            tier = np.random.choice(
                list(self.coverage_tiers.keys()),
                p=list(self.coverage_tiers.values())
            )
            
            # Base employee record
            employees.append({
                'EmployeeID': emp_id,
                'RelationshipType': 'Employee',
                'CoverageTier': tier,
                'Division': f"Division {random.randint(1, 5)}",
                'Location': random.choice(['NY', 'CA', 'TX', 'FL', 'IL']),
                'EnrollmentStatus': 'Active' if random.random() > 0.05 else 'Termed'
            })
            
            # Add dependents based on tier
            if tier in ['ES', 'FAM']:
                employees.append({
                    'EmployeeID': emp_id,
                    'RelationshipType': 'Spouse',
                    'CoverageTier': tier,
                    'Division': employees[-1]['Division'],
                    'Location': employees[-1]['Location'],
                    'EnrollmentStatus': employees[-1]['EnrollmentStatus']
                })
            
            if tier in ['EC', 'FAM']:
                num_children = random.randint(1, 3)
                for c in range(num_children):
                    employees.append({
                        'EmployeeID': emp_id,
                        'RelationshipType': 'Child',
                        'CoverageTier': tier,
                        'Division': employees[-1]['Division'],
                        'Location': employees[-1]['Location'],
                        'EnrollmentStatus': employees[-1]['EnrollmentStatus']
                    })
        
        return pd.DataFrame(employees)
    
    def generate_claims(self, employees_df):
        """Generate claims data."""
        claims = []
        
        # Get unique employee/member combinations
        members = employees_df[['EmployeeID', 'RelationshipType']].drop_duplicates()
        
        for month in range(self.months):
            report_month = self.start_date + timedelta(days=30 * month)
            
            for _, member in members.iterrows():
                # Determine if member has claims this month (70% chance)
                if random.random() > 0.30:
                    # Generate 1-5 claims
                    num_claims = np.random.poisson(1.5) + 1
                    
                    for _ in range(min(num_claims, 5)):
                        claim_type = np.random.choice(
                            list(self.claim_types.keys()),
                            p=list(self.claim_types.values())
                        )
                        
                        if claim_type == 'Medical':
                            claims.append(self._generate_medical_claim(
                                member, report_month
                            ))
                        elif claim_type == 'Pharmacy':
                            claims.append(self._generate_pharmacy_claim(
                                member, report_month
                            ))
                        elif claim_type == 'Fixed':
                            claims.append(self._generate_fixed_cost(
                                member, report_month
                            ))
        
        return pd.DataFrame(claims)
    
    def _generate_medical_claim(self, member, report_month):
        """Generate a medical claim."""
        service = np.random.choice(
            list(self.medical_services.keys()),
            p=[s[2] for s in self.medical_services.values()]
        )
        
        min_cost, max_cost, _ = self.medical_services[service]
        
        # Generate costs with some high-cost outliers
        if random.random() < 0.02:  # 2% chance of high-cost claim
            billed = random.uniform(max_cost, max_cost * 10)
        else:
            billed = random.uniform(min_cost, max_cost)
        
        allowed = billed * random.uniform(0.4, 0.8)  # Insurance discount
        paid = allowed * random.uniform(0.7, 0.9)     # After deductible/coinsurance
        
        return {
            'ClientID': 'CLIENT_001',
            'EmployeeID': member['EmployeeID'],
            'RelationshipType': member['RelationshipType'],
            'ReportMonth': report_month,
            'ServiceMonth': report_month - timedelta(days=random.randint(0, 45)),
            'PaidMonth': report_month,
            'PlanType': 'Medical',
            'ClaimType': 'Medical',
            'ServiceType': service,
            'BilledAmount': round(billed, 2),
            'AllowedAmount': round(allowed, 2),
            'PaidAmount': round(paid, 2),
            'MemberCopay': round(allowed - paid, 2),
            'RetroAdjustmentFlag': 'Y' if random.random() < 0.05 else 'N'
        }
    
    def _generate_pharmacy_claim(self, member, report_month):
        """Generate a pharmacy claim."""
        tier = np.random.choice(
            list(self.drug_tiers.keys()),
            p=[t[2] for t in self.drug_tiers.values()]
        )
        
        min_cost, max_cost, _ = self.drug_tiers[tier]
        
        # Specialty drugs can be very expensive
        if tier == 'Specialty' and random.random() < 0.3:
            cost = random.uniform(max_cost, max_cost * 3)
        else:
            cost = random.uniform(min_cost, max_cost)
        
        paid = cost * 0.8  # Plan pays 80% on average
        
        return {
            'ClientID': 'CLIENT_001',
            'EmployeeID': member['EmployeeID'],
            'RelationshipType': member['RelationshipType'],
            'ReportMonth': report_month,
            'ServiceMonth': report_month,
            'PaidMonth': report_month,
            'PlanType': 'Rx',
            'ClaimType': 'Pharmacy',
            'DrugTier': tier,
            'BilledAmount': round(cost, 2),
            'AllowedAmount': round(cost, 2),
            'PaidAmount': round(paid, 2),
            'MemberCopay': round(cost - paid, 2),
            'PharmacyRebateAmount': 0,  # Will be calculated later
            'RetroAdjustmentFlag': 'N'
        }
    
    def _generate_fixed_cost(self, member, report_month):
        """Generate fixed cost entries."""
        return {
            'ClientID': 'CLIENT_001',
            'EmployeeID': member['EmployeeID'],
            'RelationshipType': member['RelationshipType'],
            'ReportMonth': report_month,
            'ServiceMonth': report_month,
            'PaidMonth': report_month,
            'PlanType': 'Medical',
            'ClaimType': 'Fixed',
            'ServiceType': 'Admin Fee',
            'BilledAmount': 35.00,
            'AllowedAmount': 35.00,
            'PaidAmount': 35.00,
            'MemberCopay': 0,
            'RetroAdjustmentFlag': 'N'
        }
    
    def add_high_cost_claimants(self, claims_df):
        """Add some high-cost claimants to make data more realistic."""
        # Select 1-2% of employees to be high-cost
        unique_employees = claims_df['EmployeeID'].unique()
        num_high_cost = max(1, int(len(unique_employees) * 0.015))
        high_cost_employees = np.random.choice(unique_employees, num_high_cost, replace=False)
        
        new_claims = []
        
        for emp_id in high_cost_employees:
            # Generate a catastrophic claim
            catastrophic_month = np.random.choice(claims_df['ReportMonth'].unique())
            
            claim = {
                'ClientID': 'CLIENT_001',
                'EmployeeID': emp_id,
                'RelationshipType': 'Employee',
                'ReportMonth': catastrophic_month,
                'ServiceMonth': catastrophic_month,
                'PaidMonth': catastrophic_month,
                'PlanType': 'Medical',
                'ClaimType': 'Medical',
                'ServiceType': 'Hospital Inpatient - Catastrophic',
                'BilledAmount': round(random.uniform(200000, 500000), 2),
                'AllowedAmount': round(random.uniform(150000, 400000), 2),
                'PaidAmount': round(random.uniform(100000, 350000), 2),
                'MemberCopay': 5000.00,  # Out-of-pocket maximum
                'RetroAdjustmentFlag': 'N'
            }
            
            new_claims.append(claim)
        
        return pd.concat([claims_df, pd.DataFrame(new_claims)], ignore_index=True)
    
    def add_enrollment_data(self, claims_df, employees_df):
        """Add enrollment data to claims."""
        # Merge with employee data
        claims_df = claims_df.merge(
            employees_df[['EmployeeID', 'CoverageTier', 'Division', 'Location', 'EnrollmentStatus']],
            on='EmployeeID',
            how='left'
        )
        
        # Add enrollment dates
        claims_df['EnrollmentStartDate'] = self.start_date - timedelta(days=random.randint(0, 365))
        claims_df['EnrollmentEndDate'] = pd.NaT
        
        # For termed employees, add end date
        termed_mask = claims_df['EnrollmentStatus'] == 'Termed'
        claims_df.loc[termed_mask, 'EnrollmentEndDate'] = (
            claims_df.loc[termed_mask, 'ReportMonth'] + timedelta(days=random.randint(0, 90))
        )
        
        return claims_df
    
    def add_calculated_fields(self, df):
        """Add calculated fields that would normally come from processing."""
        # Add member months (simplified)
        df['MemberMonths'] = 1
        
        # Add stop-loss fields
        df['StopLossEligibleAmount'] = 0
        df['StopLossReimbursement'] = 0
        
        # Calculate stop-loss for high-cost claims
        employee_totals = df.groupby(['EmployeeID', 'ReportMonth'])['PaidAmount'].sum()
        high_cost_members = employee_totals[employee_totals > 250000]
        
        for (emp_id, month), total in high_cost_members.items():
            mask = (df['EmployeeID'] == emp_id) & (df['ReportMonth'] == month)
            df.loc[mask, 'StopLossEligibleAmount'] = total - 250000
            # Reimbursement comes 2 months later
            future_month = month + pd.DateOffset(months=2)
            future_mask = (df['EmployeeID'] == emp_id) & (df['ReportMonth'] == future_month)
            if future_mask.any():
                df.loc[future_mask.idxmax(), 'StopLossReimbursement'] = total - 250000
        
        # Calculate totals
        df['TotalEnrolled'] = df.groupby('ReportMonth')['EmployeeID'].transform('nunique')
        df['EmployeeCount'] = df[df['RelationshipType'] == 'Employee'].groupby('ReportMonth')['EmployeeID'].transform('nunique')
        
        # Add budget fields
        df['BudgetedPMPM'] = 500.00
        
        return df
    
    def generate_complete_dataset(self):
        """Generate complete sample dataset."""
        print("Generating employee data...")
        employees_df = self.generate_employees()
        
        print("Generating claims data...")
        claims_df = self.generate_claims(employees_df)
        
        print("Adding high-cost claimants...")
        claims_df = self.add_high_cost_claimants(claims_df)
        
        print("Adding enrollment data...")
        claims_df = self.add_enrollment_data(claims_df, employees_df)
        
        print("Adding calculated fields...")
        final_df = self.add_calculated_fields(claims_df)
        
        return final_df
    
    def save_sample_data(self, output_dir=None):
        """Generate and save sample data files."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'data' / 'input'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate data
        df = self.generate_complete_dataset()
        
        # Save as CSV
        csv_file = output_dir / 'sample_healthcare_data.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved sample data to: {csv_file}")
        
        # Save as Excel
        excel_file = output_dir / 'sample_healthcare_data.xlsx'
        df.to_excel(excel_file, index=False)
        print(f"Saved sample data to: {excel_file}")
        
        # Generate summary statistics
        print("\nSummary Statistics:")
        print(f"Total records: {len(df):,}")
        print(f"Unique employees: {df['EmployeeID'].nunique():,}")
        print(f"Date range: {df['ReportMonth'].min()} to {df['ReportMonth'].max()}")
        print(f"Total paid amount: ${df['PaidAmount'].sum():,.2f}")
        print(f"Average PMPM: ${df['PaidAmount'].sum() / df['MemberMonths'].sum():.2f}")
        
        return df


if __name__ == '__main__':
    # Generate sample data
    generator = SampleDataGenerator(num_employees=500, months=12)
    generator.save_sample_data()