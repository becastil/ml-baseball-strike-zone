# Power BI Dashboard Setup Guide

## Step 1: Initial Setup
1. In Power BI, ensure your theme is applied (View → Themes → Browse for themes → select healthcare_dashboard_theme.json)
2. Go to the Data view and verify your CSV is loaded
3. Change data types if needed:
   - ReportMonth → Date
   - All amount fields → Decimal Number

## Step 2: Create Date Table (Optional but Recommended)
In Modeling → New Table, paste:
```
DateTable = 
CALENDAR(
    MIN('sample_healthcare_data_minimal'[ReportMonth]),
    MAX('sample_healthcare_data_minimal'[ReportMonth])
)
```

## Step 3: Create Measures
1. Right-click on your data table → New Measure
2. Copy each measure from `power_bi_measures.txt`
3. Organize measures in a folder: Right-click measure → Move to folder → New folder "Measures"

## Step 4: Build Dashboard Pages

### Page 1: Executive Summary

**Top Section - KPI Cards (4 across)**
1. Insert 4 Card visuals across the top
2. Configure each card:

   **Card 1 - Total Net Cost**
   - Value: Total Net Cost
   - Title: "Total Net Cost"
   - Format: Currency, 0 decimals
   
   **Card 2 - Net PMPM**
   - Value: Net PMPM
   - Title: "Cost PMPM"
   - Format: Currency, 2 decimals
   
   **Card 3 - Member Count**
   - Value: Member Count
   - Title: "Total Members"
   - Format: Whole Number
   
   **Card 4 - Budget Variance**
   - Value: Budget Variance %
   - Title: "vs Budget"
   - Format: Percentage, 1 decimal
   - Conditional formatting on font color using Cost Status Color measure

**Middle Section - Trend Chart**
1. Insert Line Chart
2. X-axis: ReportMonth
3. Y-axis: Total Paid Amount
4. Add reference line: Budget PMPM * Total Member Months
5. Title: "Monthly Cost Trend vs Budget"

**Bottom Section - Cost Breakdown**
1. Insert Donut Chart
2. Values: Medical Costs, Net Pharmacy Costs
3. Title: "Cost by Type"

### Page 2: Cost Analysis

**Layout: 2x2 Grid**

**Top Left - Waterfall Chart**
1. Insert Waterfall Chart
2. Category: Create manual categories
3. Values: Medical Costs (increase), Pharmacy Costs (increase), Pharmacy Rebates (decrease)
4. Title: "Net Cost Build-Up"

**Top Right - Table**
1. Insert Table
2. Columns: Division, Medical Costs, Pharmacy Costs, Total Paid Amount, Member Count, Gross PMPM
3. Title: "Cost by Division"

**Bottom Left - Clustered Bar Chart**
1. Insert Clustered Bar Chart
2. Y-axis: Coverage Tier
3. Values: Member Count
4. Title: "Enrollment by Coverage Tier"

**Bottom Right - KPI Visual**
1. Insert KPI
2. Indicator: Net PMPM
3. Target: Budget PMPM
4. Trend axis: ReportMonth

### Page 3: Utilization & High Cost

**Top Section - High Cost Summary Cards**
1. Three cards showing:
   - High Cost Members (count)
   - High Cost Claims Amount
   - % of Total Cost (High Cost Claims Amount / Total Paid Amount)

**Middle Section - Scatter Chart**
1. Insert Scatter Chart
2. X-axis: Claims per Member
3. Y-axis: Average Claim Cost
4. Values: Member Count (bubble size)
5. Legend: Division
6. Title: "Utilization Patterns by Division"

**Bottom Section - Top N Table**
1. Insert Table
2. Add Top N filter (Top 10 by Total Paid Amount)
3. Group by EmployeeID
4. Columns: EmployeeID, Total Paid Amount, Claims Count
5. Title: "Top 10 High Cost Members (Anonymized)"

### Page 4: Filters & Slicers

**Create Interactive Filter Panel**
1. Insert Slicers for:
   - ReportMonth (Date range)
   - Division (Dropdown)
   - PlanType (Buttons)
   - Coverage Tier (List)
   - Location (Dropdown)

2. Sync slicers across all pages:
   - Select slicer → View → Sync slicers
   - Check all pages

## Step 5: Formatting Tips

### For All Visuals:
1. Turn on rounded corners (8px radius)
2. Add subtle shadows
3. Use consistent padding (10px)
4. Align to grid (View → Gridlines)

### Color Usage:
- Primary (#1E4D8B): Headers, main metrics
- Secondary (#16A085): Positive values
- Negative (#E74C3C): Over budget, alerts
- Neutral (#95A5A6): Secondary text

### Fonts:
- Titles: 14pt Roboto Medium
- Values: 28-32pt Roboto Bold (for KPIs)
- Labels: 10-11pt Roboto Regular

## Step 6: Performance Optimization

1. **Data Model**:
   - Keep only necessary columns
   - Set proper data types
   - Create relationships if using date table

2. **Report Settings**:
   - File → Options → Report settings
   - Disable "Allow users to change filter types"
   - Enable "Persistent filters"

3. **Publish**:
   - Publish to Power BI Service
   - Set up scheduled refresh
   - Configure row-level security if needed

## Quick Tips:

- Use Ctrl+Click to select multiple visuals for alignment
- Alt+Shift+F10 for Selection pane to organize layers
- Use Bookmarks for different views (View → Bookmarks)
- Format Painter to copy formatting between visuals

## Testing Your Dashboard:

1. Verify all measures calculate correctly
2. Test all slicers and filters
3. Check drill-through functionality
4. Ensure mobile layout works (View → Mobile layout)
5. Validate conditional formatting appears correctly