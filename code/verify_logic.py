import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import load_and_clean_data, prepare_variables
from src.econometrics import estimate_tfp_panel_fe

def extended_verification():
    print("\n" + "=" * 70)
    print("EXTENDED VERIFICATION SCRIPT (ROUND 2)")
    print("=" * 70)

    # 1. Load Data
    df = load_and_clean_data()
    df = prepare_variables(df)

    # === CHECK 1: DATA PERIOD ===
    print("\n[CHECK 1: DATA PERIOD]")
    print(f"Min Year: {df['Year'].min()}")
    print(f"Max Year: {df['Year'].max()}")
    print(f"Thesis claims: 2015-2024")
    if df['Year'].min() != 2015 or df['Year'].max() != 2024:
        print(">>> DISCREPANCY FOUND: Code data period != Thesis claim")
    else:
        print(">>> OK: Period matches")

    # === CHECK 2: DEFLATION ===
    print("\n[CHECK 2: DEFLATION]")
    print("Thesis (Line 117) claims: 'Deflated Value Added'")
    print("Code (data_loader.py): No deflation applied, uses raw VA from AIDA")
    print(">>> DISCREPANCY: Thesis text is incorrect OR deflation must be added")

    # 2. Estimate TFP
    df, _ = estimate_tfp_panel_fe(df)

    # === CHECK 3: REGIONAL TFP COMPARISON ===
    print("\n[CHECK 3: REGIONAL TFP GAP]")
    df['IsSouth'] = (df['MacroArea'] == 'Sud').astype(int)
    df['IsCentroNord'] = (df['MacroArea'].isin(['Nord', 'Centro'])).astype(int)
    
    mean_tfp_south = df[df['IsSouth'] == 1]['TFP'].mean()
    mean_tfp_north = df[df['IsCentroNord'] == 1]['TFP'].mean()
    median_tfp_south = df[df['IsSouth'] == 1]['TFP'].median()
    median_tfp_north = df[df['IsCentroNord'] == 1]['TFP'].median()
    
    std_tfp_south = df[df['IsSouth'] == 1]['TFP'].std()
    std_tfp_north = df[df['IsCentroNord'] == 1]['TFP'].std()
    
    n_south = df[df['IsSouth'] == 1]['firm_id'].nunique()
    n_north = df[df['IsCentroNord'] == 1]['firm_id'].nunique()
    
    print(f"N firms (Center-North): {n_north}")
    print(f"N firms (South):        {n_south}")
    print(f"Mean TFP (Center-North):   {mean_tfp_north:.4f} (std: {std_tfp_north:.4f})")
    print(f"Mean TFP (South):          {mean_tfp_south:.4f} (std: {std_tfp_south:.4f})")
    print(f"Median TFP (Center-North): {median_tfp_north:.4f}")
    print(f"Median TFP (South):        {median_tfp_south:.4f}")
    
    print(f"\nMean GAP (N - S):   {mean_tfp_north - mean_tfp_south:.4f}")
    print(f"Median GAP (N - S): {median_tfp_north - median_tfp_south:.4f}")
    
    # Thesis claim (Line 166): "Center-North distribution displays a higher median"
    if median_tfp_north > median_tfp_south:
        print(">>> Thesis claim (L166) SUPPORTED: CN median > S median")
    else:
        print(">>> Thesis claim (L166) CONTRADICTION: CN median <= S median")

    # === CHECK 4: LABOR PRODUCTIVITY (Y/L) ===
    print("\n[CHECK 4: LABOR PRODUCTIVITY (Y/L)]")
    mean_lp_south = df[df['IsSouth'] == 1]['LaborProd'].mean()
    mean_lp_north = df[df['IsCentroNord'] == 1]['LaborProd'].mean()
    median_lp_south = df[df['IsSouth'] == 1]['LaborProd'].median()
    median_lp_north = df[df['IsCentroNord'] == 1]['LaborProd'].median()
    
    print(f"Mean LaborProd (Center-North): {mean_lp_north:.4f}")
    print(f"Mean LaborProd (South):        {mean_lp_south:.4f}")
    print(f"Median LaborProd (Center-North): {median_lp_north:.4f}")
    print(f"Median LaborProd (South):        {median_lp_south:.4f}")

    # === CHECK 5: J-CURVE DIRECTION ===
    print("\n[CHECK 5: J-CURVE LOGIC]")
    mean_tech_south = df[df['IsSouth'] == 1]['TechIntensity'].mean()
    mean_tech_north = df[df['IsCentroNord'] == 1]['TechIntensity'].mean()
    
    df['Tech_Sq'] = df['TechIntensity'] ** 2
    X = df[['TechIntensity', 'Tech_Sq']]
    X = sm.add_constant(X)
    model = sm.OLS(df['TFP'], X).fit()
    gamma1 = model.params['TechIntensity']
    gamma2 = model.params['Tech_Sq']
    turning_point = -gamma1 / (2 * gamma2)
    
    print(f"Mean Tech (Center-North): {mean_tech_north:.4%}")
    print(f"Mean Tech (South):        {mean_tech_south:.4%}")
    print(f"Turning Point:            {turning_point:.4%}")
    print(f"Gamma1: {gamma1:.4f}, Gamma2: {gamma2:.4f}")
    
    # Are both regions below turning point?
    if mean_tech_north < turning_point and mean_tech_south < turning_point:
        print(">>> Both regions are in the INVESTMENT DIP (below turning point)")
        print("    J-Curve CANNOT explain a positive N-S gap if N has more tech")
        if mean_tech_north > mean_tech_south:
            print("    N has MORE tech => N suffers more => N should be LOWER (ceteris paribus)")
            print("    Actual data: " + ("N > S (supports thesis)" if mean_tfp_north > mean_tfp_south else "N <= S (contradicts thesis narrative)"))
    
    # === FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    issues = []
    
    if df['Year'].min() != 2015 or df['Year'].max() != 2024:
        issues.append("- Data period mismatch (1996-2025 vs 2015-2024)")
    issues.append("- Missing deflation (thesis claims deflated, code uses nominal)")
    
    if median_tfp_north <= median_tfp_south:
        issues.append("- TFP Median: Data shows S >= N (contradicts thesis L166)")
    
    if mean_tech_north > mean_tech_south and mean_tfp_north <= mean_tfp_south:
        issues.append("- J-Curve logic inverted: N has more tech but same/lower TFP")
    
    if issues:
        print("Issues found:")
        for i in issues:
            print(i)
    else:
        print("No major issues found.")

    # === CHECK 6: SECTORAL ANALYSIS ===
    print("\n[CHECK 6: SECTORAL ANALYSIS (Means)]")
    sectors = df.groupby('Sector')[['TechIntensity', 'TFP']].mean()
    print(sectors)
    
    # === CHECK 7: ROBUSTNESS (LAGGED MODEL) ===
    print("\n[CHECK 7: LAGGED MODEL]")
    df = df.sort_values(['firm_id', 'Year'])
    df['Tech_Lag'] = df.groupby('firm_id')['TechIntensity'].shift(1)
    df['Tech_Lag_Sq'] = df['Tech_Lag'] ** 2
    
    df_lag = df.dropna(subset=['Tech_Lag', 'TFP']).copy()
    
    X_lag = sm.add_constant(df_lag[['Tech_Lag', 'Tech_Lag_Sq']])
    # Add time dummies for lag model? Thesis doesn't specify but standard is yes. 
    # For quick check, skip or add? Code usually adds them. 
    # Let's keep it simple to compare with previous estimates.
    
    model_lag = sm.OLS(df_lag['TFP'], X_lag).fit(cov_type='cluster', cov_kwds={'groups': df_lag['firm_id']})
    
    g1_lag = model_lag.params['Tech_Lag']
    g2_lag = model_lag.params['Tech_Lag_Sq']
    tp_lag = -g1_lag / (2 * g2_lag)
    
    print(f"Gamma1_Lag: {g1_lag:.4f}")
    print(f"Gamma2_Lag: {g2_lag:.4f}")
    print(f"Turning Point Lag: {tp_lag:.4%}")

if __name__ == "__main__":
    extended_verification()
