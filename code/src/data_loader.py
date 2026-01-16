import pandas as pd
import numpy as np
from .config import DATA_FILE, PROVINCE_TO_MACRO, COL_ATECO, SECTOR_MAP

def load_and_clean_data():
    """
    Loads AIDA data and applies filters for high-tech manufacturing SMEs.
    
    Inclusion criteria:
    - Revenue < €50M (in thousands in the file)
    - Employees: 10-250
    - ATECO Sectors: 26, 27, 28 (high-tech manufacturing)
    - Balanced panel: only firms with 10 complete years of data
    
    Returns:
        pd.DataFrame: Balanced panel of firm-year observations
    """
    print("=" * 60)
    print("STEP 1: DATA LOADING AND CLEANING")
    print("=" * 60)
    
    print(f"Reading file: {DATA_FILE}")
    df_raw = pd.read_excel(DATA_FILE, header=None)
    df_data = df_raw.iloc[3:].copy().reset_index(drop=True)
    
    records = []
    firm_counter = {}
    
    # AIDA column indices
    COL_NAME, COL_PROVINCE, COL_DATE = 1, 2, 3
    COL_REV_LAST_MIGL, COL_EMP_LAST = 4, 5
    
    print("Applying SME filters...")
    for idx, row in df_data.iterrows():
        # SME Filters
        try:
            rev_migl = pd.to_numeric(row[COL_REV_LAST_MIGL], errors='coerce')
            emp = pd.to_numeric(row[COL_EMP_LAST], errors='coerce')
        except:
            continue
        
        if pd.isna(rev_migl) or rev_migl >= 50000:  # €50M in thousands
            continue
        if pd.isna(emp) or not (10 <= emp <= 250):
            continue
        
        company_name = str(row[COL_NAME]).strip()
        province = str(row[COL_PROVINCE]).strip().upper()[:2]
        macro = PROVINCE_TO_MACRO.get(province, None)
        if macro is None:
            continue
            
        # Filter and Assign ATECO Sector
        try:
            ateco_raw = str(row[COL_ATECO]).strip()
            # Handle numeric codes or strings like '26.11' or '261100'
            if len(ateco_raw) >= 2:
                ateco_2digit = ateco_raw[:2]
            else:
                ateco_2digit = '00'
        except:
            ateco_2digit = '00'
            
        sector_name = SECTOR_MAP.get(ateco_2digit, None)
        if sector_name is None:
            continue
        
        # Unique Firm ID
        firm_key = f"{company_name}_{province}"
        if firm_key not in firm_counter:
            firm_counter[firm_key] = len(firm_counter) + 1
        firm_id = firm_counter[firm_key]
        
        try:
            ref_year = pd.to_datetime(row[COL_DATE]).year
        except:
            ref_year = 2023
        
        # Extract data for all available years (up to 10 years back)
        for y_idx in range(10):
            year = ref_year - y_idx
            
            def get_val(start_col):
                try:
                    return pd.to_numeric(row[start_col + y_idx], errors='coerce')
                except:
                    return np.nan
            
            va = get_val(38)   # Value Added
            lc = get_val(18)   # Labor Costs
            ma = get_val(58)   # Tangible Assets (Material)
            ia = get_val(68)   # Intangible Assets (Immaterial)
            ta = get_val(48)   # Total Assets
            
            if all([pd.notna(v) and v > 0 for v in [va, lc, ma, ta]]) and pd.notna(ia) and ia >= 0:
                records.append({
                    'firm_id': firm_id,
                    'Year': year,
                    'ValueAdded': va,
                    'LaborCost': lc,
                    'MatAssets': ma,
                    'ImmatAssets': ia,
                    'TotalAssets': ta,
                    'MacroArea': macro,
                    'Sector': sector_name,
                })
    
    df = pd.DataFrame(records)
    
    print(f"✓ Loaded {len(df):,} firm-year observations (pre-balancing)")
    print(f"✓ {df['firm_id'].nunique():,} unique firms (pre-balancing)")
    
    # =========================================================================
    # BALANCED PANEL FILTER: only firms with exactly 10 years of data
    # =========================================================================
    print("\nApplying balanced panel filter (10 complete years)...")
    years_per_firm = df.groupby('firm_id')['Year'].nunique()
    firms_with_10_years = years_per_firm[years_per_firm == 10].index
    df = df[df['firm_id'].isin(firms_with_10_years)].copy()
    
    print(f"✓ Balanced Panel: {len(df):,} firm-year observations")
    print(f"✓ {df['firm_id'].nunique():,} firms with 10 complete years")
    print(f"✓ Period: {df['Year'].min()}-{df['Year'].max()}")
    
    # =========================================================================
    # YEAR FILTER: Restrict to 2015-2024 as stated in the thesis
    # =========================================================================
    print("\nApplying year filter (2015-2024)...")
    df = df[(df['Year'] >= 2015) & (df['Year'] <= 2024)].copy()
    
    # Re-check balanced panel after year filter
    years_per_firm = df.groupby('firm_id')['Year'].nunique()
    firms_with_10_years = years_per_firm[years_per_firm == 10].index
    df = df[df['firm_id'].isin(firms_with_10_years)].copy()
    
    print(f"✓ Final Panel: {len(df):,} firm-year observations")
    print(f"✓ {df['firm_id'].nunique():,} firms with complete 2015-2024 data")
    
    return df

def prepare_variables(df):
    """
    Creates variables for econometric analysis (Log-transformations and ratios)
    and handles outliers via winsorization.
    """
    print("\n" + "=" * 60)
    print("STEP 2: VARIABLE PREPARATION")
    print("=" * 60)
    
    # 1. Log-Variables (Input for Production Function)
    df['ln_Y'] = np.log(df['ValueAdded'])
    df['ln_L'] = np.log(df['LaborCost'])
    df['ln_K'] = np.log(df['MatAssets'])
    
    # 2. Variables of Interest (J-Curve)
    # Tech Intensity = Intangible Assets / Total Assets
    df['TechIntensity'] = df['ImmatAssets'] / df['TotalAssets']
    
    # Labor Productivity = Value Added / Labor Cost
    df['LaborProd'] = df['ValueAdded'] / df['LaborCost']
    
    # 3. Winsorization (1% - 99%) to remove extreme outliers
    print("Winsorization (1%-99%)...")
    cols_to_winsorize = ['ln_Y', 'ln_L', 'ln_K', 'TechIntensity', 'LaborProd']
    
    for col in cols_to_winsorize:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)
    
    # Control statistics
    print(f"✓ Mean Tech Intensity: {df['TechIntensity'].mean():.3f}")
    print(f"✓ Tech Intensity SD: {df['TechIntensity'].std():.3f}")
    
    return df
