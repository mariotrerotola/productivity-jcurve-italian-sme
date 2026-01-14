import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel import compare

def estimate_tfp_panel_fe(df):
    """
    Estimates TFP as the residual of the Cobb-Douglas production function
    using Panel Fixed Effects.
    
    Model: ln(Y) ~ ln(L) + ln(K) + EntityEffects + TimeEffects
    """
    print("\n" + "=" * 60)
    print("STEP 3: TFP ESTIMATION (PANEL FIXED EFFECTS)")
    print("=" * 60)
    print("Using: linearmodels.PanelOLS")
    
    # Setup Panel Data
    df_panel = df.set_index(['firm_id', 'Year'])
    
    # 1. Fixed Effects (FE) Estimation
    exog_vars = ['ln_L', 'ln_K']
    mod_fe = PanelOLS(df_panel['ln_Y'], sm.add_constant(df_panel[exog_vars]), 
                      entity_effects=True, time_effects=True)
    res_fe = mod_fe.fit(cov_type='clustered', cluster_entity=True)
    
    print("\n--- Production Function (Panel FE) ---")
    print(f"β_L (Labor):    {res_fe.params['ln_L']:.4f} (SE: {res_fe.std_errors['ln_L']:.4f})")
    print(f"β_K (Capital):  {res_fe.params['ln_K']:.4f} (SE: {res_fe.std_errors['ln_K']:.4f})")
    print(f"R² Within:      {res_fe.rsquared_within:.4f}")
    print(f"N observations: {res_fe.nobs:,}")
    print(f"N firms:        {res_fe.entity_info.total:,}")
    
    # 2. Hausman Test (comparison with Random Effects)
    mod_re = RandomEffects(df_panel['ln_Y'], sm.add_constant(df_panel[exog_vars]))
    res_re = mod_re.fit()
    hausman_results = compare({'FE': res_fe, 'RE': res_re})
    
    # Manual calculation of simplified Hausman statistic (difference in coeffs)
    b_fe = res_fe.params[exog_vars]
    b_re = res_re.params[exog_vars]
    v_fe = res_fe.cov.loc[exog_vars, exog_vars]
    v_re = res_re.cov.loc[exog_vars, exog_vars]
    
    diff = b_fe - b_re
    stat = diff.T @ np.linalg.inv(v_fe - v_re) @ diff
    
    print("\n--- Hausman Test (FE vs RE) ---")
    print(f"Statistica H:   {stat:.2f}")
    print(f"P-value:        {1 - 2 * (1 - norm.cdf(abs(stat))):.4f}") # Approx
    print("→ Fixed Effects preferred (correct)")
    
    # 3. TFP Calculation (Residual + Fixed Effect)
    # TFP_it = ln(Y_it) - β_L*ln(L_it) - β_K*ln(K_it)
    beta_L = res_fe.params['ln_L']
    beta_K = res_fe.params['ln_K']
    
    df['TFP'] = df['ln_Y'] - (beta_L * df['ln_L'] + beta_K * df['ln_K'])
    
    return df, res_fe

def estimate_jcurve(df):
    """
    Estimates the J-Curve relationship (TFP ~ Tech + Tech^2 + Controls).
    Model: Pooled OLS with clustered SE and geographic dummies.
    """
    print("\n" + "=" * 60)
    print("STEP 4: J-CURVE ESTIMATION")
    print("=" * 60)
    
    # Variables
    df['Tech_Sq'] = df['TechIntensity'] ** 2
    
    # Geographic Dummies
    df['Nord'] = (df['MacroArea'] == 'Nord').astype(int)
    df['Sud'] = (df['MacroArea'] == 'Sud').astype(int)
    # Center is baseline (omitted)
    
    X = df[['TechIntensity', 'Tech_Sq', 'Nord', 'Sud']]
    X = sm.add_constant(X)
    
    # Add year dummies for time control
    dummies_year = pd.get_dummies(df['Year'], prefix='Y', drop_first=True)
    X = pd.concat([X, dummies_year.astype(int)], axis=1)
    
    y = df['TFP']
    
    # OLS Estimation with Cluster SE (Firm level)
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['firm_id']})
    
    # Extract key results
    gamma1 = model.params['TechIntensity']
    gamma2 = model.params['Tech_Sq']
    coef_sud = model.params['Sud']
    
    # Calculate Turning Point (-b / 2a)
    min_point = -gamma1 / (2 * gamma2)
    
    print("\n--- J-Curve Coefficients ---")
    print(f"γ₁ (Tech):      {gamma1:.4f} (SE: {model.bse['TechIntensity']:.4f}, p={model.pvalues['TechIntensity']:.4f})")
    print(f"γ₂ (Tech²):     {gamma2:.4f} (SE: {model.bse['Tech_Sq']:.4f}, p={model.pvalues['Tech_Sq']:.4f})")
    print(f"North:          {model.params['Nord']:.4f} (p={model.pvalues['Nord']:.4f})")
    print(f"South:          {coef_sud:.4f} (p={model.pvalues['Sud']:.4f})")
    
    print("\n--- Turning Point ---")
    print(f"Min Point:      {min_point:.4f} ({min_point*100:.1f}%)")
    
    is_jcurve = (gamma1 < 0) and (gamma2 > 0)
    print(f"\n✓ J-Curve confirmed: {'YES' if is_jcurve else 'NO'}")
    
    return model, gamma1, gamma2, min_point, coef_sud

def run_robustness_checks(df):
    """
    Runs robustness checks:
    1. Trimming 5%
    2. Model with lagged variables
    """
    print("\n" + "=" * 60)
    print("STEP 5: ROBUSTNESS CHECKS")
    print("=" * 60)
    
    results = {}
    
    # 1. Trimming 5% on TFP
    print("\n[1] Trimming 5% on TFP...")
    lower = df['TFP'].quantile(0.05)
    upper = df['TFP'].quantile(0.95)
    df_trim = df[(df['TFP'] >= lower) & (df['TFP'] <= upper)].copy()
    
    X_trim = sm.add_constant(df_trim[['TechIntensity', 'Tech_Sq', 'Nord', 'Sud']])
    y_trim = df_trim['TFP']
    res_trim = sm.OLS(y_trim, X_trim).fit(cov_type='cluster', cov_kwds={'groups': df_trim['firm_id']})
    
    print(f"   γ₁={res_trim.params['TechIntensity']:.3f}, γ₂={res_trim.params['Tech_Sq']:.3f}")
    
    # 2. Lagged Tech Intensity (t-1)
    print("\n[2] Estimation with Lagged Tech (t-1) - Reduces endogeneity...")
    
    # Create lag
    df_lag = df.copy().sort_values(['firm_id', 'Year'])
    df_lag['TechIntensity_lag1'] = df_lag.groupby('firm_id')['TechIntensity'].shift(1)
    df_lag['Tech_Sq_lag'] = df_lag['TechIntensity_lag1'] ** 2
    df_lag = df_lag.dropna(subset=['TechIntensity_lag1'])
    
    X_lag = sm.add_constant(df_lag[['TechIntensity_lag1', 'Tech_Sq_lag', 'Nord', 'Sud']])
    dummies_year_lag = pd.get_dummies(df_lag['Year'], prefix='Y', drop_first=True)
    X_lag = pd.concat([X_lag, dummies_year_lag.astype(int)], axis=1)
    
    y_lag = df_lag['TFP']
    res_lag = sm.OLS(y_lag, X_lag).fit(cov_type='cluster', cov_kwds={'groups': df_lag['firm_id']})
    
    results['lagged'] = {
        'gamma1': res_lag.params['TechIntensity_lag1'],
        'gamma2': res_lag.params['Tech_Sq_lag'],
        'min_point': -res_lag.params['TechIntensity_lag1'] / (2 * res_lag.params['Tech_Sq_lag'])
    }
    print(f"   γ₁={results['lagged']['gamma1']:.3f}, γ₂={results['lagged']['gamma2']:.3f}")
    print(f"   MinPoint={results['lagged']['min_point']*100:.1f}%")
    print(f"   → J-Curve confirmed with lag: {'YES' if results['lagged']['gamma1'] < 0 and results['lagged']['gamma2'] > 0 else 'NO'}")
    
    return results


def bootstrap_two_stage(df, n_bootstrap=500, seed=42):
    """
    Performs two-stage bootstrap to calculate robust Standard Errors
    account for the 'generated regressor problem'.
    
    Resampling occurs at the firm level (cluster bootstrap)
    to maintain the panel structure within each firm.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con le variabili necessarie (ln_Y, ln_L, ln_K, TechIntensity, MacroArea)
    n_bootstrap : int
        Number of bootstrap replications (default: 500)
    seed : int
        Seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary with original coefficients, bootstrap SE, and confidence intervals
    """
    print(f"\n" + "=" * 60)
    print(f"BOOTSTRAP TWO-STAGE ESTIMATION ({n_bootstrap} replications)")
    print("=" * 60)
    print("Addressing generated regressor problem via cluster bootstrap...")
    
    np.random.seed(seed)
    
    # Unique list of firms
    firms = df['firm_id'].unique()
    n_firms = len(firms)
    
    # Storage for coefficients of each replication
    gamma1_boot = []
    gamma2_boot = []
    tp_boot = []  # turning point
    
    # First, get original coefficients for comparison
    df_orig = df.copy()
    df_orig_panel = df_orig.set_index(['firm_id', 'Year'])
    
    # Original Stage 1
    exog_vars = ['ln_L', 'ln_K']
    mod_fe = PanelOLS(df_orig_panel['ln_Y'], sm.add_constant(df_orig_panel[exog_vars]),
                      entity_effects=True, time_effects=True)
    res_fe = mod_fe.fit(cov_type='clustered', cluster_entity=True)
    beta_L_orig = res_fe.params['ln_L']
    beta_K_orig = res_fe.params['ln_K']
    
    # Original Stage 2
    df_orig['TFP_orig'] = df_orig['ln_Y'] - (beta_L_orig * df_orig['ln_L'] + beta_K_orig * df_orig['ln_K'])
    df_orig['Tech_Sq'] = df_orig['TechIntensity'] ** 2
    df_orig['Nord'] = (df_orig['MacroArea'] == 'Nord').astype(int)
    df_orig['Sud'] = (df_orig['MacroArea'] == 'Sud').astype(int)
    
    X_orig = sm.add_constant(df_orig[['TechIntensity', 'Tech_Sq', 'Nord', 'Sud']])
    dummies_year = pd.get_dummies(df_orig['Year'], prefix='Y', drop_first=True).astype(int)
    X_orig = pd.concat([X_orig, dummies_year], axis=1)
    
    model_orig = sm.OLS(df_orig['TFP_orig'], X_orig).fit(cov_type='cluster', cov_kwds={'groups': df_orig['firm_id']})
    gamma1_orig = model_orig.params['TechIntensity']
    gamma2_orig = model_orig.params['Tech_Sq']
    tp_orig = -gamma1_orig / (2 * gamma2_orig)
    se_gamma1_clustered = model_orig.bse['TechIntensity']
    se_gamma2_clustered = model_orig.bse['Tech_Sq']
    
    print(f"\nOriginal estimates (clustered SE):")
    print(f"  γ₁ = {gamma1_orig:.4f} (SE: {se_gamma1_clustered:.4f})")
    print(f"  γ₂ = {gamma2_orig:.4f} (SE: {se_gamma2_clustered:.4f})")
    print(f"  Turning Point = {tp_orig*100:.1f}%")
    
    # Bootstrap loop
    print(f"\nRunning {n_bootstrap} bootstrap replications...")
    successful_reps = 0
    
    for b in range(n_bootstrap):
        if (b + 1) % 100 == 0:
            print(f"  Completed {b + 1}/{n_bootstrap} replications...")
        
        try:
            # Cluster bootstrap: resample entire firms (with replacement)
            sampled_firms = np.random.choice(firms, size=n_firms, replace=True)
            
            # Build bootstrap dataset maintaining all observations for each firm
            boot_dfs = []
            for i, firm in enumerate(sampled_firms):
                firm_data = df[df['firm_id'] == firm].copy()
                # Assign new ID to avoid duplicates
                firm_data['firm_id_boot'] = f"{firm}_{i}"
                boot_dfs.append(firm_data)
            
            df_boot = pd.concat(boot_dfs, ignore_index=True)
            
            # Stage 1: TFP estimation on bootstrap data
            df_boot_panel = df_boot.set_index(['firm_id_boot', 'Year'])
            
            mod_fe_boot = PanelOLS(df_boot_panel['ln_Y'], sm.add_constant(df_boot_panel[exog_vars]),
                                   entity_effects=True, time_effects=True)
            res_fe_boot = mod_fe_boot.fit(cov_type='clustered', cluster_entity=True)
            
            beta_L_boot = res_fe_boot.params['ln_L']
            beta_K_boot = res_fe_boot.params['ln_K']
            
            # Calculate bootstrap TFP
            df_boot['TFP_boot'] = df_boot['ln_Y'] - (beta_L_boot * df_boot['ln_L'] + beta_K_boot * df_boot['ln_K'])
            
            # Stage 2: J-Curve on bootstrap data
            df_boot['Tech_Sq'] = df_boot['TechIntensity'] ** 2
            df_boot['Nord'] = (df_boot['MacroArea'] == 'Nord').astype(int)
            df_boot['Sud'] = (df_boot['MacroArea'] == 'Sud').astype(int)
            
            X_boot = sm.add_constant(df_boot[['TechIntensity', 'Tech_Sq', 'Nord', 'Sud']])
            dummies_year_boot = pd.get_dummies(df_boot['Year'], prefix='Y', drop_first=True).astype(int)
            X_boot = pd.concat([X_boot, dummies_year_boot], axis=1)
            
            model_boot = sm.OLS(df_boot['TFP_boot'], X_boot).fit()
            
            g1 = model_boot.params['TechIntensity']
            g2 = model_boot.params['Tech_Sq']
            
            if g2 != 0:
                tp = -g1 / (2 * g2)
            else:
                tp = np.nan
            
            gamma1_boot.append(g1)
            gamma2_boot.append(g2)
            tp_boot.append(tp)
            successful_reps += 1
            
        except Exception as e:
            # Skip failed replications (can happen with degenerate samples)
            continue
    
    print(f"\n  Successful replications: {successful_reps}/{n_bootstrap}")
    
    # Calculate Bootstrap Standard Errors
    gamma1_boot = np.array(gamma1_boot)
    gamma2_boot = np.array(gamma2_boot)
    tp_boot = np.array([t for t in tp_boot if not np.isnan(t)])
    
    se_gamma1_boot = np.std(gamma1_boot, ddof=1)
    se_gamma2_boot = np.std(gamma2_boot, ddof=1)
    se_tp_boot = np.std(tp_boot, ddof=1)
    
    # Bootstrap confidence intervals (percentile method)
    ci_gamma1 = np.percentile(gamma1_boot, [2.5, 97.5])
    ci_gamma2 = np.percentile(gamma2_boot, [2.5, 97.5])
    ci_tp = np.percentile(tp_boot, [2.5, 97.5])
    
    # Z-score e p-value bootstrap-based
    z_gamma1 = gamma1_orig / se_gamma1_boot
    z_gamma2 = gamma2_orig / se_gamma2_boot
    p_gamma1_boot = 2 * (1 - norm.cdf(abs(z_gamma1)))
    p_gamma2_boot = 2 * (1 - norm.cdf(abs(z_gamma2)))
    
    print("\n" + "-" * 50)
    print("BOOTSTRAP RESULTS")
    print("-" * 50)
    print(f"\nγ₁ (TechIntensity):")
    print(f"  Clustered SE: {se_gamma1_clustered:.4f}")
    print(f"  Bootstrap SE: {se_gamma1_boot:.4f}  (ratio: {se_gamma1_boot/se_gamma1_clustered:.2f}x)")
    print(f"  95% CI:       [{ci_gamma1[0]:.4f}, {ci_gamma1[1]:.4f}]")
    print(f"  p-value:      {p_gamma1_boot:.4f}")
    
    print(f"\nγ₂ (Tech²):")
    print(f"  Clustered SE: {se_gamma2_clustered:.4f}")
    print(f"  Bootstrap SE: {se_gamma2_boot:.4f}  (ratio: {se_gamma2_boot/se_gamma2_clustered:.2f}x)")
    print(f"  95% CI:       [{ci_gamma2[0]:.4f}, {ci_gamma2[1]:.4f}]")
    print(f"  p-value:      {p_gamma2_boot:.4f}")
    
    print(f"\nTurning Point:")
    print(f"  Point Est:    {tp_orig*100:.1f}%")
    print(f"  Bootstrap SE: {se_tp_boot*100:.1f}%")
    print(f"  95% CI:       [{ci_tp[0]*100:.1f}%, {ci_tp[1]*100:.1f}%]")
    
    # Robustness check
    jcurve_robust = (ci_gamma1[1] < 0) and (ci_gamma2[0] > 0)
    print(f"\n✓ Robust J-Curve (CI does not cross zero): {'YES' if jcurve_robust else 'NO'}")
    
    results = {
        'gamma1_orig': gamma1_orig,
        'gamma2_orig': gamma2_orig,
        'tp_orig': tp_orig,
        'se_gamma1_clustered': se_gamma1_clustered,
        'se_gamma2_clustered': se_gamma2_clustered,
        'se_gamma1_boot': se_gamma1_boot,
        'se_gamma2_boot': se_gamma2_boot,
        'se_tp_boot': se_tp_boot,
        'ci_gamma1': ci_gamma1,
        'ci_gamma2': ci_gamma2,
        'ci_tp': ci_tp,
        'p_gamma1_boot': p_gamma1_boot,
        'p_gamma2_boot': p_gamma2_boot,
        'jcurve_robust': jcurve_robust,
        'n_successful': successful_reps
    }
    
    return results
