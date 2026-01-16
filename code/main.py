import sys
from pathlib import Path

# Add the current directory to sys.path to ensure local imports work
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import RESULTS_DIR
from src.data_loader import load_and_clean_data, prepare_variables
from src.econometrics import estimate_tfp_panel_fe, estimate_jcurve, run_robustness_checks, bootstrap_two_stage
from src.visualization import plot_jcurve, plot_geo_boxplots, plot_sector_analysis

def save_text_results(df, model, gamma1, gamma2, min_point, coef_is_south, gamma1_south_diff, gamma2_south_diff, min_point_south, robustness=None):
    """Saves main results to text file."""
    print("\n" + "=" * 60)
    print("\n" + "=" * 60)
    print("STEP 7: SAVING RESULTS")
    print("=" * 60)
    
    # 1. Full Regression Output
    with open(RESULTS_DIR / 'regression_output.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("J-CURVE ESTIMATION - PANEL FIXED EFFECTS WITH INTERACTIONS\n")
        f.write("=" * 60 + "\n\n")
        f.write(model.summary().as_text())
    print(f"✓ Saved: {RESULTS_DIR / 'regression_output.txt'}")
    
    # 2. Key Values for the Paper
    with open(RESULTS_DIR / 'paper_values.txt', 'w') as f:
        f.write("VALUES FOR THE PAPER\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"N. Observations:     {len(df):,}\n")
        f.write(f"N. Firms:            {df['firm_id'].nunique():,}\n")
        f.write(f"Tech Intensity Mean: {df['TechIntensity'].mean():.3f}\n")
        f.write(f"Tech Intensity SD:   {df['TechIntensity'].std():.3f}\n\n")
        
        f.write("--- MAIN MODEL (Center-North Baseline) ---\n")
        f.write(f"γ₁ (Tech):           {gamma1:.3f}\n")
        f.write(f"γ₂ (Tech²):          {gamma2:.3f}\n")
        f.write(f"Turning Point (CN):  {min_point*100:.1f}%\n\n")
        
        f.write("--- REGIONAL INTERACTIONS (South Deviation) ---\n")
        f.write(f"IsSouth Dummy:       {coef_is_south:.3f}\n")
        f.write(f"Tech * South:        {gamma1_south_diff:.3f}\n")
        f.write(f"Tech² * South:       {gamma2_south_diff:.3f}\n")
        f.write(f"Turning Point (South): {min_point_south*100:.1f}%\n")
        
        if robustness and 'lagged' in robustness:
            f.write("\n--- ROBUSTNESS: LAGGED VARIABLES ---\n")
            f.write(f"γ₁ (Tech_lag1):      {robustness['lagged']['gamma1']:.3f}\n")
            f.write(f"γ₂ (Tech²_lag1):     {robustness['lagged']['gamma2']:.3f}\n")
            f.write(f"Turning Point (lag): {robustness['lagged']['min_point']*100:.1f}%\n")
        
        if robustness and 'bootstrap' in robustness:
            boot = robustness['bootstrap']
            f.write("\n--- BOOTSTRAP STANDARD ERRORS (Generated Regressor Correction) ---\n")
            f.write(f"N. Replications:      {boot['n_successful']}\n\n")
            f.write(f"γ₁ (TechIntensity):\n")
            f.write(f"  Clustered SE:      {boot['se_gamma1_clustered']:.4f}\n")
            f.write(f"  Bootstrap SE:      {boot['se_gamma1_boot']:.4f}  (ratio: {boot['se_gamma1_boot']/boot['se_gamma1_clustered']:.2f}x)\n")
            f.write(f"  95% CI:            [{boot['ci_gamma1'][0]:.4f}, {boot['ci_gamma1'][1]:.4f}]\n")
            f.write(f"  p-value (boot):    {boot['p_gamma1_boot']:.4f}\n\n")
            f.write(f"γ₂ (Tech²):\n")
            f.write(f"  Clustered SE:      {boot['se_gamma2_clustered']:.4f}\n")
            f.write(f"  Bootstrap SE:      {boot['se_gamma2_boot']:.4f}  (ratio: {boot['se_gamma2_boot']/boot['se_gamma2_clustered']:.2f}x)\n")
            f.write(f"  95% CI:            [{boot['ci_gamma2'][0]:.4f}, {boot['ci_gamma2'][1]:.4f}]\n")
            f.write(f"  p-value (boot):    {boot['p_gamma2_boot']:.4f}\n\n")
            f.write(f"Turning Point (Baseline):\n")
            f.write(f"  Point Estimate:    {boot['tp_orig']*100:.1f}%\n")
            f.write(f"  Bootstrap SE:      {boot['se_tp_boot']*100:.1f}%\n")
            f.write(f"  95% CI:            [{boot['ci_tp'][0]*100:.1f}%, {boot['ci_tp'][1]*100:.1f}%]\n\n")
            f.write(f"J-Curve Robust (CI excludes zero): {'YES' if boot['jcurve_robust'] else 'NO'}\n")
            
    print(f"✓ Saved: {RESULTS_DIR / 'paper_values.txt'}")

def main():
    print("\n" + "=" * 70)
    print("  ECONOMETRIC ANALYSIS J-CURVE - ITALIAN SMES")
    print("  Università di Salerno")
    print("  Version: Modular Architecture")
    print("=" * 70)
    
    # --- EXECUTION PIPELINE ---
    
    # 1. Data Ingestion & Cleaning
    df = load_and_clean_data()
    
    # 2. Feature Engineering
    df = prepare_variables(df)
    
    # 3. Econometric Estimation (Stage 1: TFP)
    df, tfp_model = estimate_tfp_panel_fe(df)
    
    # 4. Econometric Estimation (Stage 2: J-Curve)
    jcurve_model, gamma1, gamma2, min_point, coef_is_south, gamma1_south_diff, gamma2_south_diff, min_point_south = estimate_jcurve(df)
    
    # 5. Robustness Checks
    robustness = run_robustness_checks(df)
    
    # 5b. Bootstrap Standard Errors (addresses generated regressor problem)
    bootstrap_results = bootstrap_two_stage(df, n_bootstrap=500)
    robustness['bootstrap'] = bootstrap_results
    
    # 6. Visualization
    print("\n" + "=" * 60)
    print("STEP 6: GENERATING PLOTS")
    print("=" * 60)
    plot_jcurve(jcurve_model, df)
    plot_geo_boxplots(df)
    plot_sector_analysis(df)
    
    # 7. Reporting
    save_text_results(df, jcurve_model, gamma1, gamma2, min_point, coef_is_south, gamma1_south_diff, gamma2_south_diff, min_point_south, robustness)
    
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  N. Observations:    {len(df):,}")
    print(f"  N. Firms:           {df['firm_id'].nunique():,}")
    print(f"  γ₁ (Tech):          {gamma1:.4f}")
    print(f"  γ₂ (Tech²):         {gamma2:.4f}")
    print(f"  Turning Point (CN): {min_point*100:.1f}%")
    print(f"  Turning Point (S):  {min_point_south*100:.1f}%")
    print(f"  IsSouth:            {coef_is_south:.4f}")
    print(f"  ✓ J-Curve:          {'CONFIRMED' if gamma1 < 0 and gamma2 > 0 else 'NOT CONFIRMED'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
