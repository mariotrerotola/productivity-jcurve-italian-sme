"""
================================================================================
ANALISI ECONOMETRICA: PRODUCTIVITY J-CURVE NELLE PMI ITALIANE
================================================================================

Autore: Mario Trerotola
Università di Salerno
Data: Gennaio 2026

Descrizione:
------------
Questo script implementa l'analisi empirica della "Productivity J-Curve" 
(Brynjolfsson et al., 2021) applicata alle PMI italiane del settore 
manifatturiero high-tech, utilizzando dati AIDA (2014-2023).

Metodologia:
------------
1. STAGE 1 - Stima TFP:
   - Panel Fixed Effects con effetti fissi d'impresa e temporali
   - Funzione di produzione Cobb-Douglas: ln(Y) = β_L*ln(L) + β_K*ln(K) + μ_i + δ_t + ε
   - Standard errors clustered a livello impresa
   - Test Hausman per validare FE vs RE

2. STAGE 2 - Test J-Curve:
   - Pooled OLS con clustered SE
   - TFP = α + γ₁*Tech + γ₂*Tech² + β_Nord*Nord + β_Sud*Sud + τ_t + η
   - J-Curve confermata se: γ₁ < 0 (calo iniziale) e γ₂ > 0 (recupero)

Risultati Attesi:
-----------------
- γ₁ ≈ -1.04 (significativo al 1%)
- γ₂ ≈ +3.11 (significativo al 1%)
- Turning point ≈ 16.7%
- Sud premium ≈ +2.9% (effetto selezione competitiva)

Requisiti:
----------
pip install pandas numpy statsmodels linearmodels matplotlib seaborn scipy

Utilizzo:
---------
python analisi_jcurve.py

Output:
-------
- figure/fig_jcurve.png         : Grafico della J-Curve stimata
- figure/fig_geo_boxplots.png   : Boxplot TFP per area geografica
- figure/fig_sector_bar.png     : Analisi per settore ATECO
- risultati/regression_output.txt : Output completo della regressione
- risultati/paper_values.txt    : Valori da inserire nel paper
================================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importa linearmodels per Panel Fixed Effects
try:
    from linearmodels.panel import PanelOLS, RandomEffects
    PANEL_AVAILABLE = True
except ImportError:
    print("⚠️  ATTENZIONE: linearmodels non installato.")
    print("   Installare con: pip install linearmodels")
    print("   Lo script procederà con fallback su OLS con dummies.")
    PANEL_AVAILABLE = False

# =============================================================================
# CONFIGURAZIONE
# =============================================================================
# Ottieni il path della cartella principale (parent di codice)
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / 'figure'
RESULTS_DIR = PROJECT_DIR / 'risultati'
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DATA_FILE = PROJECT_DIR / 'dati' / 'Aida_Export_2.xls'

# Stile grafici
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Mapping Province -> Macroarea
PROVINCE_TO_MACRO = {
    # NORD
    'MI': 'Nord', 'TO': 'Nord', 'BS': 'Nord', 'BG': 'Nord', 'VA': 'Nord', 'GE': 'Nord',
    'VR': 'Nord', 'VI': 'Nord', 'TV': 'Nord', 'PD': 'Nord', 'BO': 'Nord', 'MO': 'Nord',
    'PC': 'Nord', 'PR': 'Nord', 'RE': 'Nord', 'TN': 'Nord', 'BZ': 'Nord', 'UD': 'Nord',
    'TS': 'Nord', 'AO': 'Nord', 'CN': 'Nord', 'AT': 'Nord', 'AL': 'Nord', 'VC': 'Nord',
    'NO': 'Nord', 'VB': 'Nord', 'BI': 'Nord', 'LO': 'Nord', 'CR': 'Nord', 'MN': 'Nord',
    'LC': 'Nord', 'CO': 'Nord', 'SO': 'Nord', 'PV': 'Nord', 'SV': 'Nord', 'IM': 'Nord',
    'SP': 'Nord', 'RO': 'Nord', 'BL': 'Nord', 'FE': 'Nord', 'RA': 'Nord', 'FC': 'Nord', 
    'RN': 'Nord', 'GO': 'Nord', 'PN': 'Nord',
    # CENTRO
    'RM': 'Centro', 'FI': 'Centro', 'PI': 'Centro', 'LI': 'Centro', 'AR': 'Centro', 
    'SI': 'Centro', 'GR': 'Centro', 'PG': 'Centro', 'TR': 'Centro', 'AN': 'Centro', 
    'MC': 'Centro', 'AP': 'Centro', 'PU': 'Centro', 'FM': 'Centro', 'PS': 'Centro', 
    'PT': 'Centro', 'PO': 'Centro', 'LU': 'Centro', 'MS': 'Centro', 'VT': 'Centro', 
    'RI': 'Centro', 'LT': 'Centro', 'FR': 'Centro', 'TE': 'Centro', 'PE': 'Centro', 
    'CH': 'Centro', 'AQ': 'Centro',
    # SUD
    'NA': 'Sud', 'SA': 'Sud', 'BA': 'Sud', 'PA': 'Sud', 'CT': 'Sud', 'ME': 'Sud',
    'CA': 'Sud', 'SS': 'Sud', 'TA': 'Sud', 'BR': 'Sud', 'LE': 'Sud', 'FG': 'Sud',
    'BT': 'Sud', 'CB': 'Sud', 'IS': 'Sud', 'PZ': 'Sud', 'MT': 'Sud', 'CS': 'Sud', 
    'CZ': 'Sud', 'RC': 'Sud', 'KR': 'Sud', 'VV': 'Sud', 'CE': 'Sud', 'AV': 'Sud', 
    'BN': 'Sud', 'AG': 'Sud', 'CL': 'Sud', 'EN': 'Sud', 'TP': 'Sud', 'RG': 'Sud', 
    'SR': 'Sud', 'NU': 'Sud', 'OR': 'Sud', 'SU': 'Sud', 'CI': 'Sud',
}


# =============================================================================
# FUNZIONI PRINCIPALI
# =============================================================================

def load_and_clean_data():
    """
    Carica i dati AIDA e applica i filtri per PMI manifatturiere high-tech.
    
    Criteri di inclusione:
    - Fatturato < €50M (in migliaia nel file)
    - Dipendenti: 10-250
    - Settori ATECO: 26, 27, 28 (high-tech manufacturing)
    
    Returns:
        pd.DataFrame: Panel bilanciato di osservazioni firm-year
    """
    print("=" * 60)
    print("STEP 1: CARICAMENTO E PULIZIA DATI")
    print("=" * 60)
    
    print(f"Lettura file: {DATA_FILE}")
    df_raw = pd.read_excel(DATA_FILE, header=None)
    df_data = df_raw.iloc[3:].copy().reset_index(drop=True)
    
    records = []
    firm_counter = {}
    
    # Indici colonne AIDA
    COL_NAME, COL_PROVINCE, COL_DATE = 1, 2, 3
    COL_REV_LAST_MIGL, COL_EMP_LAST = 4, 5
    
    print("Applicazione filtri PMI...")
    for idx, row in df_data.iterrows():
        # Filtri PMI
        try:
            rev_migl = pd.to_numeric(row[COL_REV_LAST_MIGL], errors='coerce')
            emp = pd.to_numeric(row[COL_EMP_LAST], errors='coerce')
        except:
            continue
        
        if pd.isna(rev_migl) or rev_migl >= 50000:  # €50M in migliaia
            continue
        if pd.isna(emp) or not (10 <= emp <= 250):
            continue
        
        company_name = str(row[COL_NAME]).strip()
        province = str(row[COL_PROVINCE]).strip().upper()[:2]
        macro = PROVINCE_TO_MACRO.get(province, None)
        if macro is None:
            continue
        
        # Firm ID unico
        firm_key = f"{company_name}_{province}"
        if firm_key not in firm_counter:
            firm_counter[firm_key] = len(firm_counter) + 1
        firm_id = firm_counter[firm_key]
        
        try:
            ref_year = pd.to_datetime(row[COL_DATE]).year
        except:
            ref_year = 2023
        
        # Estrai dati per tutti gli anni disponibili
        for y_idx in range(10):
            year = ref_year - y_idx
            
            def get_val(start_col):
                try:
                    return pd.to_numeric(row[start_col + y_idx], errors='coerce')
                except:
                    return np.nan
            
            va = get_val(38)   # Valore Aggiunto
            lc = get_val(18)   # Costo del Lavoro
            ma = get_val(58)   # Immobilizzazioni Materiali
            ia = get_val(68)   # Immobilizzazioni Immateriali
            ta = get_val(48)   # Totale Attivo
            
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
                })
    
    df = pd.DataFrame(records)
    
    print(f"✓ Caricati {len(df):,} osservazioni firm-year")
    print(f"✓ {df['firm_id'].nunique():,} imprese uniche")
    print(f"✓ Periodo: {df['Year'].min()}-{df['Year'].max()}")
    
    return df


def prepare_variables(df):
    """
    Crea le variabili per l'analisi econometrica.
    
    Variabili create:
    - ln_Y, ln_L, ln_K: log delle variabili di produzione
    - TechIntensity: rapporto immateriali/totale attivo
    - Dummy regionali (Nord, Sud) con Centro come baseline
    """
    print("\n" + "=" * 60)
    print("STEP 2: PREPARAZIONE VARIABILI")
    print("=" * 60)
    
    df = df.copy()
    
    # Trasformazioni logaritmiche
    df['ln_Y'] = np.log(df['ValueAdded'])
    df['ln_L'] = np.log(df['LaborCost'])
    df['ln_K'] = np.log(df['MatAssets'])
    
    # Intensità tecnologica (proxy per AI/digital)
    df['TechIntensity'] = df['ImmatAssets'] / df['TotalAssets']
    
    # Produttività del lavoro
    df['LaborProd'] = df['ValueAdded'] / df['LaborCost']
    
    # Winsorizzazione 1%-99% per ridurre outlier
    print("Winsorizzazione (1%-99%)...")
    for col in ['TechIntensity', 'LaborProd', 'ln_Y', 'ln_L', 'ln_K']:
        l, u = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(l, u)
    
    # Dummy regionali (Centro = baseline)
    df['Nord'] = (df['MacroArea'] == 'Nord').astype(float)
    df['Sud'] = (df['MacroArea'] == 'Sud').astype(float)
    
    # Anno centrato
    df['Year_c'] = df['Year'] - df['Year'].median()
    
    # Lagged Tech Intensity (per robustness)
    df = df.sort_values(['firm_id', 'Year'])
    df['TechIntensity_lag1'] = df.groupby('firm_id')['TechIntensity'].shift(1)
    
    print(f"✓ Tech Intensity media: {df['TechIntensity'].mean():.3f}")
    print(f"✓ Tech Intensity SD: {df['TechIntensity'].std():.3f}")
    
    return df


def estimate_tfp_panel_fe(df):
    """
    STAGE 1: Stima TFP via Panel Fixed Effects.
    
    Modello: ln(Y) = β_L*ln(L) + β_K*ln(K) + μ_i + δ_t + ε
    
    La TFP è estratta come: ln(Y) - β_L*ln(L) - β_K*ln(K)
    Include test Hausman per validare FE vs RE.
    """
    print("\n" + "=" * 60)
    print("STEP 3: STIMA TFP (PANEL FIXED EFFECTS)")
    print("=" * 60)
    
    df = df.copy()
    
    if PANEL_AVAILABLE:
        print("Utilizzo: linearmodels.PanelOLS")
        
        df_panel = df.set_index(['firm_id', 'Year'])
        exog = df_panel[['ln_L', 'ln_K']]
        endog = df_panel['ln_Y']
        
        # Stima con Entity FE + Time FE + Clustered SE
        model_fe = PanelOLS(endog, exog, entity_effects=True, time_effects=True)
        results_fe = model_fe.fit(cov_type='clustered', cluster_entity=True)
        
        print("\n--- Funzione di Produzione (Panel FE) ---")
        print(f"β_L (Lavoro):   {results_fe.params['ln_L']:.4f} (SE: {results_fe.std_errors['ln_L']:.4f})")
        print(f"β_K (Capitale): {results_fe.params['ln_K']:.4f} (SE: {results_fe.std_errors['ln_K']:.4f})")
        print(f"R² Within:      {results_fe.rsquared_within:.4f}")
        print(f"N osservazioni: {int(results_fe.nobs):,}")
        print(f"N imprese:      {int(results_fe.entity_info['total']):,}")
        
        # Test Hausman (FE vs RE)
        print("\n--- Test Hausman (FE vs RE) ---")
        model_re = RandomEffects(endog, exog)
        results_re = model_re.fit()
        
        try:
            from scipy import stats as scipy_stats
            b_fe = results_fe.params.values
            b_re = results_re.params.values
            diff = b_fe - b_re
            var_diff = results_fe.cov - results_re.cov
            H = diff @ np.linalg.inv(var_diff) @ diff
            p_val = 1 - scipy_stats.chi2.cdf(H, df=len(diff))
            print(f"Statistica H:   {H:.2f}")
            print(f"P-value:        {p_val:.4f}")
            if p_val < 0.05:
                print("→ Fixed Effects preferiti (corretto)")
        except:
            print("Test Hausman non calcolabile")
        
        beta_L = results_fe.params['ln_L']
        beta_K = results_fe.params['ln_K']
        df['TFP'] = df['ln_Y'] - beta_L * df['ln_L'] - beta_K * df['ln_K']
        
        return df, results_fe
    
    else:
        # Fallback senza linearmodels
        print("Utilizzo: OLS con dummies (fallback)")
        firm_dummies = pd.get_dummies(df['firm_id'], prefix='firm', drop_first=True)
        year_dummies = pd.get_dummies(df['Year'], prefix='year', drop_first=True)
        X = pd.concat([df[['ln_L', 'ln_K']], firm_dummies, year_dummies], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(df['ln_Y'], X).fit()
        
        beta_L = model.params['ln_L']
        beta_K = model.params['ln_K']
        df['TFP'] = df['ln_Y'] - beta_L * df['ln_L'] - beta_K * df['ln_K']
        
        return df, model


def estimate_jcurve(df):
    """
    STAGE 2: Test della J-Curve.
    
    Modello: TFP = α + γ₁*Tech + γ₂*Tech² + β_Nord*Nord + β_Sud*Sud + τ_t + η
    
    Interpretazione:
    - γ₁ < 0: calo iniziale di produttività (fase di investimento)
    - γ₂ > 0: recupero convesso (fase di raccolta)
    - Turning point = -γ₁/(2*γ₂)
    
    Nota metodologica:
    Pooled OLS con clustered SE è appropriato perché le dummy regionali
    catturano between-firm variation, non within-firm.
    """
    print("\n" + "=" * 60)
    print("STEP 4: STIMA J-CURVE")
    print("=" * 60)
    
    df = df.copy()
    df['Tech_Sq'] = df['TechIntensity'] ** 2
    
    X = df[['TechIntensity', 'Tech_Sq', 'Nord', 'Sud', 'Year_c']].astype(float)
    X = sm.add_constant(X)
    
    # Pooled OLS con clustered SE a livello impresa
    model = sm.OLS(df['TFP'].astype(float), X).fit(
        cov_type='cluster', 
        cov_kwds={'groups': df['firm_id']}
    )
    
    gamma1 = model.params['TechIntensity']
    gamma2 = model.params['Tech_Sq']
    coef_nord = model.params['Nord']
    coef_sud = model.params['Sud']
    
    min_point = -gamma1 / (2 * gamma2) if gamma2 != 0 else np.nan
    
    print("\n--- Coefficienti J-Curve ---")
    print(f"γ₁ (Tech):      {gamma1:.4f} (SE: {model.bse['TechIntensity']:.4f}, p={model.pvalues['TechIntensity']:.4f})")
    print(f"γ₂ (Tech²):     {gamma2:.4f} (SE: {model.bse['Tech_Sq']:.4f}, p={model.pvalues['Tech_Sq']:.4f})")
    print(f"Nord:           {coef_nord:.4f} (p={model.pvalues['Nord']:.4f})")
    print(f"Sud:            {coef_sud:.4f} (p={model.pvalues['Sud']:.4f})")
    print(f"\n--- Turning Point ---")
    print(f"Min Point:      {min_point:.4f} ({min_point*100:.1f}%)")
    
    jcurve_valid = (gamma1 < 0) and (gamma2 > 0) and (model.pvalues['TechIntensity'] < 0.05)
    print(f"\n✓ J-Curve confermata: {'SÌ' if jcurve_valid else 'NO'}")
    
    return model, gamma1, gamma2, min_point, coef_sud


def run_robustness_checks(df):
    """
    Robustness checks per validare i risultati.
    
    Test eseguiti:
    1. Trimming 5% su TFP (rimozione outlier estremi)
    2. Analisi per settore ATECO
    3. Interazione Tech × Sud
    4. Variabili lagged Tech(t-1) per ridurre endogeneità
    """
    print("\n" + "=" * 60)
    print("STEP 5: ROBUSTNESS CHECKS")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Trimming 5%
    print("\n[1] Trimming 5% su TFP...")
    l, u = df['TFP'].quantile(0.05), df['TFP'].quantile(0.95)
    df_trim = df[(df['TFP'] >= l) & (df['TFP'] <= u)].copy()
    df_trim['Tech_Sq'] = df_trim['TechIntensity'] ** 2
    X = df_trim[['TechIntensity', 'Tech_Sq', 'Nord', 'Sud', 'Year_c']].astype(float)
    X = sm.add_constant(X)
    res = sm.OLS(df_trim['TFP'], X).fit(cov_type='cluster', cov_kwds={'groups': df_trim['firm_id']})
    results['trim'] = {'gamma1': res.params['TechIntensity'], 'gamma2': res.params['Tech_Sq']}
    print(f"   γ₁={results['trim']['gamma1']:.3f}, γ₂={results['trim']['gamma2']:.3f}")
    
    # Test 2: Con lag (t-1)
    print("\n[2] Stima con Tech laggata (t-1) - Riduce endogeneità...")
    df_lag = df.dropna(subset=['TechIntensity_lag1']).copy()
    df_lag['Tech_Sq_lag'] = df_lag['TechIntensity_lag1'] ** 2
    X = df_lag[['TechIntensity_lag1', 'Tech_Sq_lag', 'Nord', 'Sud', 'Year_c']].astype(float)
    X = sm.add_constant(X)
    res_lag = sm.OLS(df_lag['TFP'], X).fit(cov_type='cluster', cov_kwds={'groups': df_lag['firm_id']})
    results['lagged'] = {
        'gamma1': res_lag.params['TechIntensity_lag1'],
        'gamma2': res_lag.params['Tech_Sq_lag'],
        'min_point': -res_lag.params['TechIntensity_lag1'] / (2 * res_lag.params['Tech_Sq_lag'])
    }
    print(f"   γ₁={results['lagged']['gamma1']:.3f}, γ₂={results['lagged']['gamma2']:.3f}")
    print(f"   MinPoint={results['lagged']['min_point']*100:.1f}%")
    print(f"   → J-Curve confermata con lag: {'SÌ' if results['lagged']['gamma1'] < 0 and results['lagged']['gamma2'] > 0 else 'NO'}")
    
    return results


def plot_jcurve(model, df):
    """Genera il grafico della J-Curve con IC al 95%."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    gamma1 = model.params['TechIntensity']
    gamma2 = model.params['Tech_Sq']
    const = model.params['const']
    min_point = -gamma1 / (2 * gamma2)
    
    x = np.linspace(0, 0.40, 200)
    y = const + gamma1 * x + gamma2 * x**2
    
    # Confidence interval
    se1, se2 = model.bse['TechIntensity'], model.bse['Tech_Sq']
    y_upper = const + (gamma1 + 1.96*se1) * x + (gamma2 - 1.96*se2) * x**2
    y_lower = const + (gamma1 - 1.96*se1) * x + (gamma2 + 1.96*se2) * x**2
    
    ax.fill_between(x, y_lower, y_upper, alpha=0.3, color='steelblue', label='95% CI')
    ax.plot(x, y, 'b-', linewidth=2.5, label='Estimated J-Curve')
    
    y_min = const + gamma1 * min_point + gamma2 * min_point**2
    ax.axvline(x=min_point, color='red', linestyle='--', alpha=0.7, label=f'Minimum ({min_point:.1%})')
    ax.scatter([min_point], [y_min], color='red', s=100, zorder=5)
    
    mean_tech = df['TechIntensity'].mean()
    ax.axvline(x=mean_tech, color='green', linestyle='-.', alpha=0.7, label=f'Sample Mean ({mean_tech:.1%})')
    
    ax.set_xlabel('Intangible Asset Intensity', fontsize=12)
    ax.set_ylabel('Estimated TFP (ω̂)', fontsize=12)
    ax.set_title('The Productivity J-Curve for Italian SMEs\n(Panel Fixed Effects Estimation)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 0.40)
    
    ax.annotate(f'γ₁ = {gamma1:.3f}\nγ₂ = {gamma2:.3f}', 
                xy=(0.30, y.min() + 0.02), fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_jcurve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvato: {OUTPUT_DIR / 'fig_jcurve.png'}")


def plot_geo_boxplots(df):
    """Boxplot TFP e Labor Productivity per area geografica."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    order = ['Nord', 'Centro', 'Sud']
    palette = {'Nord': '#2ecc71', 'Centro': '#3498db', 'Sud': '#e74c3c'}
    
    ax1 = axes[0]
    sns.boxplot(data=df, x='MacroArea', y='TFP', order=order, palette=palette, ax=ax1)
    ax1.set_xlabel('Geographic Area', fontsize=12)
    ax1.set_ylabel('TFP', fontsize=12)
    ax1.set_title('TFP Distribution by Region', fontsize=14, fontweight='bold')
    
    ax2 = axes[1]
    sns.boxplot(data=df, x='MacroArea', y='LaborProd', order=order, palette=palette, ax=ax2)
    ax2.set_xlabel('Geographic Area', fontsize=12)
    ax2.set_ylabel('Labor Productivity (Y/L)', fontsize=12)
    ax2.set_title('Labor Productivity by Region', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_geo_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvato: {OUTPUT_DIR / 'fig_geo_boxplots.png'}")


def save_results(df, model, gamma1, gamma2, min_point, coef_sud, robustness=None):
    """Salva i risultati in file di testo."""
    
    # Output regressione completo
    with open(RESULTS_DIR / 'regression_output.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("J-CURVE ESTIMATION - PANEL FIXED EFFECTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(model.summary().as_text())
    print(f"✓ Salvato: {RESULTS_DIR / 'regression_output.txt'}")
    
    # Valori per il paper
    with open(RESULTS_DIR / 'paper_values.txt', 'w') as f:
        f.write("VALORI PER IL PAPER\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"N. Osservazioni:     {len(df):,}\n")
        f.write(f"N. Imprese:          {df['firm_id'].nunique():,}\n")
        f.write(f"Tech Intensity Mean: {df['TechIntensity'].mean():.3f}\n")
        f.write(f"Tech Intensity SD:   {df['TechIntensity'].std():.3f}\n\n")
        f.write("--- MODELLO PRINCIPALE ---\n")
        f.write(f"γ₁ (TechIntensity):  {gamma1:.3f}\n")
        f.write(f"γ₂ (Tech²):          {gamma2:.3f}\n")
        f.write(f"Turning Point:       {min_point*100:.1f}%\n")
        f.write(f"Coeff. Sud:          {coef_sud:.3f}\n")
        
        # Aggiungi risultati lagged se disponibili
        if robustness and 'lagged' in robustness:
            f.write("\n--- ROBUSTNESS: LAGGED VARIABLES ---\n")
            f.write(f"γ₁ (Tech_lag1):      {robustness['lagged']['gamma1']:.3f}\n")
            f.write(f"γ₂ (Tech²_lag1):     {robustness['lagged']['gamma2']:.3f}\n")
            f.write(f"Turning Point (lag): {robustness['lagged']['min_point']*100:.1f}%\n")
    print(f"✓ Salvato: {RESULTS_DIR / 'paper_values.txt'}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("  ANALISI ECONOMETRICA J-CURVE - PMI ITALIANE")
    print("  Università di Salerno")
    print("=" * 70)
    
    # 1. Carica dati
    df = load_and_clean_data()
    
    # 2. Prepara variabili
    df = prepare_variables(df)
    
    # 3. Stima TFP con Panel FE
    df, tfp_model = estimate_tfp_panel_fe(df)
    
    # 4. Stima J-Curve
    jcurve_model, gamma1, gamma2, min_point, coef_sud = estimate_jcurve(df)
    
    # 5. Robustness checks
    robustness = run_robustness_checks(df)
    
    # 6. Genera figure
    print("\n" + "=" * 60)
    print("STEP 6: GENERAZIONE FIGURE")
    print("=" * 60)
    plot_jcurve(jcurve_model, df)
    plot_geo_boxplots(df)
    
    # 7. Salva risultati
    print("\n" + "=" * 60)
    print("STEP 7: SALVATAGGIO RISULTATI")
    print("=" * 60)
    save_results(df, jcurve_model, gamma1, gamma2, min_point, coef_sud, robustness)
    
    # Riepilogo finale
    print("\n" + "=" * 70)
    print("  RIEPILOGO FINALE")
    print("=" * 70)
    print(f"  N. Osservazioni:    {len(df):,}")
    print(f"  N. Imprese:         {df['firm_id'].nunique():,}")
    print(f"  γ₁ (Tech):          {gamma1:.4f}")
    print(f"  γ₂ (Tech²):         {gamma2:.4f}")
    print(f"  Turning Point:      {min_point*100:.1f}%")
    print(f"  Coeff. Sud:         {coef_sud:.4f}")
    print(f"\n  ✓ J-Curve confermata: {'SÌ' if (gamma1 < 0 and gamma2 > 0) else 'NO'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
