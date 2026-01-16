# The Impact of Digital Technologies on Productivity: An Empirical Analysis of the J-Curve in Italian SMEs

**Supplementary Material & Replication Package**

**Author:** Mario Trerotola  
**Affiliation:** University of Salerno  
**Date:** January 2026

---

## Abstract

The “Productivity J-Curve” hypothesis posits that digital investments initially depress productivity before generating positive returns. Existing empirical evidence is largely confined to large publicly traded firms in the United States, with limited evidence for Small and Medium-sized Enterprises (SMEs). Using a balanced panel of 7,478 Italian manufacturing firms (2015–2024) from the AIDA database, we implement a two-stage Fixed Effects estimation strategy. First, we derive Total Factor Productivity from a Cobb–Douglas production function. Second, we regress TFP on technological intensity (intangible assets over total assets) in quadratic form, including regional interaction terms to test distinct North-South dynamics. Results confirm the J-Curve pattern: $\gamma_1 = -1.02$ ($p<0.001$) and $\gamma_2 = +3.69$ ($p<0.001$), with a turning point at 13.8% technological intensity. With a sample mean intensity of 3.1%, the findings indicate that the majority of Italian SMEs operate below the turning point, in the adjustment-cost phase. Crucially, interaction terms between technological intensity and the South dummy are statistically insignificant (Tech×South: $p=0.77$; Tech²×South: $p=0.70$), demonstrating that the productivity-technology relationship does not differ by region. This finding supports a *competitive selection* hypothesis: surviving Southern firms exhibit efficiency levels comparable to their Northern counterparts, suggesting that aggregate disparities reflect lower adoption intensity rather than intrinsic inefficiency.

---

## Data Availability

The dataset employed in this analysis is derived from the **AIDA** database (Bureau van Dijk), covering financial and structural data for Italian companies.
- **Reference Period**: 2015–2024
- **Sample Size**: 7,478 manufacturing SMEs (balanced panel)
- **File**: `data/Aida_Export_2.xls` (Not included in repository)

> [!IMPORTANT]
> **Data Access Disclaimer**: The raw dataset is **not included** in this repository due to licensing restrictions imposed by Bureau van Dijk. To replicate the study, researchers must obtain the `AIDA` dataset through their own institutional subscription.

---

## Empirical Framework

The empirical strategy follows a robust two-stage approach:

1.  **TFP Estimation (Stage 1):** Total Factor Productivity is estimated using a **Panel Fixed Effects** model (controlling for unobserved firm-specific heterogeneity and time-specific shocks) with clustered standard errors.
    $$ \ln(Y_{it}) = \beta_L \ln(L_{it}) + \beta_K \ln(K_{it}) + \mu_i + \delta_t + \epsilon_{it} $$

2.  **J-Curve Test (Stage 2):** The estimated TFP is regressed on linear and squared technological intensity ($Tech$) to test for non-linear returns to digital adoption.
    $$ \hat{\omega}_{it} = \alpha + \gamma_1 Tech_{it} + \gamma_2 Tech_{it}^2 + \mathbf{Controls} + \eta_{it} $$

### Main Findings
*   **Initial Impact**: $\gamma_1 = -1.02$ (Significant TFP decline in early adoption phases)
*   **Long-term Recovery**: $\gamma_2 = +3.69$ (Positive returns at higher intensity levels)
*   **Turning Point**: The inflection point is estimated at **13.8%** Technological Intensity.

---

## Repository Structure

This repository contains the code, data, and supplementary materials required to replicate the findings presented in the manuscript.

```text
.
├── Tesi_Economia_IA_JCurve.tex    # Main manuscript (LaTeX source)
├── references.bib                 # Bibliography file
├── code/
│   ├── main.py                    # Main execution script
│   └── src/                       # Source modules
│       ├── config.py              # Configuration & constants
│       ├── data_loader.py         # Data cleaning & loading
│       ├── econometrics.py        # TFP & J-Curve estimation
│       └── visualization.py       # Plotting utilities
├── data/
│   └── (Aida_Export_2.xls)        # Dataset (Not included)
├── figures/                       # Verification and Results Plots
│   ├── fig_jcurve.png             # J-Curve Estimation
│   ├── fig_geo_boxplots.png       # Regional Heterogeneity
│   ├── fig_sector_bar.png         # Sectoral Analysis
└── results/                       # Regression Logs
```

---

## Reproduction of Results

To reproduce the analysis and generate the figures, please follow the procedure outlined below:

### 1. Software Prerequisites
The analysis requires **Python 3.8+** and the following scientific computing libraries:

```bash
pip install pandas numpy statsmodels linearmodels matplotlib seaborn scipy openpyxl
```

### 2. Execution Strategy
Execute the primary analysis script located in the `codice` directory:

```bash
cd code
python main.py
```

**Process Overview:**
1.  **Data Preprocessing**: Loads and cleans the raw panel data from `data/`.
2.  **Estimation**: Performs the Panel Fixed Effects TFP estimation and the subsequent quadratic regression.
3.  **Reporting**: Outputs summary statistics and regression tables to the console and `results/`.
4.  **Visualization**: Generates vector graphics and saves them to the `figures/` directory.

### 3. Localization
The entire codebase, including comments, docstrings, and output logs, has been localized into **English** to ensure international accessibility.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
