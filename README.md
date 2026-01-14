# The Impact of Digital Technologies on Productivity: An Empirical Analysis of the J-Curve in Italian SMEs

**Supplementary Material & Replication Package**

**Author:** Mario Trerotola  
**Affiliation:** University of Salerno  
**Course:** Economics of Artificial Intelligence  
**Date:** January 2026

---

## Abstract

While the “Productivity J-Curve” hypothesis is well-documented for large US firms, empirical evidence for Small and Medium-sized Enterprises remains scarce. This paper extends the framework to Italian manufacturing SMEs, using a two-stage Panel Fixed Effects model on a balanced panel of 8,683 firms from AIDA (2015–2024).

In the first stage, Total Factor Productivity is derived as the residual from a Cobb–Douglas production function. The second stage regresses TFP on Technological Intensity in quadratic form. A Hausman test ($H=49.35$, $p<0.001$) confirms the appropriateness of Fixed Effects.

Results validate the J-curve: initial investments reduce TFP ($\gamma_1 = -1.04$), with returns turning positive only beyond 14.6\% intensity ($\gamma_2 = +3.58$). Regarding regional heterogeneity, we find no robust productivity disadvantage for Southern SMEs, challenging the traditional North–South divide narrative. These findings suggest that policy should prioritize intangible capital accumulation over hardware subsidies, and that the “Solow Paradox” in Italy reflects delayed adjustment costs rather than technological failure.

---

## Data Availability

The dataset employed in this analysis is derived from the **AIDA** database (Bureau van Dijk), covering financial and structural data for Italian companies.
- **Reference Period**: 2015–2024
- **Sample Size**: 8,683 manufacturing SMEs (balanced panel)
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
*   **Initial Impact**: $\gamma_1 = -1.04$ (Significant TFP decline in early adoption phases)
*   **Long-term Recovery**: $\gamma_2 = +3.58$ (Positive returns at higher intensity levels)
*   **Turning Point**: The inflection point is estimated at **14.6%** Technological Intensity.

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
