# The Impact of Digital Technologies on Productivity: An Empirical Analysis of the J-Curve in Italian SMEs

**Supplementary Material & Replication Package**

**Author:** Mario Trerotola  
**Affiliation:** University of Salerno  
**Course:** Economics of Artificial Intelligence  
**Date:** January 2026

---

## � Abstract

Although the “Productivity J-Curve” hypothesis has been widely examined for large US firms, there is still limited empirical evidence for Small and Medium-sized Enterprises. This paper broadens the theoretical framework to an economy largely composed of SMEs and offers new evidence on the impact of digital and AI-related investments on productivity in Italy’s manufacturing sector.

We employ a two-stage Panel Fixed Effects econometric framework on a sample of more than 8,700 Italian SMEs sourced from AIDA (2014–2023). In the first stage, we derive Total Factor Productivity as the residual from a Cobb–Douglas production function, and in the second stage we regress this measure on Technological Intensity, specified in quadratic form to capture potential non-linear effects. A Hausman test supports the choice of Fixed Effects over Random Effects.

The results validate the presence of a J-curve: early-stage investments are associated with a drop in TFP ($\gamma_1 = -1.04$), whereas returns turn positive only once investment intensity exceeds 16.7\% ($\gamma_2 = +3.11$). Unexpectedly, Southern SMEs display a productivity advantage (+2.9\%), which we interpret as evidence of competitive selection. Overall, the evidence indicates that policy should focus on fostering intangible capital accumulation rather than relying solely on hardware subsidies.

---

## � Repository Contents

This repository contains the code, data, and supplementary materials required to replicate the findings presented in the paper.

```text
.
├── Tesi_Economia_IA_JCurve.tex    # Main manuscript (LaTeX source)
├── references.bib                 # Bibliography file
├── codice/
│   └── analisi_jcurve.py          # Main Python script for econometric analysis
├── dati/
│   └── Aida_Export_2.xls          # Raw dataset (AIDA, Bureau van Dijk)
├── figure/                        # Generated plots
│   ├── fig_jcurve.png             # Estimated J-Curve visualization
│   ├── fig_geo_boxplots.png       # Regional productivity distribution
│   └── fig_sector_bar.png         # Sectoral heterogeneity analysis
└── risultati/                     # Output logs and tables
```

---

## ⚙️ Methodology

The empirical strategy follows a two-stage approach:

1.  **TFP Estimation (Stage 1):** We estimate Total Factor Productivity using a **Panel Fixed Effects** model (Firm + Year effects) with clustered standard errors.
    $$ \ln(Y_{it}) = \beta_L \ln(L_{it}) + \beta_K \ln(K_{it}) + \mu_i + \delta_t + \epsilon_{it} $$

2.  **J-Curve Test (Stage 2):** We regress the estimated TFP on linear and squared technological intensity ($Tech$).
    $$ \hat{\omega}_{it} = \alpha + \gamma_1 Tech_{it} + \gamma_2 Tech_{it}^2 + \mathbf{Controls} + \eta_{it} $$

**Key Results:**
*   $\gamma_1 = -1.04$ (Initial TFP Drop)
*   $\gamma_2 = +3.11$ (Subsequent Recovery)
*   Turning Point: **16.7%** Intensity

---

## 💻 Replication Instructions

To reproduce the analysis and generate the figures, follow these steps:

### 1. Requirements
Ensure you have Python 3.8+ installed along with the following libraries:

```bash
pip install pandas numpy statsmodels linearmodels matplotlib seaborn scipy openpyxl
```

### 2. Execution
Run the main script from the `codice` directory:

```bash
cd codice
python analisi_jcurve.py
```

The script will:
1.  Load and clean the raw data from `dati/`.
2.  Perform the econometric estimation (Panel FE + Pooled OLS).
3.  Print summary statistics and regression results to the console.
4.  Generate and save all figures to the `figure/` directory.

---

## � Citation

If you use this code or data, please cite the associated paper:

> Trerotola, M. (2026). *The Impact of Digital Technologies on Productivity: An Empirical Analysis of the J-Curve in Italian SMEs*. University of Salerno.

For questions or issues with replication, please contact the author.
