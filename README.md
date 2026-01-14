# The Impact of Digital Technologies on Productivity: An Empirical Analysis of the J-Curve in Italian SMEs

**Supplementary Material & Replication Package**

**Author:** Mario Trerotola  
**Affiliation:** University of Salerno  
**Course:** Economics of Artificial Intelligence  
**Date:** January 2026

---

## Abstract

Although the “Productivity J-Curve” hypothesis has been widely examined for large US firms, there is still limited empirical evidence for Small and Medium-sized Enterprises (SMEs). This paper broadens the theoretical framework to an economy largely composed of SMEs and offers new evidence on the impact of digital and AI-related investments on productivity in Italy’s manufacturing sector.

We employ a two-stage Panel Fixed Effects econometric framework on a balanced panel of more than 8,600 Italian SMEs sourced from the AIDA database (2015–2024). In the first stage, we derive Total Factor Productivity (TFP) as the residual from a Cobb–Douglas production function. In the second stage, we regress this measure on Technological Intensity, specified in quadratic form to capture potential non-linear effects. A Hausman test supports the choice of Fixed Effects over Random Effects specification.

The results validate the presence of a J-curve: early-stage investments are associated with a temporary drop in TFP ($\gamma_1 = -1.04$), whereas returns turn positive only once investment intensity exceeds 14.6\% ($\gamma_2 = +3.58$). Regarding regional heterogeneity, Southern SMEs do not display any statistically significant productivity disadvantage compared to the rest of the country, challenging the traditional narrative of a North-South divide. Overall, the findings suggest that industrial policy should prioritize intangible capital accumulation over exclusive hardware subsidies.

---

## Data Availability

The dataset employed in this analysis is derived from the **AIDA** database (Bureau van Dijk), covering financial and structural data for Italian companies.
- **Reference Period**: 2015–2024
- **Sample Size**: 8,683 manufacturing SMEs (balanced panel)
- **File**: `dati/Aida_Export_2.xls`

*Note: Access to the raw data may be subject to subscription restrictions by Bureau van Dijk.*

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
├── codice/
│   └── analisi_jcurve.py          # Econometric analysis script (Python)
├── dati/
│   └── Aida_Export_2.xls          # Dataset
├── figure/                        # Verification and Results Plots
│   ├── fig_jcurve.png             # J-Curve Estimation
│   ├── fig_geo_boxplots.png       # Regional Heterogeneity
│   └── fig_sector_bar.png         # Sectoral Analysis
└── risultati/                     # Regression Logs
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
cd codice
python analisi_jcurve.py
```

**Process Overview:**
1.  **Data Preprocessing**: Loads and cleans the raw panel data from `dati/`.
2.  **Estimation**: Performs the Panel Fixed Effects TFP estimation and the subsequent quadratic regression.
3.  **Reporting**: Outputs summary statistics and regression tables to the console and `risultati/`.
4.  **Visualization**: Generates vector graphics and saves them to the `figure/` directory.

---

## Citation

Please cite this work as follows:

**Text:**
Trerotola, M. (2026). *The Impact of Digital Technologies on Productivity: An Empirical Analysis of the J-Curve in Italian SMEs*. University of Salerno.

**BibTeX:**
```bibtex
@misc{trerotola2026jcurve,
  author       = {Mario Trerotola},
  title        = {The Impact of Digital Technologies on Productivity: An Empirical Analysis of the J-Curve in Italian SMEs},
  year         = {2026},
  institution  = {University of Salerno},
  note         = {Replication Package}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
