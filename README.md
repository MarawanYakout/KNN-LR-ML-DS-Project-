# GDP and Migration Analysis Project using KNN & LR Machine Learning

## Overview

This project analyzes worldwide GDP trends and migration patterns in OECD countries using machine learning models to predict future GDP rates. The analysis combines historical GDP data with unemployment rates and migration statistics to provide comprehensive insights. Although the machine learning model did not include the migration or unemployment data in its analysis it used it in providing some data science analytics. The main analysis and machine learning algorithms were used to predict GDB changes from 2022 onwards using trained data from 1980 up till 2030 obtained from the official IMF website.

## Project Structure

```
├── organized_data/
│   ├── gdp_rate_worldwide.csv    # Combined GDP and unemployment data
│   └── migration_oecd.csv        # Migration statistics for OECD countries via optained from official Website
├── unorganised_data/
│   ├── gdp_imf.csv              # Raw GDP data from IMF
│   └── unemployment_imf.csv      # Raw unemployment data from IMF
├── Figures/
│   └── gbpvstime.png            # Generated visualizations
│   └── KNN_predictions.png      # Generated visualizations
│   └── LR_predictions.png      # Generated visualizations
├── d1_linear_scikit.py          # Main analysis script
├── ds_plots.py                  # Visualization functions
├── file_checker.py              # Data integrity verification
└── requirements.txt             # Project dependencies
```

## Features

- GDP prediction using KNN and Linear Regression models
- Historical data analysis from 1980 to 2022
- Future GDP predictions up to 2030
- Migration pattern analysis across OECD countries
- Data visualization and statistical analysis

## Requirements

```python
matplotlib
numpy
pandas
scikit-learn
```

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the file checker to verify data integrity:

```bash
python file_checker.py
```

2. Execute the main analysis script:

```bash
python knn_lr_gdp.py
```

3. Generate visualization plots:

```bash
python ds_plots.py
```

## Data Sources

- GDP and unemployment data: International Monetary Fund (IMF)
- Migration statistics: Organisation for Economic Co-operation and Development (OECD)

## Models

1. K-Nearest Neighbors (KNN)

   - Used for GDP prediction
   - Configured with n_neighbors = 4 (HIGHEST ACCURACY LOSSES ACCURACY IF LOWER OR HIGHER)
   - Includes accuracy metrics (MSE, MAE, R-squared)

2. Linear Regression
   - Alternative GDP prediction model
   - Provides trend analysis
   - Includes prediction confidence metrics

## Visualizations

The project generates several plots:

- Worldwide Real GDP Growth Rate Over Years
- Unemployment Rate Worldwide Growth Rate Over Years
- Migrant Numbers Over Years by Country
- GDP Predictions with KNN and Linear Regression Models

## Notes

- Training data spans from 1980 to 2022
- Test data includes years 2023-2030
- Migration data focuses on OECD countries from 2012 onwards - For exteded analysis on the data
