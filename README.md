# Data Drift Detector 

This repository contains an implementation of a data drift detector, which can be used as an automatic step in a production pipeline, in order to detect drifts in Tabular datasets.

# Highlights:
- Automated statistical data drift detector module 
- Works on any tabular numerical data (Floats / Int / Categorical)
- Standard ML interface, easy to use (Create 🡪 fit 🡪 test)
- “Advanced mode” provides additional control:
  - Manual selection of statistical testers for features
  - Tester Sensitivity – using p-value
  - Tester Stability – repeated failures before alarm
- The module is extensible - very simple to add custom testers

# Implemented Data Drift Testers
- Univariate (Single feature) testers:
  - [Kolmogorov-Smirnov (KS) test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) – For numerical features
  - [Chi-Squared goodness of fit test](https://en.wikipedia.org/wiki/Chi-squared_test) – For categorical  features
- Multivariate testers:
  - [KL Divergence test](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) with calculated p-val for threshold over 5 runs
  - [Maximum Mean Discrepency test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) (Kernel based) - Numerical implementation

# Getting started

## Local installation
1. `git clone https://github.com/mryanivtal/mlops.git`

2. `pip install -r requirements.txt`

3. Follow the example in the following notebook: [Demo notebook](data_drift/data_drift_module_demo.ipynb)

# Basic Usage
## How to fit a drift detector to a dataset
![How to fit](data_drift/how_to_fit.png)

## How to test for drifts
![How to test](data_drift/how_2_test.png)

# E2E examples (Google Colab notebook)
There are 2 notebooks that demonstrate a complete inference pipeline drift detection (xgboost regressor) on 2 tabular datasets:
The Boston Housing  and French Motor Claims datasets.
As part of the pipeline we injected drift to selected features, and show how the drift detector is alerts about them.
The simplest running option is to upload the notebook to Google Colab (it will download everything it needs when you run it).
See: [E2E notebook](data_drift/e2e_data_drift_pipeline_demo.ipynb)




