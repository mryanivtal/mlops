from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
from boston_ds import BostonDS
from data_helper import sample_from_data
from drift_detection.drift_detector import DriftDetector
from drift_detection.drift_testers.mmd_drift_tester import MMDDriftTester
from drift_detection.drift_testers.pca_ks_drift_tester import PcaKsDriftTester
from models import XgbModel
from utils import calc_perf_kpis, recurring_val_in_lists

# ============================================================= Initial data setup
# Load and prep boston data

# Get model data
# todo: generalize the dataset class,
# todo: get type of target (Regression / classification, maybe metadata - values / range)
boston = BostonDS()

x = boston.x
y = boston.y
x_cont_features = boston.cont_features
x_int_features = boston.int_features
x_cat_features = boston.cat_features

# from entire dataset, choose subset for initial train/test
TRAIN_TEST_SIZE = 520
x_sample, y_sample = sample_from_data(x, y, TRAIN_TEST_SIZE)
x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size=0.4, random_state=10)

# ============================================================= Train step

# todo: split this step to overall train and save references function
# Build model on train data
model = XgbModel()
model.fit(x_train, y_train)

# predict y on test data
y_pred = model.predict(x_test)

# Create drift detector with all default testers
drift_detector = DriftDetector()
drift_detector.add_default_testers(x_train, x_cont_features, x_int_features, x_cat_features, p_val_threshold=0.005)
drift_detector.add_tester(PcaKsDriftTester('pca_ks', x_test, x_cont_features + x_int_features, 0.1))
drift_detector.add_tester(MMDDriftTester('mmd', x_test, x_cont_features + x_int_features, 0.03))

# initial drift test - initial test vs train
drift_test_results = drift_detector.test_drift(x_test)

# Calc and store initial model performance KPIs on test
kpi = calc_perf_kpis(x_test, y_test, y_pred)
kpi['drift_found'] = drift_test_results['drift_found']
kpi['drift_exceptions'] = drift_test_results['data']['test_exceptions']

perf_kpis = pd.DataFrame(columns=kpi.keys()).append(kpi, ignore_index=True)

# mmd_results = MMD_pd(X_train, X_test)
# ============================================================= Runtime step
number_of_batches = 250
sample_size = 50

# Runtime loop
for i in range(number_of_batches):
    # Sample batch from data
    x_sample, y_sample = sample_from_data(x, y, sample_size)

    # modify data with trend (feature drift)
    x_sample['RM'] = x_sample['RM'] + x['RM'].mean() * 0.005 * i
    x_sample['LSTAT'] = x_sample['LSTAT'] + x['LSTAT'].mean() * 0.005 * i

    # predict
    y_pred = model.predict(x_sample)

    # calc RMSE (For demo only, cannot do in real runtime - no labels there
    kpi_sample = calc_perf_kpis(x_sample, y_sample, y_pred)

    # Drift test - Compare to ground truth, calc error kpis
    drift_test_results = drift_detector.test_drift(x_sample)
    kpi_sample['drift_found'] = drift_test_results['drift_found']
    kpi_sample['drift_exceptions'] = drift_test_results['data']['test_exceptions']
    perf_kpis = perf_kpis.append(kpi_sample, ignore_index=True)


# Find the first iteration where the feature changed twice in a row (to make sure chanage is consistent)
perf_kpis['RM_change'] = recurring_val_in_lists(perf_kpis['drift_exceptions'], 'ks_RM', 2)
perf_kpis['LSTAT_change'] = recurring_val_in_lists(perf_kpis['drift_exceptions'], 'ks_LSTAT', 2)
perf_kpis['pca_changed'] = recurring_val_in_lists(perf_kpis['drift_exceptions'], 'pca_ks', 1)
perf_kpis['mmd_changed'] = recurring_val_in_lists(perf_kpis['drift_exceptions'], 'mmd', 3)


# Plot
fig, axs = plt.subplots(figsize=(12, 12))
axs.plot(perf_kpis['RMSE'])

if perf_kpis['RM_change'].sum() > 0:
    first_rm = np.where(perf_kpis['RM_change'] == True)[0].min()
    axs.axvline(x=first_rm, color='r', label='RM_KS')

if perf_kpis['LSTAT_change'].sum() > 0:
    first_lstat = np.where(perf_kpis['LSTAT_change'] == True)[0].min()
    axs.axvline(x=first_lstat, color='g', label='LSTAT_KS')

if perf_kpis['pca_changed'].sum() > 0:
    first_pca = np.where(perf_kpis['pca_changed'] == True)[0].min()
    axs.axvline(x=first_pca, color='b', label='PCA_KS')

if perf_kpis['mmd_changed'].sum() > 0:
    first_mmd = np.where(perf_kpis['mmd_changed'] == True)[0].min()
    axs.axvline(x=first_mmd, color='y', label='MMD')

axs.legend()
plt.show()

print('Perf KPIs:', perf_kpis)
