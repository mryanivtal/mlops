from typing import List

import numpy as np
from matplotlib.pyplot import get_cmap
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt, cm, cycler
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
drift_detector.add_default_testers(x_cont_features, x_int_features, x_cat_features)
drift_detector.fit(x_train)

# initial drift test - initial test vs train
drift_test_results = drift_detector.test_drift(x_test)

# Calc and store initial model performance KPIs on test
kpi = calc_perf_kpis(x_test, y_test, y_pred)
kpi['drift_detected'] = drift_test_results['drift_detected']
kpi['test_exceptions'] = drift_test_results['test_exceptions']

perf_kpis = pd.DataFrame(columns=kpi.keys()).append(kpi, ignore_index=True)

# mmd_results = MMD_pd(X_train, X_test)
# ============================================================= Runtime step
number_of_batches = 300
start_drift_at_batch = 100
sample_size = 50


# Runtime loop
for i in range(number_of_batches):
    # Sample batch from data
    x_sample, y_sample = sample_from_data(x, y, sample_size)

    if i > start_drift_at_batch:
        # modify data with trend (feature drift)
        x_sample['RM'] = x_sample['RM'] + x['RM'].std() * 0.01 * (i - start_drift_at_batch)
        x_sample['LSTAT'] = x_sample['LSTAT'] + x['LSTAT'].std() * 0.01 * (i - start_drift_at_batch)

    # predict
    y_pred = model.predict(x_sample)

    # calc RMSE (For demo only, cannot do in real runtime - no labels there
    kpi_sample = calc_perf_kpis(x_sample, y_sample, y_pred)

    # Drift test
    drift_test_results = drift_detector.test_drift(x_sample)
    kpi_sample['drift_detected'] = drift_test_results['drift_detected']
    kpi_sample['test_exceptions'] = drift_test_results['test_exceptions']
    perf_kpis = perf_kpis.append(kpi_sample, ignore_index=True)

# Plot
history = drift_detector.history_df

fig, axs = plt.subplots(figsize=(12, 12))
axs.plot(perf_kpis['RMSE'])

# draw all drift detections on the plot - first detection point for each
fail_detections = []
cmap = get_cmap('hsv', 15)

axs.axvline(x=start_drift_at_batch, label='drift_start', color='r', linestyle='dashed')

for i, test_name in enumerate(drift_detector.get_test_names()):
    if history[test_name].sum() > 0:
        detection_time = np.where(history[test_name] == True)[0].min()
        axs.axvline(x=detection_time, label=test_name, color=cmap(i))

axs.legend()
plt.show()

print('Perf KPIs:', perf_kpis)
