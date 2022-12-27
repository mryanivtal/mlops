import numpy as np
from matplotlib.pyplot import get_cmap
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
from boston_ds import BostonDS
from drift_detection.drift_testers.ks_drift_tester import KsDriftTester
from helpers.data_helper import sample_from_data
from helpers.model_helper import XgbModel
from helpers.utils import calc_perf_kpis

from drift_detection.drift_detector import DriftDetector

# ============================================================= Initial data setup
# Load and prep boston data
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

# Build model on train data
model = XgbModel()
model.fit(x_train, y_train)

# predict y on test data
y_pred = model.predict(x_test)

# Create Feature drift detector with all default testers
x_drift_detector = DriftDetector()
x_drift_detector.add_default_testers(x_cont_features, x_int_features, x_cat_features)

# create Concept drift detector with KL divergence
y_drift_detector = DriftDetector()
y_drift_detector.add_tester(KsDriftTester('concept_ks', 'y', 0.005))

x_drift_detector.fit(x_train)
y_drift_detector.fit(pd.DataFrame(y_pred, columns=['y']))

# initial drift test - initial test vs train
x_drift_test_results = x_drift_detector.test_drift(x_test)

# Calc and store initial model performance KPIs on test
kpi = calc_perf_kpis(x_test, y_test, y_pred)
kpi['drift_detected'] = x_drift_test_results['drift_detected']
kpi['test_exceptions'] = x_drift_test_results['test_exceptions']

perf_kpis = pd.DataFrame(columns=kpi.keys()).append(kpi, ignore_index=True)

# ============================================================= Runtime step

number_of_batches = 300
start_drift_at_batch = 100
sample_size = 50


# Runtime loop
for i in range(number_of_batches):
    # Sample batch from data (No drift yet)
    x_sample, y_sample = sample_from_data(x, y, sample_size)

    # modify data batch to create feature drift
    if i > start_drift_at_batch:
        x_sample['RM'] = x_sample['RM'] + x['RM'].std() * 0.01 * (i - start_drift_at_batch)
        x_sample['LSTAT'] = x_sample['LSTAT'] + x['LSTAT'].std() * 0.01 * (i - start_drift_at_batch)

    # predict
    y_pred = model.predict(x_sample)

    # calc RMSE (For demo only, cannot do in real runtime - no labels there
    kpi_sample = calc_perf_kpis(x_sample, y_sample, y_pred)

    # Execute drift test
    x_drift_test_results = x_drift_detector.test_drift(x_sample)
    y_drift_test_results = y_drift_detector.test_drift(pd.DataFrame(y_pred, columns=['y']))

    kpi_sample['drift_detected'] = x_drift_test_results['drift_detected'] & y_drift_test_results['drift_detected']
    kpi_sample['test_exceptions'] = x_drift_test_results['test_exceptions'] + y_drift_test_results['test_exceptions']
    perf_kpis = perf_kpis.append(kpi_sample, ignore_index=True)

# ========================================================================== Plot

fig, axs = plt.subplots(figsize=(12, 12))

# plot RMSE (Loss function) line
axs.plot(perf_kpis['RMSE'])

# plot vertical lin for data drift start point
axs.axvline(x=start_drift_at_batch, label='drift_start', color='r', linestyle='dashed')

# Get drift detector x_history for plots
x_history = x_drift_detector.history_df
y_history = y_drift_detector.history_df

# plot vertical line for each tester that fired
fail_detections = []
cmap = get_cmap('hsv', 15)

for i, test_name in enumerate(x_drift_detector.get_test_names()):
    if x_history[test_name].sum() > 0:
        detection_time = np.where(x_history[test_name] == True)[0].min()
        axs.axvline(x=detection_time, label=test_name, color=cmap(i))

for i, test_name in enumerate(y_drift_detector.get_test_names()):
    if y_history[test_name].sum() > 0:
        detection_time = np.where(y_history[test_name] == True)[0].min()
        axs.axvline(x=detection_time, label=test_name, color=cmap(i))

# Display plot
axs.legend()
plt.show()

print('Perf KPIs:', perf_kpis)
