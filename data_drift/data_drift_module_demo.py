import numpy as np
from matplotlib.pyplot import get_cmap
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
from boston_ds import BostonDS
from drift_detection.drift_testers.ks_drift_tester import KsDriftTester
from drift_detection.drift_testers.mmd_drift_tester import MMDDriftTester
from helpers.data_helper import sample_from_data, change_int_values
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
drift_detector = DriftDetector()
drift_detector.autoselect_testers(x_cont_features, x_int_features, x_cat_features)
# drift_detector.add_tester(MMDDriftTester('mmd', x_cont_features, dist_threshold=-1), consecutive_fails=3)

drift_detector.fit(x_train)

# initial drift test - initial test vs train
drift_test_results = drift_detector.test_drift(x_test)

# Calc and store initial model performance KPIs on test
kpi = calc_perf_kpis(x_test, y_test, y_pred)
kpi['drift_detected'] = drift_test_results['drift_detected']
kpi['test_exceptions'] = drift_test_results['test_exceptions']

perf_kpis = pd.DataFrame(columns=kpi.keys()).append(kpi, ignore_index=True)

# ===== Add control plots data collector
control_data = {}
for tester_name in drift_detector.drift_test_set.get_test_names():
    df = pd.DataFrame(columns=['threshold', 'value'])
    control_data[tester_name] = df

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
        x_sample = change_int_values(x_sample, 'CHAS', 0, 1, 0.01 * (i - start_drift_at_batch))

    # predict
    y_pred = model.predict(x_sample)

    # Execute drift test
    drift_test_results = drift_detector.test_drift(x_sample)

    # calc RMSE (For demo only, cannot do in real runtime - no labels there
    kpi_sample = calc_perf_kpis(x_sample, y_sample, y_pred)
    kpi_sample['drift_detected'] = drift_test_results['drift_detected']
    kpi_sample['test_exceptions'] = drift_test_results['test_exceptions']
    perf_kpis = perf_kpis.append(kpi_sample, ignore_index=True)

    # Get data for control charts
    for tester in drift_detector.drift_test_set.drift_testers:
        control_record = {
            'threshold': tester.get_threshold(),
            'value': tester.last_value}

        control_data[tester.test_name] = control_data[tester.test_name].append(control_record, ignore_index=True)
# ========================================================================== Plot

fig, axs = plt.subplots(figsize=(12, 12))

# plot RMSE (Loss function) line
axs.plot(perf_kpis['RMSE'])

# plot vertical lin for data drift start point
axs.axvline(x=start_drift_at_batch, label='drift_start', color='r', linestyle='dashed')

# Get drift detector x_history for plots
x_history = drift_detector.history_df

# plot vertical line for each tester that fired
fail_detections = []
cmap = get_cmap('hsv', 15)

for i, test_name in enumerate(drift_detector.get_test_names()):
    if x_history[test_name].sum() > 0:
        detection_time = np.where(x_history[test_name] == True)[0].min()
        axs.axvline(x=detection_time, label=test_name, color=cmap(i))

# Display plot
axs.legend()
plt.show()
print('Perf KPIs:', perf_kpis)

# ==============================================================Plot control
plots_to_display = ['ks_CRIM', 'kl_div', 'ks_RM']
n_testers = len(plots_to_display)
fig, axs = plt.subplots(n_testers, figsize=(12, 2*n_testers))
i = 0

for item in control_data.items():
    if item[0] in plots_to_display:
        axs[i].plot(item[1].iloc[:, 0], color='b')
        axs[i].plot(item[1].iloc[:, 1], color='r')
        axs[i].set_title(item[0], loc='left')
        i += 1

plt.show()





