import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
from boston_ds import BostonDS
from data_helper import sample_from_data
from drift_detection.chi_drift_tester import ChiDriftTester
from drift_detection.drift_test_set import DriftTestSet
from drift_detection.ks_drift_tester import KsDriftTester
from models import XgbModel
from kpi_helper import calc_perf_kpis

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

# Create drift test set with reference data
drift_test_set = DriftTestSet('drift_test_set')
P_VAL_THRESHOLD = 0.005
for feature in x_cont_features:
    test_name = 'ks_' + feature
    drift_test_set.add(KsDriftTester(test_name, x_train, feature, P_VAL_THRESHOLD))
    # todo: add also int features?

P_VAL_THRESHOLD = 0.99
for feature in x_cat_features:
    test_name = 'chi_' + feature
    drift_test_set.add(ChiDriftTester(test_name, x_train, feature, P_VAL_THRESHOLD))

# initial drift test - initial test vs train
drift_test_results = drift_test_set.test_drift(x_test)

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
    drift_test_results = drift_test_set.test_drift(x_sample)
    kpi['drift_found'] = drift_test_results['drift_found']
    kpi_sample['drift_exceptions'] = drift_test_results['data']['test_exceptions']
    perf_kpis = perf_kpis.append(kpi_sample, ignore_index=True)

# todo: need to move this to production like step within the loop and retrain where needed
# Find the first iteration where the feature changed twice in a row (to make sure chanage is consistent)
perf_kpis['RM_change'] = perf_kpis['drift_exceptions'].apply(lambda row: 'ks_RM' in row).rolling(2).aggregate(sum) == 2
first_rm = np.where(perf_kpis['RM_change'] == True)[0].min()

perf_kpis['LSTAT_change'] = perf_kpis['drift_exceptions'].apply(lambda row: 'ks_LSTAT' in row).rolling(2).aggregate(
    sum) == 2
first_lstat = np.where(perf_kpis['LSTAT_change'] == True)[0].min()

# Plot
fig, axs = plt.subplots(figsize=(12, 12))
axs.plot(perf_kpis['RMSE'])
axs.axvline(x=first_rm, color='r')
axs.axvline(x=first_lstat, color='g')
plt.show()

print('Perf KPIs:', perf_kpis)
