import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
from boston import BostonDS
from data_helper import get_5x_dataset_with_noise, sample_from_data
from ks_test import KsTest
from models import XgbModel

# ============================================================= Initial data setup
# Load and prep boston data
from kpi_helper import calc_perf_kpis

boston = BostonDS()
x = boston.x
y = boston.y
x_cont_features = boston.cont_features

# split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=10)

# Create dataset 5x size with minor noise
noise_add_features = ['NOX', 'CRIM', 'DIS', 'AGE']
x_synthetic, y_synthetic = get_5x_dataset_with_noise(x, y, noise_add_features, noise_mu=0, noise_rate=0.01)

# ============================================================= Train step

# Build model on train data,
model = XgbModel()
model.fit(x_train, y_train)

# predict y on test data
y_pred = model.predict(x_test)

# Save reference x,y data for feature/concept drift tests
ref_X = x_train.copy()
ref_y = y_pred.copy()

# Calc initial model performance KPIs on test
kpi = calc_perf_kpis(x_test, y_test, y_pred)
kpi['ks_exceptions'] = KsTest(x_train, x_test).exceptions_list()

perf_kpis = pd.DataFrame(columns=kpi.keys()).append(kpi, ignore_index=True)

# mmd_results = MMD_pd(X_train, X_test)
# ============================================================= Runtime step
number_of_batches = 250
sample_size = 50

for i in range(number_of_batches):
    # Sample batch from synthetic data
    x_sample, y_sample = sample_from_data(x_synthetic, y_synthetic, sample_size)

    # modify data with trend (feature drift)
    x_sample['RM'] = x_sample['RM'] + x_synthetic['RM'].mean() * 0.005 * i
    x_sample['LSTAT'] = x_sample['LSTAT'] + x_synthetic['LSTAT'].mean() * 0.005 * i

    # predict, compare to ground truth, calc error kpis
    y_pred = model.predict(x_sample)
    kpi_sample = calc_perf_kpis(x_sample, y_sample, y_pred)
    kpi_sample['ks_exceptions'] = KsTest(x_train, x_sample).exceptions_list()
    perf_kpis = perf_kpis.append(kpi_sample, ignore_index=True)

    # Run ksTest on sample and original train data to compare distributions
    ks_results = KsTest(x_train, x_sample)

# Find the first iteration where the feature changed twice in a row (to make sure chanage is consistent)
perf_kpis['RM_change'] = perf_kpis['ks_exceptions'].apply(lambda row: 'RM' in row).rolling(2).aggregate(sum) == 2
first_rm = np.where(perf_kpis['RM_change'] == True)[0].min()

perf_kpis['LSTAT_change'] = perf_kpis['ks_exceptions'].apply(lambda row: 'LSTAT' in row).rolling(2).aggregate(sum) == 2
first_lstat = np.where(perf_kpis['LSTAT_change'] == True)[0].min()

# Plot
fig, axs = plt.subplots(figsize=(12, 12))
axs.plot(perf_kpis['RMSE'])
axs.axvline(x=first_rm, color='r')
axs.axvline(x=first_lstat, color='g')
plt.show()

print('Perf KPIs:', perf_kpis)
