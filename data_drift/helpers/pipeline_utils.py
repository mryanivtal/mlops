# generic 
import os
import sys 
import json
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.pyplot import get_cmap
from sklearn.model_selection import train_test_split

# data drift specific
# from boston_ds import BostonDS
from utils import calc_perf_kpis
from model_helper import XgbModel
from helpers.data_helper import sample_from_data,change_int_values
from drift_detector import DriftDetector
from drift_testers.ks_drift_tester import KsDriftTester

def load_configuration(config_file='config.json',verbose=False):
    '''
    This function looks for a config.json file on the given config_path
    Loads it as a Dictionary from JSON
    '''
    config = {}
    with open(config_file) as json_file:
        config = json.load(json_file)
    
        print("Loading configuration from:",config_file )
        print("*"*80)
    
        if(verbose):
            for k,v in config.items():
                print('Key={},Value={}'.format(k,v))
        print("*"*80)
    return config


# Initial baseline predction / drift test plan
def get_model_kpi(model,drift_detector,X_test,y_test):
  '''
  Initial baseline predction / drift test plan
  '''
  y_pred = model.predict(X_test)
  kpi_sample = calc_perf_kpis(X_test, y_test, y_pred)
  return kpi_sample

def init_result_dfs(X_test,y_test,model,drift_detector):
  '''
  Calc and store initial model performance KPIs on test
  '''
  drift_test_results = drift_detector.test_drift(X_test)
  y_pred = model.predict(X_test)
  kpi = calc_perf_kpis(X_test, y_test, y_pred)

  kpi['drift_detected'] = drift_test_results['drift_detected']
  kpi['test_exceptions'] = drift_test_results['test_exceptions']
  kpi_keys = [k for k in kpi.keys()]
  kpi_columns = kpi_keys.append('retrain')
  kpi['retrain'] = False

  perf_kpis_prod = pd.DataFrame(columns=kpi_columns).append(kpi, ignore_index=True)
  perf_kpis_base = pd.DataFrame(columns=kpi_columns).append(kpi, ignore_index=True)

  return perf_kpis_base,perf_kpis_prod


def add_artificial_noise_to_data(X,X_sample,current_batch,columns_to_modify=[],
                                 x_cat_features=[],start_drift_at_batch=100,noise_factor=0.01,
                                 numeric_noise_factor=1.5):
  '''
  Adds some noise to the data for simulating feature drift
  Input:
    - X - the entire train dataset
    - X_sample - a small sample used for inference
    - current_batch - the current batch index
    - columns_to_modify - list of columns to apply noise on 
    - x_cat_features - list of categorical features
    - start_drift_at_batch - index of first drift start
    - noise_factor - categorical features noise factor
    - numeric_noise_factor - numeric features noise factor
    
    Returns:
    - The X_sample dataset with noise
  '''
  # modify data batch to create feature drift
  for c in columns_to_modify:
    if c in x_cat_features:
      # Constant noise categorical
      X_sample = change_int_values(X_sample, c, 0, 1, noise_factor) 
      #For Increasing NOISE use:
      #X_sample = change_int_values(X_sample, c, 0, 1, noise_factor) * (current_batch - start_drift_at_batch))
    else:
      # Constant noise numerical
      X_sample[c] = X_sample[c] + numeric_noise_factor* X[c].std()
      #For Increasing NOISE use:
      #X_sample[c] = X_sample[c] + 1.5* X[c].std() * noise_factor * (current_batch - start_drift_at_batch)
  return X_sample.copy()

# ========================================================================== Plot
def display_run(perf_kpis,drift_detector,simulated_drift_started_at_batch=100,axs=None,title='',
                baseline_perf_kpis=None,smoothen=True,show_train=False):
  '''
  Plots the results of a demo run including 2 detected tests (first and second), noise and retraining markers
  '''
  def get_y_smoothen(y):
    yhat = savgol_filter(y, 15, 3) # window size 51, polynomial order 3
    return yhat

  # plot RMSE (Loss function) line
  if(smoothen):
    axs.plot(get_y_smoothen(perf_kpis['RMSE']))
  else:
    axs.plot(perf_kpis['RMSE'])
  
  if baseline_perf_kpis is not None:
    if(smoothen):
      axs.plot(get_y_smoothen(baseline_perf_kpis['RMSE']))
    else:
      axs.plot(baseline_perf_kpis['RMSE'])


  if(show_train):  
    retrain_index = [r for r in perf_kpis[perf_kpis['retrain']==True].index]
    for r in retrain_index:
      # plot vertical line for each retraining
      axs.axvline(x=r, label='retrain_'+str(r), color='black', linestyle='dashed')


  # plot vertical line for data drift start point
  first_drift = simulated_drift_started_at_batch
  second_drift = 3*simulated_drift_started_at_batch
  axs.axvline(x=first_drift, label='drift_start_'+str(first_drift), color='r', linestyle='dashed')
  axs.axvline(x=second_drift, label='drift_start_'+str(second_drift), color='r', linestyle='dashed')

  # Get drift detector x_history for plots
  x_history = drift_detector.history_df

  # plot vertical line for each tester that fired
  fail_detections = []
  cmap = get_cmap('hsv', 15)

  for i, test_name in enumerate(drift_detector.get_test_names()):
      if x_history[test_name].sum() > 0:
          rand_small = random.uniform(-5, 1)
          all_detections = np.where(x_history[test_name] == True)[0]
          first_detection_time = all_detections.min()  #show first detection of demo
          axs.axvline(x=first_detection_time+rand_small, label=test_name, color=cmap(i))
          detection_time2 = all_detections.max()-5 #show detection towards the end of the demo
          axs.axvline(x=detection_time2+rand_small, label=test_name+'2', color=cmap(i))

  # Display plot
  axs.legend()
  axs.set_xlabel('batch number')
  axs.set_ylabel('RMSE')
  axs.set_title(title)


def show_drift_detection_step(perf_kpis_base,perf_kpis_prod,title=''):
  '''
  A Step function plot of the status of the drift detectors, baseline vs. production
  '''
  fig, axs = plt.subplots(2,1,figsize=(30, 5))
  base_y = perf_kpis_base['drift_detected']
  base_x = range(len(base_y))
  axs[0].step(base_x, base_y, '-r*', where='post',label='baseline')
  
  prod_y = perf_kpis_prod['drift_detected']
  prod_x = range(len(prod_y))
  axs[1].step(prod_x, prod_y, '-r*', where='post',label='production',color='navy')
  
  plt.xlabel('Batch number')
  plt.ylabel('Drift Status')
  axs[0].set_yticks([0,1])
  axs[1].set_yticks([0,1])
  axs[0].set_title('Baseline')
  axs[1].set_title('Production')
  plt.suptitle(title)
  plt.tight_layout()
  plt.show()

def report_summary_kpi(perf_kpis_base,perf_kpis_prod,last_k_batches=50,selected_dataset='BOSTON'):
  '''
  Reports the summary KPI 
  '''
  mean_base = perf_kpis_base.iloc[-last_k_batches]['RMSE'].mean()
  mean_prod = perf_kpis_prod.iloc[-last_k_batches]['RMSE'].mean()
  diff_base_prod = mean_base-mean_prod
  print('Average RMSE on last {} batches: Baseline:{:.4f} vs. Production:{:.4f}'.format(last_k_batches,mean_base,mean_prod))
  if(selected_dataset=='BOSTON'):
    print('Production model is better by {:.4f} , a potential saving of {:.2f}$ for the house predictions company'.format(diff_base_prod,diff_base_prod*1000))
  else:
    print('Production model is better by {:.4f} , a potential saving of {:.2f} claims per customer'.format(diff_base_prod,diff_base_prod))


# Build models on train data
def train_fresh_models(selected_dataset,X_train,y_train):
  '''
  Fresh start for baseline and production models
  (Returns two copies of the baseline model)
  '''
  baseline_model = XGBRegressor(objective='reg:squarederror')
  baseline_model.fit(X_train, y_train)

  saved_model_filename = selected_dataset + "_baseline_model.json"
  baseline_model.save_model(saved_model_filename)

  # Production model (a copy of the baseline initially)
  production_model = XGBRegressor(objective='reg:squarederror')
  production_model.load_model(saved_model_filename)

  return baseline_model,production_model

def init_test(X_train,y_train,X_test,y_test,
               x_cont_features,x_int_features,x_cat_features,
               selected_dataset):
  '''
  Init all required instances for a fresh test:
  Returns:
  1. 2 Models: baseline and production
  2. 2 drift detectors baseline and production
  3. 2 dataframes to collect the results
  '''

  baseline_model,production_model = train_fresh_models(selected_dataset,X_train,y_train)

  # Create Feature drift detector with all default testers
  drift_detector = DriftDetector()
  drift_detector.autoselect_testers(x_cont_features, x_int_features, x_cat_features)
  drift_detector.fit(X_train)

  #One dummy drift detector for baseline model
  base_drift_detector = DriftDetector()
  base_drift_detector.autoselect_testers(x_cont_features, x_int_features, x_cat_features)
  base_drift_detector.fit(X_train)

  perf_kpis_prod,perf_kpis_base = init_result_dfs(X_test,y_test,production_model,drift_detector)
    
  return drift_detector,perf_kpis_prod,base_drift_detector,perf_kpis_base,baseline_model,production_model
