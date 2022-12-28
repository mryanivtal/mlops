import json,os

# data drift imports
from data_helper import sample_from_data
from drift_detection.drift_detector import DriftDetector
from helpers.utils import calc_perf_kpis, recurring_val_in_lists
from drift_detection.drift_testers.mmd_drift_tester import MMDDriftTester
from drift_detection.drift_testers.pca_ks_drift_tester import PcaKsDriftTester

def load_configuration(config_file='config.json',verbose=False):
    '''
    This function looks for a config.json file on the given config_path
    Loads it as a Dictionary from JSON
    '''
    config = {}
    with open(config_file) as json_file:
        config = json.load(json_file)
    
        print("Loading configuration from:",config_file )
        print("*"*70)
    
        if(verbose):
            for k,v in config.items():
                print('\tKey={},Value={}'.format(k,v))
        print("*"*70)
    return config


# Test
# boston_config_dict = load_configuration(config_file='../examples/boston_config.json',verbose=True)





'''
# TODO DELETE
config_dict = {
    'dataset_name': 'Boston',
	'dataset_csv_path': 'datasets/boston_housing.csv',
	'cat_features': [],
    'int_features': ['CHAS', 'RAD', 'TAX'],
    'cont_features': ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'PTRATIO', 'B', 'LSTAT'],
    'target_label': 'PRICE'
}

print(config_dict)
# with open("config.json", "w") as fp:
#     json.dump(config_dict,fp) 
assert(config_dict==config_dict2)
print(config_dict==config_dict2)
'''