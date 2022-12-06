from typing import List

import pandas as pd

from drift_detection.chi_drift_tester import ChiDriftTester
from drift_detection.drift_test_set import DriftTestSet
from drift_detection.ks_drift_tester import KsDriftTester


class DriftDetector:
    def __init__(self, dataset: pd.DataFrame, cont_features: List, int_features: List, cat_features: List):
        self.features = dataset.columns.to_list()
        self.cat_features = cat_features
        self.int_features = int_features
        self.cont_features = cont_features
        self.drift_test_set = DriftTestSet('drift_test_set')

        P_VAL_THRESHOLD = 0.005
        for feature in cont_features:
            test_name = 'ks_' + feature
            self.drift_test_set.add(KsDriftTester(test_name, dataset, feature, P_VAL_THRESHOLD))
            # todo: add also int features?

        for feature in cat_features:
            test_name = 'chi_' + feature
            self.drift_test_set.add(ChiDriftTester(test_name, dataset, feature, P_VAL_THRESHOLD))

    def test_drift(self, data: object):
        return self.drift_test_set.test_drift(data)

    # todo: add custom tester setup
    # Todo: add MMD tests, pca tests...
