from typing import List

import pandas as pd

from drift_detection.abstract_drift_tester import AbstractDriftTester
from drift_detection.drift_testers.chi_drift_tester import ChiDriftTester
from drift_detection.drift_test_set import DriftTestSet
from drift_detection.drift_testers.ks_drift_tester import KsDriftTester
from drift_detection.drift_testers.mmd_drift_tester import MMDDriftTester
from drift_detection.drift_testers.pca_ks_drift_tester import PcaKsDriftTester


class DriftDetector:
    def __init__(self):
        """
        Creates an empty drift detector object
        """
        self.drift_test_set = []

    def add_default_testers(self, dataset: pd.DataFrame, cont_features: List, int_features: List, cat_features: List):
        """
        Add all default tests to drift detector:
        KS for continuous features,
        Chi2 for categorical,
        todo: int2 currently not tested
        :param dataset: Pandas dataframe
        :param cont_features: column names of continuous features
        :param int_features: column names of integer features
        :param cat_features: column names of categorical features
        :param p_val_threshold: p value threshold for all tests
        """
        self.drift_test_set = DriftTestSet('drift_test_set')

        for feature in cont_features:
            test_name = 'ks_' + feature
            self.add_tester(KsDriftTester(test_name, dataset, feature, 0.005))
            # todo: add also int features?

        for feature in cat_features:
            test_name = 'chi_' + feature
            self.add_tester(ChiDriftTester(test_name, dataset, feature, 0.005))

        self.add_tester(PcaKsDriftTester('pca_ks', dataset, cont_features + int_features, 0.1))
        self.add_tester(MMDDriftTester('mmd', dataset, cont_features + int_features, 0.03))

    def add_tester(self, test: AbstractDriftTester):
        self.drift_test_set.add(test)

    def test_drift(self, data: object):
        return self.drift_test_set.test_drift(data)

