from typing import List
import pandas as pd
from drift_detection.drift_testers.abstract_drift_tester import AbstractDriftTester
from drift_detection.drift_testers.chi_drift_tester import ChiDriftTester
from drift_detection.drift_test_set import DriftTestSet
from drift_detection.drift_testers.ks_drift_tester import KsDriftTester
from drift_detection.drift_testers.mmd_drift_tester import MMDDriftTester


class DriftDetector:
    def __init__(self):
        """
        Creates an empty drift detector object
        """
        self.drift_test_set = None
        self.single_iteration_history = None
        self.history_df = None
        self.test_consecutive_fails = {}

    def add_default_testers(self, cont_features: List, int_features: List, cat_features: List):
        """
        Add all default unit_tests to drift detector:
        KS for continuous features,
        Chi2 for categorical,
        todo: int2 currently not tested

        :param dataset: Pandas dataframe
        :param cont_features: column names of continuous features
        :param int_features: column names of integer features
        :param cat_features: column names of categorical features
        :param p_val_threshold: p value threshold for all unit_tests
        """
        self.drift_test_set = DriftTestSet('drift_test_set')

        for feature in cont_features:
            test_name = 'ks_' + feature
            self.add_tester(KsDriftTester(test_name, feature, 0.005))
            # todo: add also int features?

        for feature in cat_features:
            test_name = 'chi_' + feature
            self.add_tester(ChiDriftTester(test_name, feature, 0.005))

        self.add_tester(MMDDriftTester('mmd', cont_features + int_features, 0.03))

        # PCA is crap, probably because the data is not scaled.
        # self.add_tester(PcaKsDriftTester('pca_ks', cont_features + int_features, 0.1))

    def add_tester(self, test: AbstractDriftTester, consecutive_fails=3):
        self.drift_test_set.add(test)
        self.test_consecutive_fails[test.test_name] = consecutive_fails

    def fit(self, data: pd.DataFrame):
        self._reset_history()
        self.drift_test_set.fit(data)
        _ = self.test_drift(data)

    def test_drift(self, data: pd.DataFrame):
        result = self.drift_test_set.test_drift(data)
        self._update_history(result)
        return self.history_df.iloc[-1].to_dict()

    def get_test_names(self):
        return self.drift_test_set.get_test_names()

    def _reset_history(self):
        self.single_iteration_history = []
        hist_columns = ['drift_detected', 'test_exceptions']
        test_names = self.drift_test_set.get_test_names()
        hist_columns = hist_columns + test_names + \
                       [test_name + '_cons_fails' for test_name in test_names] + \
                       [test_name + '_iter_fail' for test_name in test_names]
        self.history_df = pd.DataFrame(columns=hist_columns)

    def _update_history(self, result):
        self.single_iteration_history.append(result)
        df_dict = {'drift_detected': result['drift_found'],
                   'test_exceptions': result['data']['test_exceptions']}
        for test_name in self.drift_test_set.get_test_names():
            # single batch fail indication
            df_dict[test_name + '_iter_fail'] = True if test_name in df_dict['test_exceptions'] else False

            # consecutive fails counter for test
            test_prev_cons_fails = 0 if len(self.history_df) == 0 else self.history_df.iloc[-1][test_name + '_cons_fails']
            df_dict[test_name + '_cons_fails'] = test_prev_cons_fails + df_dict[test_name + '_iter_fail'] if df_dict[test_name + '_iter_fail'] else 0

            # bottom line for tester rule
            df_dict[test_name] = True if df_dict[test_name + '_cons_fails'] >= self.test_consecutive_fails[test_name] else False

        self.history_df = self.history_df.append(df_dict, ignore_index=True)
