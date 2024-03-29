from abc import ABC
from drift_detection.drift_testers.abstract_drift_tester import AbstractDriftTester
import pandas as pd


class DriftTestSet(AbstractDriftTester, ABC):
    def __init__(self, test_set_name: str):
        self.test_set_name = test_set_name
        self.drift_testers = []

    def add(self, tester: AbstractDriftTester):
        self.drift_testers.append(tester)

    def fit(self, ref_data: pd.DataFrame):
        for tester in self.drift_testers:
            tester.fit(ref_data)

    def test_drift(self, data: pd.DataFrame):
        tester_results_list = []
        drift_test_exceptions = []
        drift_found = False

        for tester in self.drift_testers:
            tester_results = tester.test_drift(data)
            tester_results_list.append(tester_results)
            if tester_results['drift_found']:
                drift_found = True
                drift_test_exceptions.append(tester_results['test_name'])

        results = {
            'test_name': self.test_set_name,
            'drift_found': drift_found,
            'data': {
                'test_exceptions': drift_test_exceptions,
                'test_results_list': tester_results_list
            }
        }
        return results

    def get_test_names(self):
        return [test.test_name for test in self.drift_testers]

