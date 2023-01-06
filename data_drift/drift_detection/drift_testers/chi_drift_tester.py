from abc import ABC
from typing import Dict
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
from drift_detection.drift_testers.abstract_drift_tester import AbstractDriftTester


class ChiDriftTester(AbstractDriftTester, ABC):
    def __init__(self, test_name: str, col_name: str, p_threshold: float):
        self.ref_data = None
        self.p_threshold = p_threshold
        self.col_name = col_name
        self.test_name = test_name
        self.is_fit = False

    def fit(self, ref_data: pd.DataFrame):
        self.ref_data = ref_data
        self.is_fit = True

    def test_drift(self, data: np.array) -> Dict:
        if not self.is_fit:
            raise Exception(f'Drift tester {self.test_name} was not fit')

        results = {}
        results['test_name'] = self.test_name

        # Make sure both histograms have the same keys
        x_hist = data[self.col_name].value_counts()
        ref_hist = self.ref_data[self.col_name].value_counts()
        unique_keys = set(x_hist.index.to_list() + ref_hist.index.to_list())

        for key in unique_keys:
            if not key in x_hist.index.to_list():
                x_hist = x_hist.append(pd.Series({key: 0}, index=[key]))
            if not key in ref_hist.index.to_list():
                ref_hist = ref_hist.append(pd.Series({key: 0}, index=[key]))

        results['data'] = chi2_contingency([x_hist, ref_hist])
        results['drift_found'] = results['data'][1] < self.p_threshold
        self.last_value = results['data'][1]

        return results

    def get_threshold(self):
        return self.p_threshold





