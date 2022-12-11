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

    def test_drift(self, x: np.array) -> Dict:
        if not self.is_fit:
            raise Exception(f'Drift tester {self.test_name} was not fit')

        results = {}
        results['test_name'] = self.test_name
        results['data'] = chi2_contingency([x[self.col_name].value_counts(), self.ref_data[self.col_name].value_counts()])
        results['drift_found'] = results['data'][1] < self.p_threshold
        return results




