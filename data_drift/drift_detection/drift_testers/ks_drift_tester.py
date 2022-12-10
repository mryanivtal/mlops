from abc import ABC
from typing import List, Dict
from scipy import stats
import numpy as np
import pandas as pd
from drift_detection.abstract_drift_tester import AbstractDriftTester


class KsDriftTester(AbstractDriftTester, ABC):
    def __init__(self, test_name: str, ref_data: pd.DataFrame, col_name: str, p_threshold: float):
        self.ref_data = ref_data
        self.p_threshold = p_threshold
        self.col_name = col_name
        self.test_name = test_name

    def test_drift(self, data: pd.DataFrame) -> Dict:
        results = {}
        results['test_name'] = self.test_name
        results['data'] = stats.kstest(data[self.col_name], self.ref_data[self.col_name])
        results['drift_found'] = results['data'][1] < self.p_threshold
        return results
