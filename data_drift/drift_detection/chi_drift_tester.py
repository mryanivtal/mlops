from abc import ABC
from typing import List, Dict

from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd

from data_helper import change_int_values
from drift_detection.abstract_drift_tester import AbstractDriftTester


class ChiDriftTester(AbstractDriftTester, ABC):
    def __init__(self, test_name: str, ref_data: pd.DataFrame, col_name: str, p_threshold: float):
        self.ref_data = ref_data
        self.p_threshold = p_threshold
        self.col_name = col_name
        self.test_name = test_name

    def test_drift(self, x: np.array) -> Dict:
        results = {}
        results['test_name'] = self.test_name
        results['data'] = chi2_contingency([x[self.col_name].value_counts(), self.ref_data[self.col_name].value_counts()])
        results['drift_found'] = results['data'][1] < self.p_threshold
        return results



if __name__ == '__main__':
    # Test function
    aa = pd.DataFrame(np.ones([100, 3]), columns=['a', 'b', 'c'])
    bb = pd.DataFrame(np.ones([100, 3]), columns=['a', 'b', 'c'])

    aa = change_int_values(aa, 'a', 1, 0, 0.5)
    bb = change_int_values(bb, 'a', 1, 0, 0.5)

    aa = change_int_values(aa, 'b', 1, 0, 0.2)
    bb = change_int_values(bb, 'b', 1, 0, 0.6)

    test_same = ChiDriftTester('tester_same', aa, 'a', 0.005)
    test_diff = ChiDriftTester('tester_different', aa, 'b', 0.005)

    res_same = test_same.test_drift(bb)
    res_diff = test_diff.test_drift(bb)

    print(res_same)
    print(res_diff)



