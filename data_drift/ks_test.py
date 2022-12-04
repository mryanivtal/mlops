from typing import List
import pandas as pd
from scipy import stats

class KsTest:
    def __init__(self, a: pd.DataFrame, b: pd.DataFrame, p_threshold=0.005):
        self.p_threshold = p_threshold
        self.results = {}

        for col in a.columns:
            self.results[col] = stats.kstest(a[col], b[col])

    def exceptions_found(self) -> bool:
        for key in self.results.keys():
            if self.results[key][1] < self.p_threshold:
                return True

        return False

    def exceptions_list(self) -> List:
        ex_list = []
        for key in self.results.keys():
            if self.results[key][1] < self.p_threshold:
                ex_list.append(key)

        return ex_list

    def get_results(self):
        return self.results
