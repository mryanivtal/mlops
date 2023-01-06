from typing import Dict

import pandas as pd


class AbstractDriftTester:
    def __init__(self):
        self.test_name = None
        self.is_fit = False

    def test_drift(self, data: pd.DataFrame) -> Dict:
        raise NotImplementedError

    def fit(self, ref_data: pd.DataFrame):
        raise NotImplementedError



