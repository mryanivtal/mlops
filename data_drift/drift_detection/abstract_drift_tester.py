import pandas as pd


class AbstractDriftTester:
    def test_drift(self, data: object):
        raise NotImplementedError

    def fit(self, ref_data: pd.DataFrame):
        raise NotImplementedError

