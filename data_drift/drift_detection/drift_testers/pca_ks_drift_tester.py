from abc import ABC
from typing import List, Dict
from scipy import stats
import pandas as pd
from sklearn.decomposition import PCA

from drift_detection.drift_testers.abstract_drift_tester import AbstractDriftTester


class PcaKsDriftTester(AbstractDriftTester, ABC):
    def __init__(self, test_name: str, col_names: List[str], p_threshold: float):
        self.p_threshold = p_threshold
        self.col_names = col_names
        self.test_name = test_name

        self.ref_data = None
        self.pca = None
        self.ref_data_pca = None
        self.is_fit = False

    def fit(self, ref_data: pd.DataFrame):
        self.ref_data = ref_data
        self.pca = PCA(n_components=1)
        self.ref_data_pca = self.pca.fit_transform(ref_data[self.col_names]).squeeze()
        self.is_fit = True

    def test_drift(self, data: pd.DataFrame) -> Dict:
        if not self.is_fit:
            raise Exception(f'Drift tester {self.test_name} was not fit')

        pca_x = self.pca.transform(data[self.col_names]).squeeze()
        results = {}
        results['test_name'] = self.test_name
        results['data'] = stats.kstest(pca_x, self.ref_data_pca)
        results['drift_found'] = results['data'][1] < self.p_threshold
        return results
