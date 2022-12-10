from abc import ABC
from typing import List, Dict
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from drift_detection.abstract_drift_tester import AbstractDriftTester


class PcaKsDriftTester(AbstractDriftTester, ABC):
    def __init__(self, test_name: str, ref_data: pd.DataFrame, col_names: List[str], p_threshold: float):
        self.ref_data = ref_data
        self.p_threshold = p_threshold
        self.col_names = col_names
        self.test_name = test_name

        self.pca = PCA(n_components=1)
        self.ref_data_pca = self.pca.fit_transform(ref_data[self.col_names]).squeeze()

    def test_drift(self, data: pd.DataFrame) -> Dict:
        pca_x = self.pca.transform(data[self.col_names]).squeeze()

        results = {}
        results['test_name'] = self.test_name
        results['data'] = stats.kstest(pca_x, self.ref_data_pca)
        results['drift_found'] = results['data'][1] < self.p_threshold
        return results
