from abc import ABC
from typing import List, Dict
import pandas as pd
import torch
from drift_detection.drift_testers.abstract_drift_tester import AbstractDriftTester
import numpy as np

from drift_detection.drift_testers.kl_divergence_estimate import KLdivergence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KLDDriftTester(AbstractDriftTester, ABC):
    def __init__(self, test_name: str, col_names: List[str], dist_threshold: float = -1):
        self.ref_data = None
        # self.temp_data_storage = None
        self.ref_data_size = 0
        self.col_names = col_names
        self.test_name = test_name
        self.is_fit = False

        if dist_threshold == -1:
            self.threshold_autotune = True
            self.dist_threshold = None
        else:
            self.threshold_autotune = False
            self.dist_threshold = dist_threshold

    def fit(self, ref_data: pd.DataFrame):
        self.ref_data = ref_data
        self.temp_data_storage = ref_data.copy()
        self.ref_data_size = len(ref_data)

        if self.threshold_autotune:
            self.dist_threshold = self._tune_threshold(ref_data)

        self.is_fit = True

    def test_drift(self, data: pd.DataFrame) -> Dict:
        # need exact same data size between reference and current test
        self.temp_data_storage = self.temp_data_storage.append(data)
        self.temp_data_storage = self.temp_data_storage[len(self.temp_data_storage) - self.ref_data_size:]

        results = {}
        results['test_name'] = self.test_name
        results['data'] = KLdivergence(self.temp_data_storage[self.col_names], self.ref_data[self.col_names])
        results['drift_found'] = results['data'] > self.dist_threshold

        return results


    def _tune_threshold(self, data: pd.DataFrame, n_splits=5) -> float:
        """
        Runs [n_splits=5] iterations of split data -> calculate kl_div, then defines threshold as avg. + 3 sigmas.
        :param data: pandas
        :return: distance threshold
        """
        if len(data) < 50:
            raise Exception('Data is too small for threshold auto-tune)')

        dist_vals = []

        for split in range(n_splits):
            # shuffle, split exactly to two
            shuffled_data = data.copy().sample(frac=1)
            if len(shuffled_data) % 2 > 0:
                shuffled_data = shuffled_data.iloc[:-1, :]

            shuffled_data = shuffled_data.reset_index(drop=True)

            train_idx, test_idx = np.split(shuffled_data.index.to_numpy(), 2)

            dist = KLdivergence(shuffled_data.iloc[train_idx, :][self.col_names], shuffled_data.iloc[test_idx, :][self.col_names])
            dist_vals.append(dist)

        dist_vals = np.array(dist_vals)
        threshold = dist_vals.mean() + 3 * dist_vals.std()

        return threshold

