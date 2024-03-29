from abc import ABC
from typing import List, Dict
import pandas as pd
import torch
from drift_detection.drift_testers.abstract_drift_tester import AbstractDriftTester
import sklearn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MMDDriftTester(AbstractDriftTester, ABC):
    def __init__(self, test_name: str, col_names: List[str], dist_threshold: float = -1):
        self.ref_data = None
        self.temp_data_storage = None
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
        results['data'] = self._mmd_pd(self.temp_data_storage[self.col_names], self.ref_data[self.col_names])
        results['drift_found'] = results['data'].item() > self.dist_threshold

        self.last_value = results['data'].item()


        return results

    def _mmd(self, x, y, kernel='multiscale'):
        """Emprical maximum mean discrepancy. The lower the result
           the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
            kernel: kernel type such as "multiscale" or "rbf"
        """
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

        if kernel == "multiscale":

            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a ** 2 * (a ** 2 + dxx) ** -1
                YY += a ** 2 * (a ** 2 + dyy) ** -1
                XY += a ** 2 * (a ** 2 + dxy) ** -1

        if kernel == "rbf":

            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)

        return torch.mean(XX + YY - 2. * XY)

    def _mmd_pd(self, a, b, kernel='multiscale'):
        anp = a.to_numpy().astype('float64')
        bnp = b.to_numpy().astype('float64')
        t1 = torch.from_numpy(anp)
        t2 = torch.from_numpy(bnp)
        return self._mmd(t1, t2, kernel)

    def _tune_threshold(self, data: pd.DataFrame) -> float:
        if len(data) < 50:
            raise Exception('Data is too small for threshold auto-tune)')

        n_splits = 5
        dist_vals = []

        for split in range(n_splits):
            # shuffle, split exactly to two
            shuffled_data = data.copy().sample(frac=1)
            if len(shuffled_data) % 2 > 0:
                shuffled_data = shuffled_data.iloc[:-1, :]

            shuffled_data = shuffled_data.reset_index(drop=True)

            train_idx, test_idx = np.split(shuffled_data.index.to_numpy(), 2)

            dist = self._mmd_pd(shuffled_data.iloc[train_idx, :], shuffled_data.iloc[test_idx, :])
            dist_vals.append(dist)

        dist_vals = np.array(dist_vals)
        threshold = dist_vals.mean() + 3 * dist_vals.std()

        return threshold

    def get_threshold(self):
        return self.dist_threshold
