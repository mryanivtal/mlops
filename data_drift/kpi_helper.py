from xgboost import XGBRegressor
import numpy as np
from sklearn import metrics
from typing import Dict


def calc_perf_kpis(x_test, y_test, y_pred) -> Dict:
    results = {
        'R_sq': metrics.r2_score(y_test, y_pred),
        'ajd_R_sq': 1 - (1 - metrics.r2_score(y_test, y_pred)) * (len(y_test) - 1) / (
                    len(y_test) - x_test.shape[1] - 1),
        'MAE': metrics.mean_absolute_error(y_test, y_pred),
        'MSE': metrics.mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    }

    return results
