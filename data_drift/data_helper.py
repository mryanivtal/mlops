import numpy as np
import pandas as pd
from typing import List


# Add random gaussian noise to features in dataframe
def add_gaussian_noise_to_features(df: pd.DataFrame, feature_names: List, noise_mu: float,
                                   noise_sigma: float) -> pd.DataFrame:
    df = df.copy()
    df_len = len(df)

    for feature in feature_names:
        noise = np.random.randn(df_len) * noise_sigma + noise_mu
        df[feature] = df[feature] + noise

    return df


# Flip classes at randon in df features

def change_int_values(df: pd.DataFrame, feature_name, from_value, to_value, flip_rate) -> pd.DataFrame:
    df = df.copy()

    rows_with_from_values = df.index[df[feature_name] == from_value].tolist()
    amount_to_flip = int(flip_rate * len(rows_with_from_values))
    rows_to_flip = np.random.choice(rows_with_from_values, amount_to_flip, replace=False)
    df.loc[rows_to_flip, feature_name] = to_value
    return df


def sample_from_data(x, y, sample_size):
    sample_indexes = np.random.choice(range(len(x)), size=sample_size, replace=False)
    x_sample = x.iloc[sample_indexes, :].copy()
    y_sample = y.iloc[sample_indexes].copy()

    return x_sample, y_sample


if __name__ == '__main__':
    # Test function
    aa = pd.DataFrame(np.ones([100, 3]), columns=['a', 'b', 'c'])
    add_gaussian_noise_to_features(aa, ['a', 'c'], 2, 0.5)

    # Test function
    a = pd.DataFrame(np.random.randint(3, size=(100, 3)), columns=['a', 'b', 'c'])
    a['b'] = np.ones(len(a), dtype=int)
    print(a)
    a = change_int_values(a, 'b', 1, 99, 0.5)
    print(a)
