import warnings
from sklearn.datasets import load_boston
import pandas as pd

from data_helper import add_gaussian_noise_to_features

warnings.filterwarnings(action='ignore', category=FutureWarning)


class BostonDS:
    def __init__(self):
        self._load_boston_data()

        # Create dataset 5x size with minor noise
        noise_add_features = ['NOX', 'CRIM', 'DIS', 'AGE']
        x_synthetic, y_synthetic = self._get_5x_dataset_with_noise(self.x, self.y, noise_add_features, noise_mu=0,
                                                                   noise_rate=0.01)
        self.x = x_synthetic
        self.y = y_synthetic

    def _load_boston_data(self):
        boston = load_boston()
        df = pd.DataFrame(boston.data)

        df.columns = boston.feature_names
        df['PRICE'] = boston.target

        x = df.drop(['PRICE'], axis=1)
        y = df['PRICE']

        self.x = x.convert_dtypes()
        self.y = y.convert_dtypes()

        self.cat_features = []
        self.int_features = ['CHAS', 'RAD', 'TAX']
        self.cont_features = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'PTRATIO', 'B', 'LSTAT']

    # Create 5x dataset with added gaussian noise
    def _get_5x_dataset_with_noise(self, x, y, noise_features, noise_mu=0, noise_rate=0.05):
        # need to also add noise to class features and possibly to target

        x_synthetic = x.append(x).append(x).append(x).append(x).reset_index(drop=True).copy()
        y_synthetic = y.append(y).append(y).append(y).append(y).reset_index(drop=True).copy()

        noise_add_feature_std = x_synthetic[noise_features].std()

        for feature in noise_features:
            x_synthetic = add_gaussian_noise_to_features(x_synthetic, [feature], noise_mu,
                                                         noise_rate * noise_add_feature_std[feature])

        return x_synthetic, y_synthetic


if __name__ == '__main__':
    boston = BostonDS()
    print(boston.x.shape, boston.y.shape)
    print(boston.cat_features, boston.int_features, boston.cont_features)
