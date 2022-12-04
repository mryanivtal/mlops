import warnings
from sklearn.datasets import load_boston
import pandas as pd

warnings.filterwarnings(action='ignore', category=FutureWarning)


class BostonDS:
    def __init__(self):
        self._load_boston_data()

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


if __name__ == '__main__':
    boston = BostonDS()
    print(boston.x.shape, boston.y.shape)
    print(boston.cat_features, boston.int_features, boston.cont_features)
