import sklearn
from xgboost import XGBRegressor


class XgbModel:
    def __init__(self):
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.model = XGBRegressor()

    def fit(self, x, y):
        x = self.scaler.fit_transform(x)
        self.model.fit(x, y)

    def predict(self, x):
        x = self.scaler.transform(x)
        y = self.model.predict(x)
        return y
