import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler

class Cancer:
    scaler: StandardScaler | None = None

    def __init__(self, id, label, data_row):
        self.id = id
        self.label = label
        self.data = data_row  # pandas Series
        self.similarity = 999

    def set_similarity(self, other):
        if Cancer.scaler is None:
            raise RuntimeError("Cancer.scaler not fitted")

        feat_cols = [c for c in self.data.index
                     if c not in ('id', 'diagnosis', 'Unnamed: 32')]

        x = pd.DataFrame([self.data[feat_cols].values], columns=feat_cols)
        y = pd.DataFrame([other.data[feat_cols].values], columns=feat_cols)

        x_std = Cancer.scaler.transform(x).flatten()
        y_std = Cancer.scaler.transform(y).flatten()

        cos = float(np.dot(x_std, y_std) / (norm(x_std) * norm(y_std)))
        self.similarity = 0.5 * (cos + 1)


    def get_similarity(self):
        return self.similarity

    def get_label(self):
        return self.label

    def __str__(self) -> str:
        return f"id={self.id} label={self.label}  sim={self.similarity:.4f}"