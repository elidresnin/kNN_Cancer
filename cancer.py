import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler

class Cancer:
    scaler: StandardScaler | None = None

    def __init__(self, label, data):
        self.label = label
        self.data = data
        self.similarity = 0.0

    def get_label(self):
        return self.label

    def set_similarity(self, other):
        cols = [c for c in self.data.index
                if c not in ('id', 'diagnosis', 'Unnamed: 32')]
        x = self.data[cols].to_numpy(dtype=float).reshape(1, -1)
        y = other.data[cols].to_numpy(dtype=float).reshape(1, -1)

        x_std = Cancer.scaler.transform(x)[0]
        y_std = Cancer.scaler.transform(y)[0]

        cos = float(np.dot(x_std, y_std) / (norm(x_std) * norm(y_std)))
        self.similarity = (cos + 1) * 0.5


    def get_similarity(self):
        return self.similarity

    def __str__(self):
        display = f"label = {self.label} \n"
        # display += f"similarity = {self.get_similarity()} \n|"
        if self.get_similarity() == 0:
            display += " " * 30 + "\n|"
        else:
            display += f"similarity = {self.get_similarity():.5f}"
            display += " " * 10 + "\n|"
        return display