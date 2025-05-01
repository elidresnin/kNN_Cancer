from cancer import Cancer
from statistics import mode
import pandas as pd
from sklearn.preprocessing import StandardScaler

class CancerMatcher:

    def __init__(self, csv_path: str):
        data = pd.read_csv(csv_path)

        num_cols = [c for c in data.columns
                    if c not in ('id', 'diagnosis', 'Unnamed: 32')]

        Cancer.scaler = StandardScaler().fit(data[num_cols])

        self.cases: list[Cancer] = [
            Cancer(row["id"], row["diagnosis"], row)      # row is a Series
            for _, row in data.iterrows()
        ]

    def compute_similarity(self, cancer):
        for c in self.cases:
            c.set_similarity(cancer)

    def most_similar(self) -> Cancer:
        return max(self.cases, key=lambda x: x.similarity)

    def find_k_most_similar(self, k: int) -> list[Cancer]:
        return sorted(self.cases, key=lambda x: x.similarity, reverse=True)[:k]

    def k_nearest_neighbors(self, k: int):
        return mode([c.label for c in self.find_k_most_similar(k)])

    def weighted_k_nearest_neighbors(self, k: int):
        weighted = {}
        for rank, c in enumerate(self.find_k_most_similar(k)):
            weighted[c.label] = weighted.get(c.label, 0) + (k - rank)

        return max(weighted, key=weighted.get)
