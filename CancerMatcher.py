from cancer import Cancer
from statistics import mode
import pandas as pd
from sklearn.preprocessing import StandardScaler

class CancerMatcher:
    def __init__(self, file_path):
        self.data = pd.read_csv('Cancer_Data.csv')
        num_cols = [c for c in self.data.columns
                    if c not in ('id', 'diagnosis', 'Unnamed: 32')]

        Cancer.scaler = StandardScaler().fit(self.data[num_cols])


    def compute_similarity(self, cancer):
        for d in self.data:
            d.set_similarity(cancer)

    def most_similar(self):
        return sorted(self.data, key = lambda x: x.similarity, reverse = True)[0]

    def find_k_most_similar(self, k):
        return sorted(self.data, key = lambda x: x.similarity, reverse = True)[: k]

    def k_nearest_neighbors(self, k):
        return mode([x.label for x in self.find_k_most_similar(k)])

    def weighted_k_nearest_neighbors(self, k):
        k_neighbors = self.find_k_most_similar(k)
        weighted_votes = {}
        for rank, digit in enumerate(k_neighbors):
            weight = k - rank
            label = digit.get_label()
            weighted_votes[label] = weighted_votes.get(label, 0) + weight
        return max(weighted_votes, key=weighted_votes.get)

    def get_data(self):
        return self.data

