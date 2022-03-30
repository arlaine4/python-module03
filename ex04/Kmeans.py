import pandas as pd
import argparse
import sys
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', dest='filepath', default='solar_system_census.csv', type=str,
                        action='store')
    parser.add_argument('-nc', '--ncentroid', dest='ncentroid', default=4, type=int, action='store')
    parser.add_argument('-mi', '--max_iter', dest='max_iter', default=30, type=int, action='store')
    options = parser.parse_args()
    return options


def rework_df(dataframe):
    new_df = dataframe.drop('Unnamed: 0', axis=1)
    return new_df, np.array(new_df)


class KmeansClustering:
    def __init__(self, max_iter=20, n_centroid=4):
        self.n_centroid = n_centroid
        self.max_iter = max_iter
        self.centroids = []

    def fit(self, X):
        print(KMeans(n_clusters=self.n_centroid).fit(X))
        print()


if __name__ == "__main__":
    option = parse_arguments()
    if not option.filepath.endswith('.csv'):
        sys.exit('Invalid format for dataframe')
    try:
        df = pd.read_csv(option.filepath)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        sys.exit('File error')
    df, X = rework_df(df)
    k_mean = KmeansClustering(option.max_iter, option.ncentroid)
    k_mean.fit(X)
