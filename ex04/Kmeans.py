import pandas as pd
import argparse
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


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
    X = np.array(new_df)
    #scaler = MinMaxScaler()
    #X = scaler.fit_transform(X)
    return new_df, X


class KmeansClustering:
    def __init__(self, max_iter=20, n_centroid=4):
        self.n_centroid = n_centroid
        self.max_iter = max_iter
        self.centroids = []
        self.model = KMeans(n_clusters=self.n_centroid,
                         max_iter=self.max_iter,
                         n_init=42,
                         init='random')

    def fit(self, X):
        self.model.fit(X)
        print('Cluster centers positions :\n', self.model.cluster_centers_)

    def predict(self, X):
        y_hat = self.model.predict(X)
        print('Predictions :\n', y_hat)
        return y_hat


    def plot_clusters_3D(self, X, y_hat):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_hat)
        ax.set_xlabel('Height')
        ax.set_ylabel('Weight')
        ax.set_zlabel('Bone density')
        ax.legend()
        plt.show()


if __name__ == "__main__":
    option = parse_arguments()
    if not option.filepath.endswith('.csv'):
        sys.exit('Invalid format for dataframe')
    try:
        df = pd.read_csv(option.filepath)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        sys.exit('File error')
    df, X = rework_df(df)
    model = KmeansClustering(option.max_iter, option.ncentroid)
    model.fit(X)
    y_hat = model.predict(X)
    model.plot_clusters_3D(X, y_hat)
