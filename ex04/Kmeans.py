import pandas as pd
import argparse
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
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
                            init='random',
                            algorithm='elkan')

    def fit(self, X):
        self.model.fit(X)
        self.centroids = self.model.cluster_centers_
        print('\n\033[37;1;4mCluster centers positions\033[0m :\n\n', self.model.cluster_centers_)


    def fit_plot(self, X):
        save_centroids = []
        n_init_grid_search = [10, 30]
        algo_grid_search = ['auto', 'elkan']
        for n_init in n_init_grid_search:
            for algo in algo_grid_search:
                model = KMeans(n_clusters=self.n_centroid,
                           max_iter=self.max_iter,
                           n_init=n_init,
                           init='random',
                           algorithm=algo)
                model.fit(X)
                save_centroids.append(model)
                #self.plot_clusters_3D(X, self.predict(X), model.cluster_centers_)
        for centroid in save_centroids:
            self.plot_clusters_3D(X, centroid.predict(X), model=model)


    def predict(self, X):
        y_hat = self.model.predict(X)
        print('\n\n\033[37;1;4mPredictions\033[0m :\n\n', y_hat)
        y_hat = list(y_hat)
        print(f'\n\033[37;1;4mCentroid 0\033[0m : {y_hat.count(0)} / {len(y_hat)}'
              f' --> {round(y_hat.count(0) / len(y_hat) * 100, 2)}%')
        print(f'\033[37;1;4mCentroid 1\033[0m : {y_hat.count(1)} / {len(y_hat)}'
              f' --> {round(y_hat.count(1) / len(y_hat) * 100, 2)}%')
        print(f'\033[37;1;4mCentroid 2\033[0m : {y_hat.count(2)} / {len(y_hat)}'
              f' --> {round(y_hat.count(2) / len(y_hat) * 100, 2)}%')
        print(f'\033[37;1;4mCentroid 3\033[0m : {y_hat.count(3)} / {len(y_hat)}'
              f' --> {round(y_hat.count(3) / len(y_hat) * 100, 2)}%', end='\n\n')
        return y_hat


    def plot_clusters_3D(self, X, y_hat, model=None):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_hat)
        if not model:
            ax.scatter(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], s=50, c='black')
        else:
            ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                       model.cluster_centers_[:, 2], s=50, c='black')
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
    model.fit_plot(X)
    model.plot_clusters_3D(X, y_hat)
