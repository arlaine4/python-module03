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
                self.plot_clusters_3D(X, self.predict(X), model.cluster_centers_)
        for centroid in save_centroids:
            self.plot_clusters_3D(X, centroid.predict(X), model=model)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def get_regions(self, y_hat, X, nb_regions=4):
        habs = [[] for i in range(self.n_centroid + 1)]
        for i in range(len(habs)):
            habs[i] = [X[j] for j in range(len(X)) if i == y_hat[j]]
        tab = [[1000, 0, 1000, 0, 1000, 0] for i in range(nb_regions)]
        for i in range(len(habs)):
            for j in range(len(habs[i])):
                if habs[i][j][0] < tab[i][0]:
                    tab[i][0] = habs[i][j][0]
                if habs[i][j][0] > tab[i][1]:
                    tab[i][1] = habs[i][j][0]
                if habs[i][j][1] < tab[i][2]:
                    tab[i][2] = habs[i][j][1]
                if habs[i][j][1] > tab[i][3]:
                    tab[i][3] = habs[i][j][1]
                if habs[i][j][2] < tab[i][4]:
                    tab[i][4] = habs[i][j][2]
                if habs[i][j][2] > tab[i][5]:
                    tab[i][5] = habs[i][j][2]
        print(f"\n\nClusters Regions :\n")
        for i, region in enumerate(tab):
            print(f'\033[37;1;4mRegion {i}\033[0m : {region}')


    def predict(self, X):
        y_hat = self.model.predict(X)
        print('\n\n\033[37;1;4mPredictions\033[0m :\n\n', y_hat)
        self.get_regions(y_hat, X)
        y_hat = list(y_hat)
        print(f"\n\n\033[37;1;4mCentroids\033[0m : \n")
        print(f'Centroid 0 : {y_hat.count(0)} / {len(y_hat)}'
              f' --> {round(y_hat.count(0) / len(y_hat) * 100, 2)}%')
        print(f'Centroid 1 : {y_hat.count(1)} / {len(y_hat)}'
              f' --> {round(y_hat.count(1) / len(y_hat) * 100, 2)}%')
        print(f'Centroid 2 : {y_hat.count(2)} / {len(y_hat)}'
              f' --> {round(y_hat.count(2) / len(y_hat) * 100, 2)}%')
        print(f'Centroid 3 : {y_hat.count(3)} / {len(y_hat)}'
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
    #model.fit_plot(X)
    model.plot_clusters_3D(X, y_hat)
