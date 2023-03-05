import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.decomposition import PCA
import umap
import networkx as nx
import community

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, dendrogram



class ClusteringEvaluation:
    """
    A class for evaluating different clustering algorithms on a dataset.

    Parameters:
    -----------
    X : ndarray
        The input data.
    max_clusters : int, optional (default=None)
        The maximum number of clusters to consider. If not specified, set to len(X)-1.
    dim_reduction : str, optional (default=None)
        The type of dimensionality reduction to apply to the input data. Currently only supports PCA.

    Attributes:
    -----------
    X : ndarray
        The input data.
    max_clusters : int
        The maximum number of clusters to consider.
    reduction : str
        The type of dimensionality reduction applied to the input data.
    dim_reduction : name of the dimensionaly reduction
        
     """
    def __init__(self, X, max_clusters=None, dim_reduction = None):
        self.X = X
        print(f'Input data shape: {self.X.shape}')
        self.reduction = dim_reduction
        if dim_reduction is not None:
            self.dimensionnality_reduction(type)
        if max_clusters is not None:
            self.max_clusters = max_clusters
        else:
            self.max_clusters = len(X)-1

        self.supported_models = ['kmeans','hierarchical','dbscan']
            
    def preprocessing(self):
        """
        Preprocesses the input data by normalizing the data.
        """
        #remove outliers

        #normalizing the data
        scaler = preprocessing.StandardScaler().fit(self.X)
        self.X = scaler.transform(self.X)

    def dimensionnality_reduction(self,type = 'PCA'):
        """
        Applies dimensionality reduction to the input data using the specified type.

        Parameters:
        -----------
        type : str, optional (default='PCA')
            The type of dimensionality reduction to apply. Currently only supports PCA.
        """

        print(f'Input data shape: {self.X.shape}')
        self.reduction = type
        if self.reduction == 'PCA':
            self.pca = PCA(n_components=0.95)
            self.X = self.pca.fit_transform(self.X)
            print(f'new data shape: {self.X.shape}')
        else:
            raise('wrong type of reduction')

    def kmeans(self, k=0):
        """
        Computes k-means clustering on the data.

        Parameters
        ----------
        k : int, optional
            Number of clusters. If not specified, the optimal number of clusters is determined using the elbow method.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Array of cluster labels for each data point.
        metrics : dict
            Dictionary containing the evaluation metrics for the clustering.

        """

        print('computing K-means')
        #the number of cluster was not specified:
        if k==0:
            errors = []
            # try k values from 1 to max_clusters
            for k in range(1, self.max_clusters+1):
                model = KMeans(n_clusters=k, n_init = 'auto', random_state=42)
                model.fit(self.X)
                errors.append( model.inertia_)
                
            deltas = np.diff(errors)
            diff_r = deltas[1:] / deltas[:-1]
            k = np.argmin(diff_r) + 2
            print('- optimal number of clusters:', k )
            
        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(self.X)
        metrics = compute_clustering_metrics(self.X, labels)

        return labels, metrics
    
    def hierarchical(self, k=0):
        """
        Computes hierarchical clustering on the data.

        Parameters
        ----------
        k : int, optional
            Number of clusters. If not specified, the optimal number of clusters is determined using the elbow method.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Array of cluster labels for each data point.
        metrics : dict
            Dictionary containing the evaluation metrics for the clustering.

        """

        print('computing hierarchical')
        #the number of cluster was not specified:
        if k==0:
            # calculate the linkage matrix
            Z = linkage(self.X, method='ward', metric='euclidean')
            # find the optimal number of clusters
            distances = Z[:, 2]
            distances_rev = distances[::-1]
            idxs = range(1, len(distances) + 1)
            if self.max_clusters == len(self.X)-1:
                elbow_point = np.argmax(np.diff(distances_rev)) + 1
            else:
                elbow_point = 0
                for i, d in enumerate(np.diff(distances_rev)):
                    if i >= self.max_clusters - 1:
                        break
                    elbow_point = i + 1
            k = idxs[elbow_point]
            print('- optimal number of clusters:', k )

        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(self.X)
        metrics = compute_clustering_metrics(self.X, labels)

        return labels, metrics
    
    def dbscan(self, eps=0, min_samples=1):
        """
        Computes DBSCAN clustering on the data.

        Parameters
        ----------
        eps : float, optional
            The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
            If not specified, the optimal value is determined using the mean distance to the k=2 nearest neighbors.
        min_samples : int, optional
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            This includes the point itself.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Array of cluster labels for each data point.
        metrics : dict
            Dictionary containing the evaluation metrics for the clustering.

        """

        print('computing dbscan')
        #the min distanceis not specified:
        if eps==0:
            #following https://link.springer.com/chapter/10.1007/978-3-319-44944-9_24

            # Compute the nearest neighbors and distances for all points
            nbrs = NearestNeighbors(n_neighbors=2).fit(self.X)
            distances, indices = nbrs.kneighbors(self.X)
            mean_distance = np.mean(distances[:,1])
            std_distance = np.std(distances[:,1])
            eps = mean_distance + 3 * std_distance
            print('eps distance:', eps)

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(self.X)
        print('- optimal number of clusters:', len(np.unique(labels)))
        metrics = compute_clustering_metrics(self.X, labels)
        return labels, metrics
    
    def consensus_clustering(self,labels_models):
        """
        Applies consensus clustering on the data using labels generated by multiple clustering algorithms.

        Parameters
        ----------
        labels_models : dict
            Dictionary containing the clustering labels generated by multiple algorithms.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Array of cluster labels for each data point.
        metrics : dict
            Dictionary containing the evaluation metrics for the clustering.

        """
        labels = np.zeros((len(self.X), 3))
        for i,model in enumerate(labels_models.keys()):
            labels[:, i] = labels_models[model]

        consensus = np.zeros((len(self.X), len(self.X)))
        n_algorithms = labels.shape[1]

        for i in range(len(self.X)):
            for j in range(i+1, len(self.X)):
                agreement = sum(labels[i, :] == labels[j, :])
                consensus[i, j] = agreement / n_algorithms
                consensus[j, i] = consensus[i, j]


        adj_matrix = consensus > 0.5

        G = nx.Graph(adj_matrix)
        partition = community.best_partition(G)
        # extract the clusters from the partition
        clusters = {}
        for node, cluster_id in partition.items():
            clusters[node] = cluster_id

        labels = np.array([clusters.get(i,np.nan) for i in range(len(self.X))])
        metrics = compute_clustering_metrics(self.X, labels)
        return labels, metrics
      
    def run_all_models(self):
        """
        Runs all the clustering algorithms on the data and computes the evaluation metrics for each algorithm.

        Returns
        -------
        df : pandas DataFrame
            Dataframe containing the evaluation metrics for all the clustering algorithms.

        """

        metrics_models = {}
        labels_models = {}
        labels_models['kmean'], metrics_models['kmean'] = self.kmeans()
        labels_models['hierarchical'], metrics_models['hierarchical'] = self.hierarchical()
        labels_models['dbscan'], metrics_models['dbscan'] = self.dbscan(eps=0)
        # perform a consensus model on previously computed labels
        labels_models['consensus'], metrics_models['consensus'] = self.consensus_clustering(labels_models)
        self.metrics_models = metrics_models
        self.labels_models = labels_models

        df = pd.DataFrame(metrics_models)

        return df

    def find_best_model(self):
        """
        Finds the best clustering algorithm based on the evaluation metrics.

        Returns
        -------
        best_model : str
            Name of the best clustering algorithm.
        df_t : pandas DataFrame
            Dataframe containing the evaluation metrics and scores for all the clustering algorithms.
        """

        df = pd.DataFrame(self.metrics_models)
        nb_models = len(df)
        ## uses a technic that uses the rank 2 as well
        df_t = df.T
        metrics = df_t.columns
        df_t['score'] = 0
        for metric in metrics:
            print(metric)
            if metric not in ['Davies-Bouldin Index','Number of clusters']:
                rank = df_t[metric].rank(ascending=False)
            else:
                rank = df_t[metric].rank(ascending=True)
            for i, row in df_t.iterrows():
                df_t.at[i, 'score'] += nb_models-(rank[i]-1)
              
        best_models = df_t.sort_values(by = 'score',ascending=False)['score']
        print("Best model(s):", best_models.idxmax())
        print("Models' score:", best_models.to_dict())

        return best_models.idxmax(),df_t

def compute_clustering_metrics(X, labels):
    """
    Computes clustering metrics for a given set of cluster labels.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.
    labels : array-like of shape (n_samples,)
        The cluster labels for each sample in X.

    Returns
    -------
    metrics : dict
        A dictionary containing the following clustering metrics:
        - Silhouette Score: A higher score indicates better clustering
        - Davies-Bouldin Index: A lower score indicates better clustering
        - Calinski-Harabasz Index: A higher score indicates better clustering
        - Number of clusters: The number of unique clusters in the labels.

    """
  
    metrics = {
            "Silhouette Score": silhouette_score(X, labels),
            "Davies-Bouldin Index": davies_bouldin_score(X, labels),
            "Calinski-Harabasz Index": calinski_harabasz_score(X, labels),
            "Number of clusters": len(np.unique(labels))
        }
    return metrics

class ClustersPlot():
    """
    Computes clustering metrics for a given set of cluster labels.

    Parameters
    ----------
    X: numpy.ndarray
        The array containing the data points.
    labels: numpy.ndarray
        The array containing the labels assigned to each point.
    y: numpy.ndarray
        The array containing the true cluster labels.
    df: pandas.DataFrame
        The DataFrame containing the data points and the cluster labels if provided.


    """
    def __init__(self, X, labels = None, y = None):

        self.labels = labels
        self.X = X
        self.y = y
        self.df = pd.DataFrame(self.X)
        if y is not None:
            self.df['clusters'] = y 

    def plot_features(self):
        '''         
        Creates a pairplot of the features with each pair of features plotted against each other, colored by the assigned cluster labels.
        '''
        sns.pairplot(self.df, hue='clusters')
        plt.show()

    def plot_clusters(self, labels, title = ''):
        '''
        Creates scatter plots of the clusters in 2D using either PCA or UMAP dimensionality reduction. The true cluster labels are plotted if provided.
        '''
        title = 'Plot for '+ title
        if self.X.shape[1]>2:
            self.pca = PCA(n_components=2)
            print(self.X.shape)
            X_plot = self.pca.fit_transform(self.X)
            print(X_plot.shape)
            self._plot_(X_plot, labels = labels, title= title + ', PCA')
            if self.y is not None:
                self._plot_( X_plot, labels = self.y, title= 'True clusters PCA' )

            # some parameters that highlight global strucutre (high nb of neighbors) and points close by (low min_dist)
            self.umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
            X_plot = self.umap.fit_transform(self.X)
            self._plot_(X_plot,labels = labels, title= title + ', UMAP')
            if self.y is not None:
                self._plot_( X_plot, labels = self.y, title= 'True clusters UMAP' )

        else:
            self._plot_( self.X, labels = labels,title= title )



    def _plot_(self, X_plot,labels, title):
        '''
        Creates a scatter plot of the clusters in 2D.
        '''

        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis')
        plt.xlabel('feature (reduction) 1')
        plt.ylabel('feature (reduction) 2')
        plt.title(title)
        plt.show()
