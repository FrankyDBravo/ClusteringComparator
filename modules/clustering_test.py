import numpy as np
from sklearn.datasets import make_blobs
from clustering import *

# generate a random dataset
X, y = make_blobs(n_samples=300,n_features= 5, centers=4, random_state=0)

# create a ClusteringEvaluation object and evaluate KMeans
cEval = ClusteringEvaluation(X)
cEval.dimensionnality_reduction()
cEval.preprocessing()
cEval.run_all_models()
best_model,df = cEval.find_best_model()

# plot features and label clusters
cPlot = ClustersPlot(X,y=y)
cPlot.plot_features()

# plot the resulting clusters
cPlot.plot_clusters(cEval.labels_models[best_model], best_model)