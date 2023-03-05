import unittest
import numpy as np
from clustering import ClusteringEvaluation

class TestClusteringEvaluation(unittest.TestCase):
    
    def setUp(self):
        self.X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12]])
    
    def test_preprocessing(self):
        clustering = ClusteringEvaluation(self.X)
        clustering.preprocessing()
        self.assertAlmostEqual(clustering.X.mean(), 0, places=7)
        self.assertAlmostEqual(clustering.X.std(), 1, places=7)
        
    def test_kmeans(self):
        clustering = ClusteringEvaluation(self.X)
        clustering.preprocessing()
        labels, metrics = clustering.kmeans(k=2)
        self.assertEqual(labels.shape[0], self.X.shape[0])
        self.assertEqual(set(labels), {0, 1})
        self.assertIn('Silhouette Score', metrics)
        self.assertIn('Calinski-Harabasz Index', metrics)
        
    def test_hierarchical(self):
        clustering = ClusteringEvaluation(self.X)
        clustering.preprocessing()
        labels, metrics = clustering.hierarchical(k=2)
        self.assertEqual(labels.shape[0], self.X.shape[0])
        self.assertEqual(set(labels), {0, 1})
        self.assertIn('Silhouette Score', metrics)
        self.assertIn('Calinski-Harabasz Index', metrics)
        
    def test_dbscan(self):
        clustering = ClusteringEvaluation(self.X)
        clustering.preprocessing()
        labels, metrics = clustering.dbscan(eps=0)
        self.assertEqual(labels.shape[0], self.X.shape[0])
        self.assertEqual(set(labels), {0, 1})
        self.assertIn('Silhouette Score', metrics)
        self.assertIn('Calinski-Harabasz Index', metrics)
        
if __name__ == '__main__':
    unittest.main()