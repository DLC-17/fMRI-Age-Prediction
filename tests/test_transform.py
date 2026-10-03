import unittest
import numpy as np
from src.transform import apply_pca, apply_smote

class TestTransform(unittest.TestCase):
    def test_apply_pca(self):
        # Create dummy data
        X_train = np.random.rand(100, 50)
        X_test = np.random.rand(20, 50)
        
        n_components = 10
        X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test, n_components=n_components)
        
        self.assertEqual(X_train_pca.shape, (100, n_components))
        self.assertEqual(X_test_pca.shape, (20, n_components))
        self.assertEqual(pca.n_components, n_components)

    def test_apply_pca_default_40(self):
        X_train = np.random.rand(80, 50)
        X_train_pca, _, pca = apply_pca(X_train, None)
        self.assertEqual(X_train_pca.shape, (80, 40))
        self.assertEqual(pca.n_components, 40)

    def test_apply_smote(self):
        # Create dummy data with class imbalance
        X = np.random.rand(100, 10)
        y = np.array([0] * 90 + [1] * 10) # 90 class 0, 10 class 1
        
        X_resampled, y_resampled = apply_smote(X, y)
        
        # SMOTE should balance the classes
        self.assertEqual(len(y_resampled), 180)
        self.assertEqual(sum(y_resampled == 0), 90)
        self.assertEqual(sum(y_resampled == 1), 90)
        self.assertEqual(X_resampled.shape, (180, 10))

if __name__ == "__main__":
    unittest.main()
