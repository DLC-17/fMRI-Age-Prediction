import pytest
import numpy as np
from src.transform import apply_pca, apply_smote

def test_apply_pca():
    # Create dummy data
    X_train = np.random.rand(100, 50)
    X_test = np.random.rand(20, 50)
    
    n_components = 10
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test, n_components=n_components)
    
    assert X_train_pca.shape == (100, n_components)
    assert X_test_pca.shape == (20, n_components)
    assert pca.n_components == n_components

def test_apply_smote():
    # Create dummy data with class imbalance
    X = np.random.rand(100, 10)
    y = np.array([0] * 90 + [1] * 10) # 90 class 0, 10 class 1
    
    X_resampled, y_resampled = apply_smote(X, y)
    
    # SMOTE should balance the classes
    assert len(y_resampled) == 180
    assert sum(y_resampled == 0) == 90
    assert sum(y_resampled == 1) == 90
    assert X_resampled.shape == (180, 10)
