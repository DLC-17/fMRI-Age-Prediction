import numpy as np
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

def apply_pca(X_train, X_test, n_components=100):
    """
    Applies PCA to the training data and transforms both train and test data.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test) if X_test is not None else None
    
    return X_train_pca, X_test_pca, pca

def apply_smote(X, y, random_state=42):
    """
    Applies SMOTE to balance the dataset based on target classes.
    Note: SMOTE is typically used for classification. If y is continuous (like age),
    it needs to be binned into classes before applying SMOTE, or a variant for regression
    (like SMOGN) should be used. Here we assume y has been suitably preprocessed.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled
