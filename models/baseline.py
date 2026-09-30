import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def features(X):
    return np.column_stack([X.mean(1),X.std(1),X.min(1),X.max(1),np.median(X,1),np.sqrt(np.mean(X**2,1))])

def make_model(seed=42):
    return Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=seed))])
