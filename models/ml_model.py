import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from features import FeatureExtractor

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """ Classifier that uses an upper and a lower threshold to classify the data. """
    def __init__(self, upper_threshold, lower_threshold, n_exceeding_threshold=1):
        """ Initialize the classifier with the upper and lower thresholds. """
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.n_exceeding_threshold = n_exceeding_threshold
    
    def fit(self, X, y):
        """ Fit the classifier to the data. """
        # No fitting required for threshold classifier
        return self

    def predict(self, X):
        """ Predict the class labels for the data. """
        # Check if the input is a 2D array
        assert X.ndim == 3, f"Input data must be a 3D array, but was {X.ndim}D."
        
        # Apply the thresholds to classify the data
        # A sample is classified as 1 iff the average over all channels is at one time above the upper threshold and at another time below the lower threshold, otherwise 0
        # In general what is described here as channel can be an arbitrary feature
        X_averaged_over_features = np.mean(X, axis=1)

        predictions = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            if np.sum(X_averaged_over_features[i] > self.upper_threshold) >= self.n_exceeding_threshold and np.sum(X_averaged_over_features[i] < self.lower_threshold) >= self.n_exceeding_threshold:
                predictions[i] = 1
            else:
                predictions[i] = 0
        
        return predictions
    

def model_selection(model_name):
    """ Select the model based on the model name. """
    if model_name == 'svc':
        from sklearn.svm import SVC
        return SVC
    
    elif model_name == 'rfc':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier

    elif model_name == 'riemann':
        from models import RiemannMDM
        return RiemannMDM

    elif model_name == 'threshold':
        return ThresholdClassifier
    
    elif model_name == 'baseline':
        from models import BaselineClf
        return BaselineClf
    
    else:
        raise ValueError(f"Model {model_name} not found.")


class MLModel(BaseEstimator, ClassifierMixin):

    """ Class to combine the full model pipeline """

    def __init__(self, model_name="threshold", feature_kwargs=None, upper_threshold=None, lower_threshold=None, n_exceeding_threshold=None):
        """ Initialize the model with the feature extractor and classifier. """
        self.feature_extractor = None
        self.model = None
        self.model_name = model_name
        self.model_cls = model_selection(model_name)
        self.feature_kwargs = feature_kwargs
        self.model_kwargs = {
            "upper_threshold": upper_threshold,
            "lower_threshold": lower_threshold,
            "n_exceeding_threshold": n_exceeding_threshold
        }
    
    def get_params(self, deep=True):
        return self.model_kwargs
        # return {
        #     "upper_threshold": self.model_kwargs.get("upper_threshold", None),
        #     "lower_threshold": self.model_kwargs.get("lower_threshold", None),
        #     "n_exceeding_threshold": self.model_kwargs.get("n_exceeding_threshold", None),
        #     # 'model_name': self.model_name,
        #     #'feature_kwargs': self.feature_kwargs,
        #     #'model_kwargs': self.model_kwargs
        # }

    def set_params(self, **parameters):
        self.model_kwargs = parameters

        # print(f"After setting {parameters}: ", self.model_kwargs, self.feature_kwargs)
        return self

    def fit(self, X, y):
        # print(f"Fitting model {self.model_name} ({self.model_cls.__name__}) with feature kwargs {self.feature_kwargs} and model kwargs {self.model_kwargs}")
        feature_kwargs = self.feature_kwargs or {}
        model_kwargs = self.model_kwargs or {}
        self.feature_extractor = FeatureExtractor(**feature_kwargs)
        self.model = self.model_cls(**model_kwargs)

        # print("Shapes before feature extractor: ", X.shape, y.shape)
        X = self.feature_extractor.fit_transform(X)
        # print("Shapes after feature extraction: ", X.shape, y.shape)
        self.model.fit(X, y)

    def predict(self, X):
        X = self.feature_extractor.transform(X)
        return self.model.predict(X)
    