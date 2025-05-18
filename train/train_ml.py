import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn import metrics
import argparse
import datetime
import pickle

from features import FeatureExtractor
from models import BaselineClf, MLModel, RiemannMDM
from train.dataset import EEGDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/datasets/csv/",
        help="Path to the dataset directory.",
    )
    parser.add_argument(  # accept list of subject sub-01 sub-02 ...
        "--subjects",
        type=str,
        nargs="+",
        default=["sub-01"],
        help="List of subject (e.g., sub-01 sub-02 ...).",
    )
    parser.add_argument(
        "--n_times",
        type=int,
        default=1,
        help="Number of times to run the training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rfc",
        choices=["svc", "rfc", "baseline", "riemann", "threshold", "line", "average"],
        help="Which classifier to train",
    )
    return parser.parse_args()


def evaluate(model_dir, X_test, y_test):
    """Evaluate the model on the test set."""

    # load the model
    print(f"Loading model from {model_dir}...")
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    # get model predictions
    y_pred = model.predict(X_test)

    # get all relevant metrics for classification
    accuracy = metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    classification_report = metrics.classification_report(y_test, y_pred)

    # save the results
    results_file = os.path.join(model_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix))
        f.write("\nClassification Report:\n")
        f.write(classification_report)


def optimize(X_train, y_train, param_list, model_name):
    """hyper parameter optimization using grid search.

    Args:
        X_train (np.ndarray): training data
        y_train (np.ndarray): training labels
        param_list (dict): dictionary of parameters to optimize

    Returns:
        dict: best parameters found during optimization
    """
    print("Modelname in optimize: ", model_name)

    # create the model
    model = MLModel(model_name=model_name)

    # create the parameter grid
    param_grid = param_list
    # param_grid = {"model__" + k: v for k, v in param_list.items()}
    # param_grid = {"model__" + k: v for k, v in [next(iter(param_list.items()))]}  # skip hyperparameter optimization by using just the first given params

    # perform grid search
    scoring = ["accuracy", "balanced_accuracy", "f1_micro", "f1_macro"]
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring=scoring, refit="f1_macro", n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train, y_train)

    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

    return grid_search.best_params_


def train_model(X_train, y_train, hpo_parameters, model_name):

    # hyperparameter optimization
    best_params = optimize(X_train, y_train, hpo_parameters, model_name)

    # load best hyper-parameters
    # print("Modelname in train_model: ", model_name)
    model = MLModel(
        model_name=model_name,
        #feature_kwargs=best_params.get("feature_kwargs", {}), 
        **best_params
    )

    # fit final model
    model.fit(X_train, y_train)
    return model


def main():
    # parse command line arguments
    args = parse_args()
    print("Command line arguments: ", args)

    # create the dataset
    ds = EEGDataset(args.data_path, args.subjects, test_split=0.2)
    X_train, y_train, X_test, y_test = ds.get_dataset(
        time_window=0.5, overlap=0.50, balance=1.0
    )
    print("dataset classes: ", np.unique(y_train, return_counts=True))

    # train & optimize the model

    #uppers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0]
    
    uppers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    lowers = [-x for x in uppers]

    # hpo_params = {
    #     "upper_threshold": uppers + [x * 10.0 for x in uppers] + [x * 100.0 for x in uppers],
    #     "lower_threshold": lowers + [x * 10.0 for x in lowers] + [x * 100.0 for x in lowers],
    #     "n_exceeding_threshold": [1],
    # }
    hpo_params = {
        "upper_threshold": [0.0],
        "lower_threshold": [0.0],
        "n_exceeding_threshold": [1],
    }
    # hpo_params = {
    #     "upper_threshold": [1.0, 10.0, 100.0],
    #     "lower_threshold": [-1.0, -10.0, -100.0],
    #     "n_exceeding_threshold": [1, 2, 3],
    # }
    # hpo_params = {
    #     "upper_threshold": [0.6, 0.8, 6.0, 8.0, 60.0, 80.0, 600.0, 800.0],
    #     "lower_threshold": [-0.6, -0.8, -0.6, -0.8, -60.0, -80.0, -600.0, -800.0],
    #     "n_exceeding_threshold": [0, 1, 2, 3, 4, 6, 8, 10, 20, 40, 80, 130, 200],
        # "feature_kwargs__pca_components": [0.90, 0.80, 0.60],
        # "model_kwargs__kernel": ["linear", "rbf"],
        # "model_kwargs__C": [0.1, 1.0, 10.0, 100.0],
    # }
    # hpo_params = {
    #     "upper_threshold": [1.0, 10.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
    #     "lower_threshold": [-1.0, -10.0, -50.0, -100.0, -150.0, -200.0, -300.0, -400.0, -500.0, -600.0, -700.0, -800.0, -900.0, -1000.0],
    #     "n_exceeding_threshold": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 80, 130, 200],
    # }
    # hpo_params = {
    #     "feature_kwargs__pca_components": [0.90, 0.80, 0.60],
    #     "model_kwargs__kernel": ["linear", "rbf"],
    #     "model_kwargs__C": [0.1, 1.0, 10.0, 100.0],
    # }
    # hpo_params = {
    #     # Covariance estimation
    #     "feature_kwargs__estimator": ['scm', 'lwf', 'oas'],
    #     "feature_kwargs__alpha":     [0.0, 0.1, 0.2],   # for covariances_X, if used
    
    #     # MDM classifier
    #     "model_kwargs__metric":  ['riemann', 'logeuclid', 'euclid'],
    #     "model_kwargs__n_jobs":   [1, -1],
        
    #     # (Optional) Geodesic‑filter variant
    #     "model_kwargs__tsupdate": [True, False],
    # }
    model = train_model(X_train, y_train, hpo_parameters=hpo_params, model_name=args.model)

    # save results
    class_name = model.model.__class__.__name__
    ckpt_dir = "checkpoints/train_ml/"
    model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id += f"_{class_name}"
    model_dir = os.path.join(ckpt_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # save the model
    print(f"Saving model to {model_dir}...")
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # evaluate the model on the test set
    evaluate(model_dir, X_test, y_test)


if __name__ == "__main__":
    main()
