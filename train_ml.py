import os
from sklearn.svm import SVC
from sklearn.ensemble.RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split


def evaluate(model, X_test, y_test):
    pass


def main(data_path, n_times):

    # save results in a directory
    model_dir = "checkpoints/train_ml/"
    i = 0
    while os.path.exists(f"{model_dir}_{i:02d}"):
        i += 1
    model_dir = f"{model_dir}_{i:02d}"
    os.makedirs(model_dir, exist_ok=True)

    # load the dataset
    X, y = None  # TODO implement the dataset first
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    # load the model
    model = SVC(kernel='rbf', C=1.0, random_state=42)
    # model = RFC(n_estimators=100, random_state=42)

    # train the model
    model.fit(X_train, y_train)

    # evaluate the model
    evaluate(model, X_val, y_val)


if __name__ == "__main__":
   main()


