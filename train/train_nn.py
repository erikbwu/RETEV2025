import json
import os

import numpy as np
import torch
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from braindecode.models import EEGNetv4, EEGITNet
import datetime
import argparse

from train.dataset import EEGDataset
from torch.utils.data import random_split


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
        "--n_epochs",
        type=int,
        default=100,
        help="Number of epochs to train the model.",
    )
    return parser.parse_args()


def batch_to_device(batch, device):
    """Move the batch to the specified device."""
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    return x, y


def test(model, loader, loss_fn, device):
    model.eval()
    loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch_to_device(batch, device)
            y_pred = model(x)
            loss +=  loss_fn(y_pred, y).item()

            _, predicted = torch.max(y_pred, 1)

            # Collect predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())



    loss /= len(loader)
    accuracy = metrics.accuracy_score(all_targets, all_predictions)
    print(f"Test loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy



def train(model, loader, loss_fn, optimizer, device):
    model.train()
    for batch in loader:
        x, y = batch_to_device(batch, device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()


def main_train_loop(model, n_epochs, train_loader, test_loader, loss_fn, optimizer, device):
    best_model = None
    best_loss = float("inf")
    for epoch in range(n_epochs):
        train(model, train_loader, loss_fn, optimizer, device)
        loss, acc = test(model, test_loader, loss_fn, device)
        if loss < best_loss:
            best_loss = loss
            best_model = model

    print(f"Best BCE: {best_loss:.4f}")
    return best_model


def evaluate(model, test_loader, loss_fn, device, model_dir):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch_to_device(batch, device)
            y_pred = model(x)
            test_loss += loss_fn(y_pred, y).item()

            # Get predicted class
            _, predicted = torch.max(y_pred, 1)

            # Collect predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Calculate average loss
    test_loss /= len(test_loader)

    # Calculate metrics using sklearn
    accuracy = metrics.accuracy_score(all_targets, all_predictions)
    confusion_matrix = metrics.confusion_matrix(all_targets, all_predictions)
    classification_report = metrics.classification_report(all_targets, all_predictions)

    # Print summary
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the results
    results_file = os.path.join(model_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix))
        f.write("\nClassification Report:\n")
        f.write(classification_report)

    return test_loss, accuracy




def main(batch_size=8):

    # parse command line arguments
    args = parse_args()

    # create the dataset
    ds = EEGDataset(args.data_path, args.subjects, test_split=0.2)
    train_ds, test_ds = ds.get_torch_dataset(
        time_window=0.75, overlap=0.50, balance=1.0
    )

    print(len(train_ds.y))
    # split train_ds into train and val sets
    val_ratio = 0.2
    train_len = int((1 - val_ratio) * len(train_ds))
    val_len = len(train_ds) - train_len
    train_ds, val_ds = random_split(train_ds, [train_len, val_len])




    # create the dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    x,y =next(iter(train_loader))
    print(x.shape)

    # train on raw data
    x = train_ds[0][0]
    n_channels, n_times = x.shape
    print(f"Sample shape: {x.shape}")
    print(f"Number of time samples (n_times): {n_times}")

    # create the model

    model_kwargs = {
        "n_chans": n_channels,
        "n_outputs": 2,
        "n_times": n_times,
        "kernel_length": 64,
    }
    model = EEGNetv4(**model_kwargs)

    #model_kwargs = {
    #    "n_chans": n_channels,
    #    "n_outputs": 2,
    #    "n_times": n_times,
    #}
    #model = EEGITNet(**model_kwargs)


    # define the loss function and optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).double()

    # save results
    class_name = model.__class__.__name__
    ckpt_dir = "checkpoints/train_nn/"
    model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id += f"_{class_name}"
    model_dir = os.path.join(ckpt_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)
    # train the model
    model = main_train_loop(model, args.n_epochs, train_loader, val_loader, loss_fn, optimizer, device)

    # save the model
    print(f"Saving model to {model_dir}...")
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
    
    # save model kwargs
    model_kwargs_path = os.path.join(model_dir, "model_kwargs.json")
    with open(model_kwargs_path, "w") as f:
        json.dump(model_kwargs, f)

    # evaluate the model
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
    )
    evaluate(model, test_loader, loss_fn, device, model_dir)



if __name__ == "__main__":
   main()


