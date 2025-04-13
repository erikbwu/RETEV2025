import os
import torch
from torch.utils.data import DataLoader
from braindecode.models import EEGNetv4


def split_dataset(dataset, val_size=0.2, test_size=0.2):
    pass


def test(model, loader, loss_fn, device):
    pass


def train(model, loader, loss_fn, optimizer, device):
    pass


def main_train_loop(model, train_loader, test_loader, loss_fn, optimizer, device):
    pass


def evaluate(model, loader, loss_fn, device):
    pass


def main(data_path, n_times, batch_size=32):

    # save results in a directory
    model_dir = "checkpoints/train_nn/"
    i = 0
    while os.path.exists(f"{model_dir}_{i:02d}"):
        i += 1
    model_dir = f"{model_dir}_{i:02d}"
    os.makedirs(model_dir, exist_ok=True)

    # load the dataset
    dataset = None  # TODO implement the dataset first
    train_ds, val_ds, test_ds = split_dataset(dataset)

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

    # create the model
    model = EEGNetv4(
        n_chans=8,
        n_classes=2,
        n_times=50,  # time window of 200ms - depends on your system
        # TODO feel free to modify the default parameters to your needs
    )

    # define the loss function and optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train the model
    main_train_loop(model_dir, model, train_loader, val_loader, loss_fn, optimizer, device)

    # evaluate the model
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
    )
    evaluate(model, test_loader, loss_fn, device)



if __name__ == "__main__":
   main()


