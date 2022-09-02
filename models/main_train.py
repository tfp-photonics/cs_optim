from model import *
from train import *
from objective import *
from losses import *
import torch.nn as nn

import os
import torch
from torch.utils.data import DataLoader

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-name", type=str, default="model", help="model name, also for saving"
    )
    parser.add_argument("-device", type=str, default="cpu")
    parser.add_argument("-n_shells", type=int, default=4, help="number of shells, 0-4")
    parser.add_argument("-hidden_dim", type=list, default=[111, 188, 200, 287, 117])
    parser.add_argument(
        "-nlaf",
        type=list,
        default=[
            nn.LeakyReLU(),
            nn.Tanhshrink(),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
        ],
    )
    parser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "-n_epochs", type=int, default=1000, help="number of training epochs"
    )
    parser.add_argument(
        "-batch_size", type=int, default=256, help="training batch size"
    )
    parser.add_argument(
        "-amsgrad", type=bool, default=False, help="enabels amsgrad for ADAM"
    )

    parser.add_argument("-step_size", type=int, default=100, help="step for scheduler")
    parser.add_argument(
        "-gamma", type=float, default=0.5, help="factor for lr scheduler"
    )

    parser.add_argument("--seed", type=int, default=42, help="generator seeds")
    parser.add_argument("-threads", type=int, default=2)

    args = parser.parse_args()

    device = args.device
    seed = args.seed

    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)

    n_points = 200
    w_min = 400
    w_max = 800
    n_params = (args.n_shells + 1) * 2

    ### specify network parameters
    hidden_dim = np.array(args.hidden_dim)  ## in params, first optuna results
    nlaf = np.array(args.nlaf)

    ### training hyperparameters
    lr = args.lr
    epochs = args.n_epochs
    batch_train = args.batch_size

    ### create model
    model = LinearNetwork(n_params, args.hidden_dim, n_points, args.nlaf)
    print("created model. number of trainable parameters: ", count_parameters(model))

    ### initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=args.amsgrad)

    ### initialize lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    ### define loss function
    loss_fn = MSEloss_fn

    ### load data
    datapath = "data/"
    data_name = "training_data.pt"

    dat = torch.load(datapath + data_name)
    train_dataset = dat["train_dataset"]
    val_dataset = dat["val_dataset"]
    wave_range = dat["wave_range"]

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_train)
    val_loader = DataLoader(dataset=val_dataset)

    ### saving path
    saving_path = "models/"

    ### train model
    net, losses, vallosses, optimizer, scheduler = train(
        model,
        args.n_epochs,
        loss_fn,
        optimizer,
        scheduler,
        train_set,
        val_set,
        name=args.name,
        path=saving_path,
        device=args.device,
    )

    ### simple training plot
    # import matplotlib.pyplot as plt

    # fig, axs = plt.subplots()
    # axs.plot(range(epochs), losses, label = 'train_loss', alpha = 0.5)
    # axs.plot(range(epochs), vallosses, label = 'val_loss', alpha = 0.5)
    # axs.set_xlabel('epochs')
    # axs.set_ylabel('loss')
    # axs.legend()
    # plt.show()
