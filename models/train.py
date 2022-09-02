import torch
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import time


## training step
def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step


###------ Count trainable parameters---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    net,
    n_epochs,
    loss_fn,
    optimizer,
    scheduler,
    train_set,
    val_set,
    name="model",
    path=".",
    device="cpu",
):
    train_step = make_train_step(net, loss_fn, optimizer)

    losses = []
    vallosses = []
    start = time.monotonic()
    print("start: ", stime.ctime())
    for i in range(n_epochs):
        lossmean = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = train_step(x_batch, y_batch)
            lossmean.append(loss)
        losses.append(sum(lossmean) / len(lossmean))

        vallossmean = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                model.eval()
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_hat = model(x_val)
                val_loss = loss_fn(y_val, y_hat).detach().numpy()
                vallossmean.append(val_loss)
        vallosses.append(sum(vallossmean) / len(vallossmean))

        scheduler.step()
        if epochs % (n_epochs // 100):
            reference = time.monotonic()
            print(
                "Training progress:" "\n  time: ",
                (reference - start),
                "\n     epoch#",
                i + 1,
                "\n     loss: ",
                np.around(losses[-1], 6),
                "\n     val_loss: ",
                np.around(val_losses[-1], 6),
                "\n     lr: ",
                scheduler._last_lr,
            )

    end = time.monotonic()
    print("training_finished in ", end - start, "s")

    print("save model")
    torch.save(
        {
            "epochs": n_epochs,
            "training_time": end - start,
            "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict": net.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "optimizer": repr(optimizer),
            "scheduler": repr(scheduler),
            "net": repr(net),
            "losses": losses,
            "val_losses": val_losses,
        },
        path + name + ".pt",
    )
    return net, losses, vallosses, optimizer, scheduler
