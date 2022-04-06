from datetime import datetime

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader


def train_torch_model_sgd(model,
                          ds_train,
                          ds_val,
                          loss_fn,
                          batch_size=16,
                          shuffle_data=False,
                          num_epochs=1000,
                          learning_rate=1e-2,
                          weight_decay=0,
                          print_every=10,
                          include_accs=False,
                          seed=None):
    """
    Train a pytorch classifier module with stochastic gradient descent.
    :param model: a `torch.nn.Module` to be trained to predict labels in the given datasets.
    :param ds_train: the training `torch.utils.data.Dataset`. This will be used to fit the model
    :param ds_val: the validation `torch.utils.data.Dataset`. THis will be used to evaluate the model
    :param loss_fn: a pytorch loss function to be optimized.
    :param batch_size: the size of the batches to be used for SGD optimization steps
    :param shuffle_data: if `True`, the data is shuffled for every epoch
    :param num_epochs: the number of epochs to train
    :param learning_rate: the magnitude of the optimization step
    :param weight_decay: L2 regularization lambda
    :param print_every: only print progress when `epoch_num % print_every == 0`
    :param include_accs: if `True`, calculates and returns classification accuracies.
    :param seed: a random seed for reproducibility.
    :return: a tuple (train_losses, val_losses, train_accs, val_accs) which are the training and validation losses
             and accuracies respectively.
    """
    # set seed if given
    if seed is not None:
        torch.manual_seed(seed)

    # initialize data-loaders
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle_data)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)  # no need to shuffle sets we do not train on

    # initialize an SGD optimizer for the given model's parameters
    optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # initialize training history aggregators
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # iterate through epochs
    for i in range(1, num_epochs + 1):
        # training epoch
        train_loss, train_acc = run_epoch(model, dl_train, loss_fn, optimizer, calc_accs=include_accs)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # validation epoch. no optimizer <==> no training
        val_loss, val_acc = run_epoch(model, dl_val, loss_fn, optimizer=None, calc_accs=include_accs)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # log epoch progress if necessary
        if i % print_every == 0:
            print_epoch(i, train_loss, train_acc, val_loss, val_acc, print_accs=include_accs)

    if include_accs:
        return train_losses, val_losses, train_accs, val_accs
    else:
        return train_losses, val_losses


def run_epoch(model, dl, loss_fn, optimizer, calc_accs=False):
    """
    performs a single epoch with a given dataloader and collects loss and accuracy
    :param model: a `torch.nn.Module` to be trained to predict labels in the given datasets.
    :param dl: a `torch.utils.data.DataLoader` that loads batches of the data on which to run.
    :param loss_fn: a pytorch loss function to be optimized.
    :param optimizer: a pytorch optimizer initialized on `model`'s parameters. if None, this is considered to be a
                      validation run, and no training step (or gradient calculation) is performed.
    :param calc_accs: if `True`, returns the accuracy as a second argument. otherwise, the returned accuracy is `None`
    :return: a tuple (epoch_loss, epoch_acc) which are the epoch loss and accuracy respectively
    """
    training_mode = optimizer is not None

    model.train(training_mode)
    total_loss = 0
    num_correct = 0
    for batch_obs, batch_actions in dl:
        with torch.set_grad_enabled(mode=training_mode):
            scores = model(batch_obs)
            loss = loss_fn(scores, batch_actions)
            preds = torch.argmax(scores, dim=-1)

        if training_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if calc_accs:
            num_correct += (batch_actions == preds).sum().item()

    return total_loss / len(dl), num_correct / len(dl.dataset) if calc_accs else None


def print_epoch(epoch_num, train_loss, train_acc, val_loss, val_acc, print_accs=False):
    """
    neatly prints the current epoch training progress.
    :param epoch_num: the current epoch number
    :param train_loss: the current epoch training loss
    :param train_acc: the current epoch training accuracy
    :param val_loss: the current epoch validation loss
    :param val_acc: the current epoch validation accuracy
    """
    print(f'epoch {epoch_num}:')
    print(f'avg training loss       = {train_loss}')
    print(f'avg validation loss     = {val_loss}')
    if print_accs:
        print(f'training acc            = {train_acc}')
        print(f'validation acc          = {val_acc}')
    print(f'timestamp: {datetime.now().strftime("%H:%M:%S.%f")}')
    print('====================================================')
    print()
