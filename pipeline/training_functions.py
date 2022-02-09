from pipeline.dnn import CNN
from pipeline.ridge import Ridge
import torch
import os
from pipeline.datasets import HugoMapped, OctaveMapped
from torch.utils.data import DataLoader
from pipeline.helpers import correlation
from torch.optim import NAdam
import numpy as np
import pickle
import optuna
from collections.abc import Iterable
import operator
import functools

torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def train_dnn(data_dir,
              participants,
              checkpoint_path,
              dataset='hugo',
              model_handle=CNN,
              train_parts=range(9),
              val_parts=range(9,12),
              epochs=1,
              lr=3e-4,
              weight_decay=0.0001,
              batch_size=128,
              early_stopping_patience=torch.inf,
              optuna_trial=None,
              seed=0,
              **mdl_kwargs):

    '''
    if dataset is 'octave', expect train (val) parts to be a dictionary of {condition:trials} pairs
    '''

    if seed is not None:

        torch.manual_seed(seed)
        np.random.seed(seed)

    # configure model and optimizers

    mdl = model_handle(**mdl_kwargs)
    mdl.to(device)
    optimizer = NAdam(mdl.parameters(), lr=lr, weight_decay=weight_decay)

    # configure dataloaders

    if dataset=='hugo':

        assert isinstance(train_parts, Iterable) and type(train_parts) is not dict
        assert isinstance(val_parts, Iterable) and type(val_parts) is not dict

        train_dataset = HugoMapped(train_parts, data_dir, participant=participants, num_input=mdl.input_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = torch.randperm(len(train_dataset)), num_workers=16, pin_memory=True)

        val_dataset = HugoMapped(val_parts, data_dir, participant=participants, num_input=mdl.input_length)
        val_loader = DataLoader(val_dataset, batch_size=1250, sampler = torch.randperm(len(val_dataset)), num_workers=16, pin_memory=True)

    elif dataset=='octave':

        assert type(train_parts) == dict
        assert type(val_parts) == dict

        train_datasets = [OctaveMapped(train_parts[cond], data_dir, participant=participants, num_input=mdl.input_length, condition=cond) for cond in train_parts]
        train_dataset = functools.reduce(operator.add, train_datasets)    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = torch.randperm(len(train_dataset)), num_workers=16, pin_memory=True)

        val_datasets = [OctaveMapped(val_parts[cond], data_dir, participant=participants, num_input=mdl.input_length, condition=cond) for cond in val_parts]
        val_dataset = functools.reduce(operator.add, val_datasets)
        val_loader = DataLoader(val_dataset, batch_size=1250, sampler = torch.randperm(len(val_dataset)), num_workers=16, pin_memory=True)

    # early stopping parameters

    best_accuracy=0
    best_epoch=0
    best_state_dict={}

    # training loop

    for epoch in range(epochs):

        if epoch > best_epoch + early_stopping_patience:
            break

        mdl.train()

        for batch, (x, y) in enumerate(train_loader):

            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float)

            y_hat = mdl(x)
            loss = -correlation(y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch%10 == 0:
            #     print(loss.item())

        # validation loop

        mdl.eval()
        accuracies = []
        with torch.no_grad():
            for batch, (x, y) in enumerate(val_loader):
                x = x.to(device, dtype=torch.float)
                y = y.to(device, dtype=torch.float)

                y_hat = mdl(x)
                accuracies.append(correlation(y, y_hat))
        
        if checkpoint_path is not None:
            torch.save(
                mdl.state_dict(),
                os.path.join(checkpoint_path, f'epoch={epoch}_accuracy={mean_accuracy}.ckpt')
                )

        mean_accuracy = torch.mean(torch.hstack(accuracies)).item()
        print(mean_accuracy)
        
        if optuna_trial is not None:
            optuna_trial.report(mean_accuracy, epoch)
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_epoch = epoch
            best_state_dict = mdl.state_dict()

    return best_accuracy, best_state_dict

def train_ridge(data_dir,
                participants,
                save_path,
                start_lag,
                end_lag,
                train_parts,
                val_parts,
                alphas=np.logspace(-7,7, 15),
                dataset='hugo'):
    
    # fetch data

    mdl = Ridge(start_lag=start_lag, end_lag=end_lag, alpha=alphas)

    if dataset=='hugo':    

        train_dataset = HugoMapped(train_parts, data_dir, participant=participants, num_input=mdl.num_lags)
        val_dataset = HugoMapped(val_parts, data_dir, participant=participants, num_input=mdl.num_lags)

    elif dataset=='octave':

        assert type(train_parts) == dict
        assert type(val_parts) == dict

        train_datasets = [OctaveMapped(train_parts[cond], data_dir, participant=participants, num_input=mdl.num_lags, condition=cond) for cond in train_parts]
        train_dataset = functools.reduce(operator.add, train_datasets)    

        val_datasets = [OctaveMapped(val_parts[cond], data_dir, participant=participants, num_input=mdl.num_lags, condition=cond) for cond in val_parts]
        val_dataset = functools.reduce(operator.add, val_datasets)

    train_eeg, train_stim = train_dataset.eeg, train_dataset.stim
    val_eeg, val_stim = val_dataset.eeg, val_dataset.stim
    
    mdl.fit(train_eeg.T, train_stim[:, np.newaxis])
    mdl.model_selection(val_eeg.T, val_stim[:, np.newaxis])

    pickle.dump(mdl, open(save_path, "wb"))