from statistics import mean
from pipeline.dnn import CNN
from pipeline.ridge import Ridge
import torch
import os
from pipeline.datasets import HugoMapped, OctaveMapped
from torch.utils.data import DataLoader
from pipeline.helpers import correlation
from torch.optim import NAdam, AdamW
import numpy as np
import pickle
import optuna

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
              train_condition='clean',
              val_condition='clean',
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

    if seed is not None:

        torch.manual_seed(seed)
        np.random.seed(seed)

    # configure model and optimizers

    mdl = model_handle(**mdl_kwargs)
    mdl.to(device)
    optimizer = NAdam(mdl.parameters(), lr=lr, weight_decay=weight_decay)

    # configure dataloaders

    if dataset=='hugo':    

        train_dataset = HugoMapped(train_parts, data_dir, participant=participants, num_input=mdl.input_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = torch.randperm(len(train_dataset)), num_workers=16, pin_memory=True)

        val_dataset = HugoMapped(val_parts, data_dir, participant=participants, num_input=mdl.input_length)
        val_loader = DataLoader(val_dataset, batch_size=1250, sampler = torch.randperm(len(val_dataset)), num_workers=16, pin_memory=True)

    elif dataset=='octave':

        train_dataset = OctaveMapped(train_parts, data_dir, participant=participants, num_input=mdl.input_length, condition=train_condition)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = torch.randperm(len(train_dataset)), num_workers=16, pin_memory=True)

        val_dataset = OctaveMapped(val_parts, data_dir, participant=participants, num_input=mdl.input_length, condition=val_condition)
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
                dataset='hugo',
                train_condition='clean',
                val_condition='lb'):
    
    # fetch data

    mdl = Ridge(start_lag=start_lag, end_lag=end_lag, alpha=alphas)

    if dataset=='hugo':    

        train_dataset = HugoMapped(train_parts, data_dir, participant=participants, num_input=mdl.num_lags)
        val_dataset = HugoMapped(val_parts, data_dir, participant=participants, num_input=mdl.num_lags)

    elif dataset=='octave':

        train_dataset = OctaveMapped(train_parts, data_dir, participant=participants, num_input=mdl.num_lags, condition=train_condition)
        val_dataset = OctaveMapped(val_parts, data_dir, participant=participants, num_input=mdl.num_lags, condition=val_condition)

    train_eeg, train_stim = train_dataset.eeg, train_dataset.stim
    val_eeg, val_stim = val_dataset.eeg, val_dataset.stim
    
    mdl.fit(train_eeg.T, train_stim[:, np.newaxis])
    mdl.model_selection(val_eeg.T, val_stim[:, np.newaxis])

    pickle.dump(mdl, open(save_path, "wb"))