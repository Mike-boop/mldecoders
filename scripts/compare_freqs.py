from pipeline.DNNs import DeTaillez, EEGNet
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray import tune
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import json
import datetime
import torch
import os
from pipeline import datasets, helpers
from pipeline.linear_models import Ridge
from torch.utils.data import DataLoader
import numpy as np
import h5py

now = datetime.datetime.now().strftime("%d%m%y_%H%M%S")

def train_mdl(config, mdl=None, kwargs={}, data_dir=None, num_epochs=5, checkpoint_path="checkpoint.ckpt", study="individuals"):
    if data_dir is None:
        raise ValueError("Please provide a path to the data file!")

    early_stop_callback = EarlyStopping(
    monitor='ptl/val_correlation',
    min_delta=0.00,
    patience=1,
    verbose=False,
    mode='max'
    )
        # note: this will raise a warning when called outside of a ray tune session. Still works as intended: saves checkpoints
    callbacks = [
        early_stop_callback,

        TuneReportCallback(
            {
                    "loss": "ptl/val_loss",
                    "mean_correlation": "ptl/val_correlation"
                },
                on="validation_end"),

        TuneReportCheckpointCallback(
                metrics={"loss":"ptl/val_loss", "mean_correlation":"ptl/val_correlation"},
                filename=checkpoint_path,
                on="validation_end")
            ]

    model = mdl(**config, **kwargs, data_dir=data_dir)
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        max_epochs=num_epochs,
        logger=TensorBoardLogger(
            save_dir=os.path.splitext(checkpoint_path)[0], name=study, version=now
        ),
        progress_bar_refresh_rate=10,
        callbacks=callbacks)
    trainer.fit(model)

    return model

def save_results(analysis, path, prefix=""):
    analysis.results_df.to_csv(os.path.join(path,prefix+"results_df.csv"))
    json.dump(analysis.best_config, open(os.path.join(path, prefix+"best_config.json"), "w"))

def individuals_study_DNNs(model_handle=DeTaillez, model_string="detaillez", test_batch_size=125, results_path=None, data_root_dir=None, data_folder=None):

    if (data_root_dir is None) or (results_path is None):
        raise ValueError("Please provide paths to the data file and the results directory!")

    kwargs = {}
    kwargs["train_parts"] = range(9)
    kwargs["val_parts"] = range(9, 12)
    kwargs["test_parts"] = range(12,15)
    kwargs["mse_weight"] = 0
    kwargs["batch_size"] = 128
    kwargs["var_regulariser"] = 0

    results = {}

    for participant in range(13):

        kwargs["train_participants"] = [participant]
        kwargs["test_participants"] = [participant]
        
        config = json.load(open(
            os.path.join(results_path, "{}/best_config.json".format(model_string))
        ))

        checkpoint_path = os.path.join(results_path, f"{model_string}/P{participant:02d}_individuals.ckpt")
        model = train_mdl(config, model_handle,kwargs=kwargs, num_epochs=5, checkpoint_path=checkpoint_path, data_dir=os.path.join(data_root_dir, data_folder, "data.h5"))

        model.eval()

        results[participant] = {"correlations":[], "null_correlations":[]}
        dataset = datasets.HugoMapped(model.test_parts, model.data_dir, participant=participant, num_input=model.input_length,channels=model.eeg_channels, return_null=True, normalise=False)
        loader = DataLoader(dataset, batch_size=test_batch_size, num_workers=16, drop_last=True)
        for x, y, y_null in loader:
            x = x.to(model.device, dtype=torch.float)
            y = y.to(model.device, dtype=torch.float)
            y_null = y_null.to(model.device, dtype=torch.float)


            y_hat = model(x)
            correlation = helpers.correlation(y_hat, y)
            null_correlation = helpers.correlation(y_hat, y_null)

            results[participant]["correlations"].append(correlation.item())
            results[participant]["null_correlations"].append(null_correlation.item())

        print(model_string)
        print("mean",np.mean(results[participant]["correlations"]), flush=True)
        print("std",np.std(results[participant]["correlations"]))
        
    json.dump(results, open(os.path.join(results_path, "{}/individuals_results_{}_{}.json".format(model_string, test_batch_size, data_folder)), "w"))

def individuals_study_ridge(test_batch_size=125, resultspath="", data_root_dir=None, data_folder=None):

    data_dir = os.path.join(data_root_dir, data_folder, 'data.h5')
    f = h5py.File(data_dir, "r")

    individuals_results = {}

    for participant in range(13):

        X_train = np.hstack([f['eeg/P0{}/part{}/'.format(participant, i)][:] for i in range(9)]).T
        X_val = np.hstack([f['eeg/P0{}/part{}/'.format(participant, i)][:] for i in range(9,12)]).T
        X_test = np.hstack([f['eeg/P0{}/part{}/'.format(participant, i)][:] for i in range(12,15)]).T

        y_train = np.hstack([f['stim/part{}/'.format(i)][:] for i in range(9)])
        y_val = np.hstack([f['stim/part{}/'.format(i)][:] for i in range(9,12)])
        y_test = np.hstack([f['stim/part{}/'.format( i)][:] for i in range(12,15)])

        alphas = np.logspace(-5,5, 11)


        mdl = Ridge(50, 0, alpha=alphas)
        mdl.fit(X_train, y_train[:, np.newaxis])
        mdl.model_selection(X_val, y_val[:, np.newaxis])
            
        individuals_results[participant] = {}
        
        scores = mdl.score_in_batches(X_test, y_test[:, np.newaxis], batch_size=test_batch_size)
        null_scores = mdl.score_in_batches(X_test, np.roll(y_test, 128*20)[:, np.newaxis], batch_size=test_batch_size)
        
        individuals_results[participant]["correlations"] = scores.flatten().tolist()
        individuals_results[participant]["null_correlations"] = null_scores.flatten().tolist()

    f.close()
    json.dump(individuals_results, open(os.path.join(resultspath, "ridge/individuals_results_{}_{}_ridge.json".format(test_batch_size, data_folder)), "w"))

def main(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_root_dir = "/media/mdt20/Storage/data/hugo_data_processing/processed_data"
    data_folders = ["280921B"]#.5-12Hz["280921E", "280921C", "280921", "280921F"]#2-8Hz, .5-8Hz, .5-16Hz, .5-32Hz
    resultspath = f'results/141121'
    
    # Train models
    models = [
        (DeTaillez, "detaillez"),
        (EEGNet, "eegnet")
    ]

    
    for data_folder in data_folders:
        for test_batch_size in [250]:
            for handle, string in models:
                individuals_study_DNNs(handle, string, test_batch_size=test_batch_size, results_path=resultspath, data_root_dir=data_root_dir, data_folder=data_folder)
            individuals_study_ridge(test_batch_size=test_batch_size, resultspath=resultspath, data_root_dir=data_root_dir, data_folder=data_folder)

if __name__=="__main__":
    main(seed=0)
    