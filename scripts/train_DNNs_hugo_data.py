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
from torch.utils.data import DataLoader
import numpy as np

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
        progress_bar_refresh_rate=500,
        callbacks=callbacks)
    trainer.fit(model)

    return model

def tune_mdl(mdl, num_param_samples=150, num_epochs=10, gpus_per_trial=1, config={}, with_args={}, data_dir=None):


    if data_dir is None:
        raise ValueError("Please provide a path to the data directory")

    additional_parameter_columns = list(config)

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=additional_parameter_columns,
        metric_columns=["loss", "mean_correlation", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_mdl,
            mdl=mdl,
            data_dir=data_dir,
            num_epochs=num_epochs,
            kwargs=with_args
        ),
            resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="mean_correlation",
        mode="max",
        config=config,
        num_samples=num_param_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="population_study_"+now,
        verbose=1)

    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis

def save_results(analysis, path, prefix=""):
    analysis.results_df.to_csv(os.path.join(path,prefix+"results_df.csv"))
    json.dump(analysis.best_config, open(os.path.join(path, prefix+"best_config.json"), "w"))
    
def tune_models(model_list=["detaillez", "eegnet", "linearmodel"], data_dir=None, results_path=None, num_param_samples=150, participants=range(13), prefix = ""):

    if (data_dir is None) or (results_path is None):
        raise ValueError("Please provide paths to the data file and the results directory!")

    kwargs = {
        "train_parts":range(9),
        "val_parts":range(9, 12),
        "test_parts":range(12,15),
        "train_participants":range(13),
        "test_participants":range(13),
        "mse_weight":0,
        "var_regulariser":0
    }

    common_config = {
        "weight_decay": tune.loguniform(1e-8, 1),
        "learning_rate": tune.loguniform(1e-8, 1),
        "batch_size": tune.choice([32, 64, 128, 256])
    }

    if "detaillez" in model_list:

        DeTaillez_config = {
            "num_hidden": tune.choice([1,2,3,4]),
            "dropout_rate": tune.uniform(0, 0.75),
            **common_config
        }
        dt_analysis = tune_mdl(DeTaillez, config=DeTaillez_config, with_args=kwargs, data_dir=data_dir, num_param_samples=num_param_samples)
        save_results(dt_analysis, os.path.join(results_path, "detaillez"), prefix=prefix)

    if "eegnet" in model_list:

        EEGNet_config = {
            "temporalFilters": tune.choice([4, 8, 16]),
            "spatialFilters": tune.choice([4,8,16]),
            "dropout_rate": tune.uniform(0, 0.75),
            **common_config
        }
        en_analysis = tune_mdl(EEGNet, config=EEGNet_config, with_args=kwargs, data_dir=data_dir, num_param_samples=num_param_samples)
        save_results(en_analysis, os.path.join(results_path, "eegnet"), prefix=prefix)

def population_study(model_handle=DeTaillez, model_string="detaillez", test_batch_size=125, results_path=None, data_dir=None):

    if (data_dir is None) or (results_path is None):
        raise ValueError("Please provide paths to the data file and the results directory!")

    all_participants = range(13)

    kwargs = {
        "train_parts":range(9),
        "val_parts":range(9, 12),
        "test_parts":range(12,15),
        "train_participants":all_participants,
        "test_participants":all_participants,
        "mse_weight":0,
        "var_regulariser":0
    }
    
    config = json.load(open(
        os.path.join(results_path, "{}/best_config.json".format(model_string))
    ))

    checkpoint_path = os.path.join(results_path, "{}/population.ckpt".format(model_string))
    if os.path.exists(checkpoint_path):
        model = model_handle.load_from_checkpoint(checkpoint_path=checkpoint_path,**config, **kwargs, data_dir=data_dir)
    else:
        model = train_mdl(config, model_handle,kwargs=kwargs, num_epochs=10, checkpoint_path=checkpoint_path, data_dir=data_dir)
    model.to('cuda')
    model.eval()

    results = {}

    for participant in kwargs["test_participants"]:
        results[participant] = {"correlations":[], "null_correlations":[]}
        dataset = datasets.HugoMapped(model.test_parts, model.data_dir, participant=participant, num_input=model.input_length, return_null=True)
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
        
    print(results)
    json.dump(results, open(os.path.join(results_path, "{}/population_results_{}.json".format(model_string, test_batch_size)), "w"))

def individuals_study(model_handle=DeTaillez, model_string="detaillez", test_batch_size=125, results_path=None, data_dir=None):

    if (data_dir is None) or (results_path is None):
        raise ValueError("Please provide paths to the data file and the results directory!")

    kwargs = {
        "train_parts":range(9),
        "val_parts":range(9, 12),
        "test_parts":range(12,15),
        "mse_weight":0,
        "var_regulariser":0
    }

    results = {}

    for participant in range(13):

        kwargs["train_participants"] = [participant]
        kwargs["test_participants"] = [participant]
        
        config = json.load(open(
            os.path.join(results_path, "{}/best_config.json".format(model_string))
        ))

        checkpoint_path = os.path.join(results_path, f"{model_string}/P{participant:02d}_individuals.ckpt")
        if os.path.exists(checkpoint_path):
            model = model_handle.load_from_checkpoint(checkpoint_path=checkpoint_path,**config, **kwargs, data_dir=data_dir)
        else:
            model = train_mdl(config, model_handle,kwargs=kwargs, num_epochs=5, checkpoint_path=checkpoint_path, data_dir=data_dir)
        model.to('cuda')
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
        
    json.dump(results, open(os.path.join(results_path, "{}/individuals_results_{}.json".format(model_string, test_batch_size)), "w"))

def cross_validation_study(model_handle=DeTaillez, model_string="detaillez", test_batch_size=125, results_path=None, data_dir=None):

    if (data_dir is None) or (results_path is None):
        raise ValueError("Please provide paths to the data file and the results directory!")

    kwargs = {
        "train_parts":range(9),
        "val_parts":range(9, 12),
        "test_parts":range(12,15),
        "mse_weight":0,
        "var_regulariser":0
    }

    results = {}

    for participant in range(13):

        kwargs["train_participants"] = [i for i in range(13) if i != participant]
        kwargs["test_participants"] = [participant]
        
        config = json.load(open(
            #os.path.join(results_path, f"{model_string}/P{participant:02d}_best_config.json")
            os.path.join(results_path, f"{model_string}/best_config.json")
        ))

        checkpoint_path = os.path.join(results_path, "{}/P{:02d}_cv.ckpt".format(model_string, participant))
        if os.path.exists(checkpoint_path):
            model = model_handle.load_from_checkpoint(checkpoint_path=checkpoint_path,**config, **kwargs, data_dir=data_dir)
        else:
            model = train_mdl(config, model_handle,kwargs=kwargs, num_epochs=10, checkpoint_path=checkpoint_path, data_dir=data_dir)
        model.to('cuda')
        model.eval()
        
        results[participant] = {"correlations":[], "null_correlations":[]}
        dataset = datasets.HugoMapped(model.test_parts, model.data_dir, participant=participant, num_input=model.input_length, return_null=True)
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
        
    print(results)
    json.dump(results, open(os.path.join(results_path, "{}/cv_results_{}.json".format(model_string, test_batch_size)), "w"))
    
def setup_results_dir(resultspath):
    if not os.path.exists(resultspath):
        os.mkdir(resultspath)


    for str in ["detaillez", "eegnet", "ridge", "generalisation"]:
        path = os.path.join(resultspath,str)
        if not os.path.exists(path):
            os.mkdir(path)

def main(seed=0, data_dir="/media/mdt20/Storage/data/hugo_data_processing/processed_data/280921B/data.h5", resultspath=f'results/0.5-12Hz'):
    torch.manual_seed(0)
    np.random.seed(0)

    num_param_samples=50
    num_cv_samples=20
    all_participants = range(13)

    # Set up results folder
    setup_results_dir(resultspath)
    
    # Tune hparams 
    models_to_tune = []
    for model_name in ["detaillez", "eegnet"]:
        if not os.path.exists(
            os.path.join(resultspath, "{}/best_config.json".format(model_name))
            ):
            models_to_tune.append(model_name)

    tune_models(models_to_tune, data_dir=data_dir, results_path=resultspath, num_param_samples=num_param_samples, participants=all_participants)

    # Train models
    models = [
        (DeTaillez, "detaillez"),
        (EEGNet, "eegnet")
    ]

    for handle, string in models:
        cross_validation_study(handle, string, test_batch_size=250, results_path=resultspath, data_dir=data_dir)
        population_study(handle, string, test_batch_size=250, results_path=resultspath, data_dir=data_dir)
        for test_batch_size in [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 20, 30, 40, 50] + [125*i for i in range(1, 11)]:
            individuals_study(handle, string, test_batch_size=test_batch_size, results_path=resultspath, data_dir=data_dir)

if __name__=="__main__":

    #main(seed=0, data_dir="/media/mdt20/Storage/data/hugo_data_processing/processed_data/280921E/data.h5", resultspath=f'results/2-8Hz')
    main(seed=0, data_dir="/media/mdt20/Storage/data/hugo_data_processing/processed_data/280921F/data.h5", resultspath=f'results/0.5-32Hz')
    #main(seed=0, data_dir="/media/mdt20/Storage/data/hugo_data_processing/processed_data/280921/data.h5", resultspath=f'results/0.5-16Hz')
    #main(seed=0, data_dir="/media/mdt20/Storage/data/hugo_data_processing/processed_data/280921B/data.h5", resultspath=f'results/0.5-12Hz')
    #main(seed=0, data_dir="/media/mdt20/Storage/data/hugo_data_processing/processed_data/280921C/data.h5", resultspath=f'results/0.5-8Hz')
