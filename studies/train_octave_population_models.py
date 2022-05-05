from pipeline.training_functions import train_dnn, train_ridge
from pipeline.evaluation_functions import get_dnn_predictions, get_conditions
from pipeline.datasets import OctaveMapped
from torch.utils.data import DataLoader
from pipeline.dnn import CNN, FCNN
import pathlib
from pathlib import Path
import json
import os
import torch
import numpy as np
import pickle

try:
     data_file = os.environ['MLDECODERS_OCTAVE_DATA_FILE']
     results_path = pathlib.Path(os.environ['MLDECODERS_RESULTS_DIR'])
except KeyError:
    print('please configure the environment!')
    exit()

print(f'running analysis with results directory {results_path} and data file {data_file}')

train_parts={'lb':[1,2], 'clean':[1,2,3]}
val_parts={'clean':[4],'lb':[3,4]}

def setup_results_dir():

    Path(os.path.join(results_path, 'trained_models', 'octave_population')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(results_path, 'predictions', 'octave_population')).mkdir(parents=True, exist_ok=True)


def train_models(participants, models=['cnn', 'fcnn', 'ridge']):

    if 'cnn' in models:

        cnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_mdl_kwargs.json'), 'r'))
        cnn_train_params = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', f'cnn_train_params.json'), 'r'))  
    
        _, cnn_ckpt = train_dnn(data_file, participants, None, **cnn_best_mdl_kwargs, **cnn_train_params,
                            epochs=30, model_handle=CNN, dataset='octave', train_parts=train_parts,
                            val_parts=val_parts, early_stopping_patience=5)

        torch.save(cnn_ckpt, os.path.join(results_path, 'trained_models', 'octave_population', f'cnn.ckpt'))

    if 'fcnn' in models:

        fcnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_mdl_kwargs.json'), 'r'))
        fcnn_train_params = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', f'fcnn_train_params.json'), 'r'))  

        _, fcnn_ckpt = train_dnn(data_file, participants, None, **fcnn_best_mdl_kwargs, **fcnn_train_params,
                                epochs=30, model_handle=FCNN, dataset='octave', train_parts=train_parts,
                                val_parts=val_parts, early_stopping_patience=5)
        
        torch.save(fcnn_ckpt, os.path.join(results_path, 'trained_models', 'octave_population', f'fcnn.ckpt'))
    
    if 'ridge' in models:
        train_ridge(data_file, participants, os.path.join(results_path, 'trained_models', 'octave_population', f'ridge.pk'), 0, 50, train_parts, val_parts, dataset='octave') 


def get_predictions(participant, condition='clean'):

    cnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_mdl_kwargs.json'), 'r'))
    fcnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_mdl_kwargs.json'), 'r')) 

    cnn = CNN(**cnn_best_mdl_kwargs)
    cnn.load_state_dict(torch.load(os.path.join(results_path, 'trained_models', 'octave_population', 'cnn.ckpt')))
    fcnn = FCNN(**fcnn_best_mdl_kwargs)
    fcnn.load_state_dict(torch.load(os.path.join(results_path, 'trained_models', 'octave_population', 'fcnn.ckpt')))
    ridge = pickle.load(open(os.path.join(results_path, 'trained_models', 'octave_population', 'ridge.pk'), "rb"))

    test_dataset = OctaveMapped([1,2,3,4], data_file, participant=participant, num_input=cnn.input_length, condition=condition)
    test_loader = DataLoader(test_dataset, batch_size=1250, num_workers=16)

    fcnn_predictions = get_dnn_predictions(fcnn, test_loader)
    cnn_predictions = get_dnn_predictions(cnn, test_loader)
    ridge_predictions = ridge.predict(test_dataset.eeg.T).flatten()[:-cnn.input_length]
    attended_ground_truth = test_dataset.stim_a[:-cnn.input_length]
    unattended_ground_truth = test_dataset.stim_u[:-cnn.input_length]
    assert ridge_predictions.size == cnn_predictions.size == fcnn_predictions.size

    np.save(os.path.join(results_path, 'predictions', 'octave_population', f'ridge_predictions_{participant}_{condition}.npy'), ridge_predictions)
    np.save(os.path.join(results_path, 'predictions', 'octave_population', f'cnn_predictions_{participant}_{condition}.npy'), cnn_predictions)
    np.save(os.path.join(results_path, 'predictions', 'octave_population', f'fcnn_predictions_{participant}_{condition}.npy'), fcnn_predictions)
    np.save(os.path.join(results_path, 'predictions', 'octave_population', f'attended_ground_truth_{participant}_{condition}.npy'), attended_ground_truth)
    np.save(os.path.join(results_path, 'predictions', 'octave_population', f'unattended_ground_truth_{participant}_{condition}.npy'), unattended_ground_truth)

def main():


    setup_results_dir()
    participants = ["YH00", "YH01", "YH02", "YH03",
                    "YH04", "YH06", "YH07", "YH08",
                    "YH09", "YH10", "YH11", "YH14",
                    "YH15", "YH16", "YH17", "YH18",
                    "YH19", "YH20"]

    train_models(participants)

    for participant in participants:
        for condition in get_conditions(participant):
            get_predictions(participant, condition=condition)


if __name__ == '__main__':
    main()