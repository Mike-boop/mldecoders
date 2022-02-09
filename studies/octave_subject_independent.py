from pipeline.evaluation_functions import get_dnn_predictions, get_ground_truth, get_conditions
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

def setup_results_dir():
    Path(os.path.join(results_path, 'predictions', 'octave_subject_independent')).mkdir(parents=True, exist_ok=True)

def get_predictions(participant, condition='clean'):

    cnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_mdl_kwargs.json'), 'r'))
    fcnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_mdl_kwargs.json'), 'r')) 

    cnn = CNN(**cnn_best_mdl_kwargs)
    cnn.load_state_dict(torch.load(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn.ckpt')))
    fcnn = FCNN(**fcnn_best_mdl_kwargs)
    fcnn.load_state_dict(torch.load(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn.ckpt')))
    ridge = pickle.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'ridge.pk'), "rb"))

    test_dataset = OctaveMapped([1,2,3,4], data_file, participant=participant, num_input=cnn.input_length, condition=condition)
    test_loader = DataLoader(test_dataset, batch_size=1250, num_workers=16)

    fcnn_predictions = get_dnn_predictions(fcnn, test_loader)
    cnn_predictions = get_dnn_predictions(cnn, test_loader)
    ridge_predictions = ridge.predict(test_dataset.eeg.T).flatten()[:-cnn.input_length]
    attended_ground_truth = test_dataset.stim_a[:-cnn.input_length]
    unattended_ground_truth = test_dataset.stim_u[:-cnn.input_length]
    assert ridge_predictions.size == cnn_predictions.size == fcnn_predictions.size

    np.save(os.path.join(results_path, 'predictions', 'octave_subject_independent', f'ridge_predictions_{participant}_{condition}.npy'), ridge_predictions)
    np.save(os.path.join(results_path, 'predictions', 'octave_subject_independent', f'cnn_predictions_{participant}_{condition}.npy'), cnn_predictions)
    np.save(os.path.join(results_path, 'predictions', 'octave_subject_independent', f'fcnn_predictions_{participant}_{condition}.npy'), fcnn_predictions)
    np.save(os.path.join(results_path, 'predictions', 'octave_subject_independent', f'attended_ground_truth_{participant}_{condition}.npy'), attended_ground_truth)
    np.save(os.path.join(results_path, 'predictions', 'octave_subject_independent', f'unattended_ground_truth_{participant}_{condition}.npy'), unattended_ground_truth)

def main():

    setup_results_dir()
    for participant in ["YH00", "YH01", "YH02", "YH03",
                        "YH04", "YH06", "YH07", "YH09", "YH10",
                        "YH11", "YH14", "YH15", "YH16", 
                        "YH17", "YH18", "YH19", "YH20", "YH08"]:

        for condition in get_conditions(participant):
            get_predictions(participant, condition=condition)


if __name__ == '__main__':
    main()