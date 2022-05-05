from pipeline.training_functions import train_dnn, train_ridge
from pipeline.evaluation_functions import get_dnn_predictions, get_ground_truth
from pipeline.datasets import HugoMapped
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
     data_file = os.environ['MLDECODERS_HUGO_DATA_FILE']
     results_path = pathlib.Path(os.environ['MLDECODERS_RESULTS_DIR'])
except KeyError:
     print('please configure the environment!')
     exit()

print(f'running analysis with results directory {results_path} and data file {data_file}')

def setup_results_dir():

     Path(os.path.join(results_path, 'trained_models', 'hugo_leave_one_out')).mkdir(parents=True, exist_ok=True)
     Path(os.path.join(results_path, 'predictions', 'hugo_leave_one_out')).mkdir(parents=True, exist_ok=True)


def train_models(participant, models = ['cnn', 'fcnn', 'ridge']):

     train_participants = [i for i in range(13) if i!=participant]

     if 'cnn' in models:

          cnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_mdl_kwargs.json'), 'r'))
          cnn_best_train_params = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_train_params.json'), 'r'))  
          
          _, cnn_ckpt = train_dnn(data_file, train_participants, None, **cnn_best_mdl_kwargs, **cnn_best_train_params, epochs=30, model_handle=CNN, early_stopping_patience=5)
          torch.save(cnn_ckpt, os.path.join(results_path, 'trained_models', 'hugo_leave_one_out', f'cnn_P{participant:02d}.ckpt'))
     
     if 'fcnn' in models:
          
          fcnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_mdl_kwargs.json'), 'r'))
          fcnn_best_train_params = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_train_params.json'), 'r'))  

          _, fcnn_ckpt = train_dnn(data_file, train_participants, None, **fcnn_best_mdl_kwargs, **fcnn_best_train_params, epochs=30, model_handle=FCNN, early_stopping_patience=5)
          torch.save(fcnn_ckpt, os.path.join(results_path, 'trained_models', 'hugo_leave_one_out', f'fcnn_P{participant:02d}.ckpt'))

     if 'ridge' in models:
          
          train_ridge(data_file, train_participants, os.path.join(results_path, 'trained_models', 'hugo_leave_one_out', f'ridge_P{participant:02d}.pk'), 0, 50, range(9), range(9,12)) 


def get_predictions(participant):

     cnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_mdl_kwargs.json'), 'r'))
     fcnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_mdl_kwargs.json'), 'r')) 

     cnn = CNN(**cnn_best_mdl_kwargs)
     cnn.load_state_dict(torch.load(os.path.join(results_path, 'trained_models', 'hugo_leave_one_out', f'cnn_P{participant:02d}.ckpt')))
     fcnn = FCNN(**fcnn_best_mdl_kwargs)
     fcnn.load_state_dict(torch.load(os.path.join(results_path, 'trained_models', 'hugo_leave_one_out', f'fcnn_P{participant:02d}.ckpt')))
     ridge = pickle.load(open(os.path.join(results_path, 'trained_models', 'hugo_leave_one_out', f'ridge_P{participant:02d}.pk'), "rb"))

     test_dataset = HugoMapped(range(12,15), data_file, participant=participant, num_input=cnn.input_length)
     test_loader = DataLoader(test_dataset, batch_size=1250, num_workers=16)

     fcnn_predictions = get_dnn_predictions(fcnn, test_loader)
     cnn_predictions = get_dnn_predictions(cnn, test_loader)
     ridge_predictions = ridge.predict(test_dataset.eeg[:, :-cnn.input_length].T).flatten()
     assert ridge_predictions.size == cnn_predictions.size == fcnn_predictions.size

     np.save(os.path.join(results_path, 'predictions', 'hugo_leave_one_out', f'ridge_predictions_P{participant:02d}.npy'), ridge_predictions)
     np.save(os.path.join(results_path, 'predictions', 'hugo_leave_one_out', f'cnn_predictions_P{participant:02d}.npy'), cnn_predictions)
     np.save(os.path.join(results_path, 'predictions', 'hugo_leave_one_out', f'fcnn_predictions_P{participant:02d}.npy'), fcnn_predictions)


def main():

     setup_results_dir()
     for participant in range(13):
          train_models(participant)
          get_predictions(participant)

     ground_truth = get_ground_truth(data_file, range(9), source='hugo')
     np.save(os.path.join(results_path, 'predictions', 'hugo_leave_one_out', 'ground_truth.npy'), ground_truth)

if __name__ == '__main__':
     main()