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
import optuna

try:
     data_file = os.environ['MLDECODERS_HUGO_DATA_FILE']
     results_path = pathlib.Path(os.environ['MLDECODERS_RESULTS_DIR'])
except KeyError:
     print('please configure the environment!')
     exit()

print(f'running analysis with results directory {results_path} and data file {data_file}')

def setup_results_dir():

     Path(os.path.join(results_path, 'trained_models', 'hugo_subject_specific')).mkdir(parents=True, exist_ok=True)
     Path(os.path.join(results_path, 'predictions', 'hugo_subject_specific')).mkdir(parents=True, exist_ok=True)

def tune_lrs(participant, models=['cnn', 'fcnn']):

     if 'cnn' in models:

          cnn_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_mdl_kwargs.json'), 'r'))
          cnn_train_params = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_train_params.json'), 'r'))
          del cnn_train_params['lr']

          def cnn_objective(trial):

               lr =  trial.suggest_loguniform('lr', 1e-8, 1e-1)
               print('>',lr)
               accuracy, _ = train_dnn(data_file, participant, None, **cnn_train_params, lr=lr, epochs=20, early_stopping_patience=3,
                                        model_handle=CNN, **cnn_mdl_kwargs, optuna_trial=trial)
               return accuracy

          gridsampler = optuna.samplers.GridSampler({"lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]})

          cnn_pruner = optuna.pruners.MedianPruner(n_startup_trials=np.infty)
          cnn_study = optuna.create_study(
               direction="maximize",
               sampler=gridsampler,
               pruner=cnn_pruner   
          )

          cnn_study.optimize(cnn_objective, n_trials=5)
          cnn_summary = cnn_study.trials_dataframe()
          cnn_summary.to_csv(os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'cnn_lr_search_P{participant:02d}.csv'))

          cnn_train_params['lr'] = cnn_study.best_trial.params['lr']
          json.dump(cnn_train_params, open(os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'cnn_train_params_P{participant:02d}.json'), 'w'))

          pickle.dump(cnn_study, open(os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'cnn_lr_study_P{participant:02d}.pk'), 'wb'))

     if 'fcnn' in models:

          fcnn_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_mdl_kwargs.json'), 'r'))
          fcnn_train_params = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_train_params.json'), 'r'))
          del fcnn_train_params['lr']

          def fcnn_objective(trial):

               lr =  trial.suggest_loguniform('lr', 1e-8, 1e-1)
               print('>',lr)
               accuracy, _ = train_dnn(data_file, participant, None, **fcnn_train_params, lr=lr, epochs=20, early_stopping_patience=3,
                                        model_handle=FCNN, **fcnn_mdl_kwargs, optuna_trial=trial)
               return accuracy

          gridsampler = optuna.samplers.GridSampler({"lr": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]})
          fcnn_pruner = optuna.pruners.MedianPruner(n_startup_trials=np.infty)

          fcnn_study = optuna.create_study(
               direction="maximize",
               sampler=gridsampler,
               pruner=fcnn_pruner
          )
          fcnn_study.optimize(fcnn_objective, n_trials=5)
          fcnn_summary = fcnn_study.trials_dataframe()
          fcnn_summary.to_csv(os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'fcnn_lr_search_P{participant:02d}.csv'))

          fcnn_train_params['lr'] = fcnn_study.best_trial.params['lr']
          json.dump(fcnn_train_params, open(os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'fcnn_train_params_P{participant:02d}.json'), 'w'))

          pickle.dump(fcnn_study, open(os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'cnn_lr_study_P{participant:02d}.pk'), 'wb'))

def train_models(participant, models=['cnn', 'fcnn', 'ridge']):

     if 'cnn' in models:

          cnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_mdl_kwargs.json'), 'r')) 
          cnn_train_params = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'cnn_train_params_P{participant:02d}.json'), 'r'))  
          
          _, cnn_ckpt = train_dnn(data_file, participant, None, **cnn_best_mdl_kwargs, **cnn_train_params, epochs=30, model_handle=CNN, early_stopping_patience=5)
          torch.save(cnn_ckpt, os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'cnn_P{participant:02d}.ckpt'))


     if 'fcnn' in models:
          
          fcnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_mdl_kwargs.json'), 'r'))
          fcnn_train_params = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'fcnn_train_params_P{participant:02d}.json'), 'r'))  
          
          _, fcnn_ckpt = train_dnn(data_file, participant, None, **fcnn_best_mdl_kwargs, **fcnn_train_params, epochs=30, model_handle=FCNN, early_stopping_patience=5)
          torch.save(fcnn_ckpt, os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'fcnn_P{participant:02d}.ckpt'))

     if 'ridge' in models:

          train_ridge(data_file, participant, os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'ridge_P{participant:02d}.pk'), 0, 50, range(9), range(9,12)) 


def get_predictions(participant):

     cnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_mdl_kwargs.json'), 'r'))
     fcnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_mdl_kwargs.json'), 'r')) 

     cnn = CNN(**cnn_best_mdl_kwargs)
     cnn.load_state_dict(torch.load(os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'cnn_P{participant:02d}.ckpt')))
     fcnn = FCNN(**fcnn_best_mdl_kwargs)
     fcnn.load_state_dict(torch.load(os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'fcnn_P{participant:02d}.ckpt')))
     ridge = pickle.load(open(os.path.join(results_path, 'trained_models', 'hugo_subject_specific', f'ridge_P{participant:02d}.pk'), "rb"))

     test_dataset = HugoMapped(range(12,15), data_file, participant=participant, num_input=cnn.input_length)
     test_loader = DataLoader(test_dataset, batch_size=1250, num_workers=16)

     fcnn_predictions = get_dnn_predictions(fcnn, test_loader)
     cnn_predictions = get_dnn_predictions(cnn, test_loader)
     ridge_predictions = ridge.predict(test_dataset.eeg.T).flatten()[:-cnn.input_length]
     assert ridge_predictions.size == cnn_predictions.size == fcnn_predictions.size

     np.save(os.path.join(results_path, 'predictions', 'hugo_subject_specific', f'ridge_predictions_P{participant:02d}.npy'), ridge_predictions)
     np.save(os.path.join(results_path, 'predictions', 'hugo_subject_specific', f'cnn_predictions_P{participant:02d}.npy'), cnn_predictions)
     np.save(os.path.join(results_path, 'predictions', 'hugo_subject_specific', f'fcnn_predictions_P{participant:02d}.npy'), fcnn_predictions)


def main():

     setup_results_dir()
     for participant in range(13):
          tune_lrs(participant)
          train_models(participant)
          get_predictions(participant)

     ground_truth = get_ground_truth(data_file, range(12,15), source='hugo')
     np.save(os.path.join(results_path, 'predictions', 'hugo_subject_specific', 'ground_truth.npy'), ground_truth)

if __name__ == '__main__':
     main()