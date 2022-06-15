import optuna
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

     Path(os.path.join(results_path, 'trained_models', 'hugo_population')).mkdir(parents=True, exist_ok=True)
     Path(os.path.join(results_path, 'predictions', 'hugo_population')).mkdir(parents=True, exist_ok=True)
     

def tune_dnns(load=False, models=['cnn', 'fcnn']):

     if 'cnn' in models:

          def cnn_objective(trial):

               train_params = {
                    'lr': trial.suggest_categorical('tr_lr', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
                    'batch_size':trial.suggest_categorical('tr_batch_size', [64, 128, 256]),
                    'weight_decay': trial.suggest_categorical('tr_weight_decay', [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
                    }
               model_kwargs = {
                    'dropout_rate':trial.suggest_uniform('dropout_rate', 0, 0.4),
                    'F1':trial.suggest_categorical('F1', [2, 4, 8]),
                    'D':trial.suggest_categorical('D', [2, 4, 8])
               }
               model_kwargs['F2'] = model_kwargs['D']*model_kwargs['F1']
               accuracy, _ = train_dnn(data_file, range(13), None, **train_params, epochs=10, model_handle=CNN, **model_kwargs, early_stopping_patience=3, optuna_trial=trial)
               return accuracy

          if load and os.path.exists(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_study.pk')):
               cnn_study = pickle.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_study.pk'), 'rb'))
          else:
               cnn_pruner = optuna.pruners.MedianPruner(n_startup_trials=np.infty)
               cnn_study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.RandomSampler(seed=0),
                    pruner=cnn_pruner
               )

          cnn_study.optimize(cnn_objective, n_trials=80)
          cnn_summary = cnn_study.trials_dataframe()
          cnn_summary.to_csv(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_param_search.csv'))

          cnn_best_params = cnn_study.best_trial.params
          cnn_best_model_kwargs = {k: cnn_best_params[k] for k in cnn_best_params if not k.startswith('tr_')}
          cnn_best_model_kwargs['F2'] = cnn_best_model_kwargs['F1']*cnn_best_model_kwargs['D']
          cnn_best_train_params = {k.replace('tr_', ''): cnn_best_params[k] for k in cnn_best_params if k.startswith('tr_')}
          json.dump(cnn_best_model_kwargs, open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_mdl_kwargs.json'), 'w'))
          json.dump(cnn_best_train_params, open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_train_params.json'), 'w'))

          pickle.dump(cnn_study, open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_study.pk'), 'wb'))

     if 'fcnn' in models:

          def fcnn_objective(trial):

               train_params = {
                    'lr': trial.suggest_categorical('tr_lr', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
                    'batch_size':trial.suggest_categorical('tr_batch_size', [64, 128, 256]),
                    'weight_decay': trial.suggest_categorical('tr_weight_decay', [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
                    }
               model_kwargs = {
                    'num_hidden': trial.suggest_int('num_hidden', 1,4),
                    'dropout_rate': trial.suggest_uniform('dropout_rate', 0, 0.5)}
               accuracy, _ = train_dnn(data_file, range(13), None, **train_params, epochs=20, model_handle=FCNN, **model_kwargs, early_stopping_patience=3, optuna_trial=trial)
               return accuracy

          if load and os.path.exists(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_study.pk')):
               fcnn_study = pickle.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_study.pk'), 'rb'))
          else:
               fcnn_pruner = optuna.pruners.MedianPruner(n_startup_trials=np.infty)
               fcnn_study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.RandomSampler(seed=0),
                    pruner=fcnn_pruner
               )

          fcnn_study.optimize(fcnn_objective, n_trials=80)
          fcnn_summary = fcnn_study.trials_dataframe()
          fcnn_summary.to_csv(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_param_search.csv'))

          fcnn_best_params = fcnn_study.best_trial.params
          fcnn_best_model_kwargs = {k: fcnn_best_params[k] for k in fcnn_best_params if not k.startswith('tr_')}
          fcnn_best_train_params = {k.replace('tr_', ''): fcnn_best_params[k] for k in fcnn_best_params if k.startswith('tr_')}
          json.dump(fcnn_best_model_kwargs, open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_mdl_kwargs.json'), 'w'))
          json.dump(fcnn_best_train_params, open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_train_params.json'), 'w'))

          pickle.dump(fcnn_study, open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_study.pk'), 'wb'))


def train_models(models = ['cnn', 'fcnn', 'ridge']):

     if 'cnn' in models:

          cnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_mdl_kwargs.json'), 'r'))
          cnn_best_train_params = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_train_params.json'), 'r'))  
          
          _, cnn_ckpt = train_dnn(data_file, range(13), None, **cnn_best_mdl_kwargs, **cnn_best_train_params, epochs=30, model_handle=CNN, early_stopping_patience=5)
          torch.save(cnn_ckpt, os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn.ckpt'))

     if 'fcnn' in models:
          fcnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_mdl_kwargs.json'), 'r'))
          fcnn_best_train_params = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_train_params.json'), 'r'))  

          _, fcnn_ckpt = train_dnn(data_file, range(13), None, **fcnn_best_mdl_kwargs, **fcnn_best_train_params, epochs=30, model_handle=FCNN, early_stopping_patience=5)
          torch.save(fcnn_ckpt, os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn.ckpt'))

     if 'ridge' in models:
          train_ridge(data_file, range(13), os.path.join(results_path, 'trained_models', 'hugo_population', 'ridge.pk'), 0, 50, range(9), range(9,12))


def get_predictions(participant):

     cnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn_mdl_kwargs.json'), 'r'))
     fcnn_best_mdl_kwargs = json.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn_mdl_kwargs.json'), 'r')) 

     cnn = CNN(**cnn_best_mdl_kwargs)
     cnn.load_state_dict(torch.load(os.path.join(results_path, 'trained_models', 'hugo_population', 'cnn.ckpt')))
     fcnn = FCNN(**fcnn_best_mdl_kwargs)
     fcnn.load_state_dict(torch.load(os.path.join(results_path, 'trained_models', 'hugo_population', 'fcnn.ckpt')))
     ridge = pickle.load(open(os.path.join(results_path, 'trained_models', 'hugo_population', 'ridge.pk'), "rb"))

     test_dataset = HugoMapped(range(12,15), data_file, participant=participant, num_input=cnn.input_length)
     test_loader = DataLoader(test_dataset, batch_size=1250, num_workers=16)

     fcnn_predictions = get_dnn_predictions(fcnn, test_loader)
     cnn_predictions = get_dnn_predictions(cnn, test_loader)
     ridge_predictions = ridge.predict(test_dataset.eeg[:, :-cnn.input_length].T).flatten()
     assert ridge_predictions.size == cnn_predictions.size == fcnn_predictions.size

     np.save(os.path.join(results_path, 'predictions', 'hugo_population', f'ridge_predictions_P{participant:02d}.npy'), ridge_predictions)
     np.save(os.path.join(results_path, 'predictions', 'hugo_population', f'cnn_predictions_P{participant:02d}.npy'), cnn_predictions)
     np.save(os.path.join(results_path, 'predictions', 'hugo_population', f'fcnn_predictions_P{participant:02d}.npy'), fcnn_predictions)


def main():

     setup_results_dir()
     tune_dnns(load=False)
     train_models()
     for participant in range(13):
          get_predictions(participant)

     ground_truth = get_ground_truth(data_file, range(12,15), source='hugo')
     np.save(os.path.join(results_path, 'predictions', 'hugo_population', 'ground_truth.npy'), ground_truth)

if __name__ == '__main__':
     main()