import os
from tabnanny import verbose
from pipeline.training_functions import train_ridge
from pipeline.datasets import HugoMapped
import pickle
from scipy.stats import pearsonr
import numpy as np

data_file = "data/hugo/0.5-8Hz/data.h5"

def train_models(participant):

     #train_ridge(data_file, participant, os.path.join('tests', 'tmp', f'ridge_P{participant:02d}.pk'), 0, 50, range(9), range(9,12))
     ridge = pickle.load(open(os.path.join('tests', 'tmp', f'ridge_P{participant:02d}.pk'), "rb"))

     test_dataset = HugoMapped(range(12,15), data_file, participant=participant, num_input=50)
     ridge_predictions = ridge.predict(test_dataset.eeg.T).flatten()
     ground_truth = test_dataset.stim
     print(pearsonr(ridge_predictions, ground_truth))
     print(np.mean(ridge.score_in_batches(test_dataset.eeg.T, ground_truth[:, None], batch_size=250)), '\n')

for participant in range(13):
    train_models(participant)