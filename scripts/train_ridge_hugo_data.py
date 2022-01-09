import numpy as np
import h5py
import json
import os
import pickle

from pipeline.linear_models import Ridge

data_dir = "/media/mdt20/Storage/data/hugo_data_processing/processed_data/280921F/data.h5"
resultspath = 'results/0.5-32Hz/'

f = h5py.File(data_dir, "r")

participants = range(13)
train_story_parts = range(9)
val_story_parts = range(9,12)
test_story_parts = range(12,15)

alphas = np.logspace(-7,7, 15)

######
#Individuals study
######

def individuals_study(test_batch_size=125, resultspath=""):

    individuals_results = {}

    for participant in participants:

        X_train = np.hstack([f['eeg/P0{}/part{}/'.format(participant, i)][:] for i in train_story_parts]).T
        X_val = np.hstack([f['eeg/P0{}/part{}/'.format(participant, i)][:] for i in val_story_parts]).T
        X_test = np.hstack([f['eeg/P0{}/part{}/'.format(participant, i)][:] for i in test_story_parts]).T

        y_train = np.hstack([f['stim/part{}/'.format(i)][:] for i in train_story_parts])
        y_val = np.hstack([f['stim/part{}/'.format(i)][:] for i in val_story_parts])
        y_test = np.hstack([f['stim/part{}/'.format( i)][:] for i in test_story_parts])


        if os.path.exists(os.path.join(resultspath, "ridge/P{:02d}_individuals_mdl.pickle".format(participant))):
            mdl = pickle.load(open(os.path.join(resultspath, "ridge/P{:02d}_individuals_mdl.pickle".format(participant)), "rb"))

        else:
            
            mdl = Ridge(50, 0, alpha=alphas)
            mdl.fit(X_train, y_train[:, np.newaxis])
            mdl.model_selection(X_val, y_val[:, np.newaxis])

            pickle.dump(mdl, open(os.path.join(resultspath, "ridge/P{:02d}_individuals_mdl.pickle".format(participant)), "wb"))

            
        individuals_results[participant] = {}
        
        scores = mdl.score_in_batches(X_test, y_test[:, np.newaxis], batch_size=test_batch_size)
        null_scores = mdl.score_in_batches(X_test, np.roll(y_test, 128*20)[:, np.newaxis], batch_size=test_batch_size)
        
        individuals_results[participant]["correlations"] = scores.flatten().tolist()
        individuals_results[participant]["null_correlations"] = null_scores.flatten().tolist()

    json.dump(individuals_results, open(os.path.join(resultspath, "ridge/individuals_results_{}.json".format(test_batch_size)), "w"))

######
#CV study
######

def cross_validation_study(test_batch_size=125, resultspath=""):

    cv_results = {}

    for participant in participants:

        X_train = np.hstack([f['eeg/P0{}/part{}/'.format(j, i)][:] for i in train_story_parts for j in participants if j != participant]).T
        X_val = np.hstack([f['eeg/P0{}/part{}/'.format(j, i)][:] for i in val_story_parts for j in participants if j != participant]).T
        X_test = np.hstack([f['eeg/P0{}/part{}/'.format(participant, i)][:] for i in test_story_parts]).T

        y_train = np.hstack([f['stim/part{}/'.format(i)][:] for i in train_story_parts for j in participants if j != participant])
        y_val = np.hstack([f['stim/part{}/'.format(i)][:] for i in val_story_parts for j in participants if j != participant])
        y_test = np.hstack([f['stim/part{}/'.format(i)][:] for i in test_story_parts])

        if os.path.exists(os.path.join(resultspath, "ridge/P{:02d}_cv_mdl.pickle".format(participant))):
            mdl = pickle.load(open(os.path.join(resultspath, "ridge/P{:02d}_cv_mdl.pickle".format(participant)), "rb"))

        else:

            mdl = Ridge(50, 0, alpha=alphas)
            mdl.fit(X_train, y_train[:, np.newaxis])
            mdl.model_selection(X_val, y_val[:, np.newaxis])

            pickle.dump(mdl, open(os.path.join(resultspath, "ridge/P{:02d}_cv_mdl.pickle".format(participant)), "wb"))


        cv_results[participant] = {}

        scores = mdl.score_in_batches(X_test, y_test[:, np.newaxis], batch_size=test_batch_size)
        null_scores = mdl.score_in_batches(X_test, np.roll(y_test, 128*20)[:, np.newaxis], batch_size=test_batch_size)
        
        cv_results[participant]["correlations"] = scores.flatten().tolist()
        cv_results[participant]["null_correlations"] = null_scores.flatten().tolist()

    json.dump(cv_results, open(os.path.join(resultspath, "ridge/cv_results_{}.json".format(test_batch_size)), "w"))


######
#population study
######

def population_study(test_batch_size=125, resultspath=""):

    population_results = {}

    X_train = np.hstack([f['eeg/P0{}/part{}/'.format(j, i)][:] for i in train_story_parts for j in participants]).T
    X_val = np.hstack([f['eeg/P0{}/part{}/'.format(j, i)][:] for i in val_story_parts for j in participants]).T
    y_train = np.hstack([f['stim/part{}/'.format(i)][:] for i in train_story_parts for j in participants])
    y_val = np.hstack([f['stim/part{}/'.format(i)][:] for i in val_story_parts for j in participants])

    if os.path.exists(os.path.join(resultspath, "ridge/population_mdl.pickle")):
        mdl = pickle.load(open(os.path.join(resultspath, "ridge/population_mdl.pickle"), "rb"))

    else:
        mdl = Ridge(50, 0, alpha=alphas)
        mdl.fit(X_train, y_train[:, np.newaxis])
        mdl.model_selection(X_val, y_val[:, np.newaxis])
        pickle.dump(mdl, open(os.path.join(resultspath, "ridge/population_mdl.pickle"), "wb"))


    for participant in participants:

        X_test = np.hstack([f['eeg/P0{}/part{}/'.format(participant, i)] for i in test_story_parts]).T
        y_test = np.hstack([f['stim/part{}/'.format(i)][:] for i in test_story_parts])

        population_results[participant] = {}

        scores = mdl.score_in_batches(X_test, y_test[:, np.newaxis], batch_size=test_batch_size)
        null_scores = mdl.score_in_batches(X_test, np.roll(y_test, 128*20)[:, np.newaxis], batch_size=test_batch_size)
        
        population_results[participant]["correlations"] = scores.flatten().tolist()
        population_results[participant]["null_correlations"] = null_scores.flatten().tolist()

    json.dump(population_results, open(os.path.join(resultspath, "ridge/population_results_{}.json".format(test_batch_size)), "w"))

if __name__ == "__main__":

    population_study(test_batch_size=250, resultspath=resultspath)
    cross_validation_study(test_batch_size=250, resultspath=resultspath)

    for test_batch_size in [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 20, 30, 40, 50] + [125*i for i in range(1, 11)]:
        individuals_study(test_batch_size=test_batch_size, resultspath=resultspath)
    f.close()
