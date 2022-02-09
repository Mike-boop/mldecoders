#!/bin/bash

source /home/mdt20/miniconda3/bin/activate env-dnns
cd /home/mdt20/Code/mldecoders

# export MLDECODERS_RESULTS_DIR="results/0.5-12Hz"
# export MLDECODERS_HUGO_DATA_FILE="data/hugo/0.5-12Hz/data.h5"
# export MLDECODERS_OCTAVE_DATA_FILE="data/octave/0.5-12Hz/data.h5"
# export MLDECODERS_EEG_UPBE="12"
# export MLDECODERS_EEG_LPBE="0.5"

# python data_preprocessing/process_hugo_data.py
# python data_preprocessing/process_octave_data.py
# python studies/train_hugo_subject_specific_models.py

# export MLDECODERS_RESULTS_DIR="results/2-8Hz"
# export MLDECODERS_HUGO_DATA_FILE="data/hugo/2-8Hz/data.h5"
# export MLDECODERS_OCTAVE_DATA_FILE="data/octave/2-8Hz/data.h5"
# export MLDECODERS_EEG_UPBE="8"
# export MLDECODERS_EEG_LPBE="2"

# python data_preprocessing/process_hugo_data.py
# python data_preprocessing/process_octave_data.py
# python studies/train_hugo_subject_specific_models.py

export MLDECODERS_RESULTS_DIR="results/0.5-8Hz"
export MLDECODERS_HUGO_DATA_FILE="data/hugo/test/data.h5"
export MLDECODERS_OCTAVE_DATA_FILE="data/octave/0.5-8Hz/data.h5"
export MLDECODERS_EEG_UPBE="8"
export MLDECODERS_EEG_LPBE="0.5"

# python studies/octave_subject_independent.py
python data_preprocessing/process_hugo_data.py
# python data_preprocessing/process_octave_data.py
# python studies/train_hugo_subject_specific_models.py
# python studies/train_octave_subject_specific_models.py
# python studies/train_hugo_leave_one_out_models.py
# python studies/train_hugo_population_models.py
# python studies/train_octave_population_models.py

# export MLDECODERS_RESULTS_DIR="results/0.5-16Hz"
# export MLDECODERS_HUGO_DATA_FILE="data/hugo/0.5-16Hz/data.h5"
# export MLDECODERS_OCTAVE_DATA_FILE="data/octave/0.5-16Hz/data.h5"
# export MLDECODERS_EEG_UPBE="16"
# export MLDECODERS_EEG_LPBE="0.5"

# python data_preprocessing/process_hugo_data.py
# python data_preprocessing/process_octave_data.py
# python studies/train_hugo_subject_specific_models.py


# export MLDECODERS_RESULTS_DIR="results/0.5-32Hz"
# export MLDECODERS_HUGO_DATA_FILE="data/hugo/0.5-32Hz/data.h5"
# export MLDECODERS_OCTAVE_DATA_FILE="data/octave/0.5-32Hz/data.h5"
# export MLDECODERS_EEG_UPBE="32"
# export MLDECODERS_EEG_LPBE="0.5"

# python data_preprocessing/process_hugo_data.py
# python data_preprocessing/process_octave_data.py
# python studies/train_hugo_subject_specific_models.py