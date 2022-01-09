#!/bin/bash

source /home/mdt20/miniconda3/bin/activate env-dnns

cd /home/mdt20/Code/mldecoders
python scripts/train_DNNs_hugo_data.py
python scripts/train_ridge_hugo_data.py
python scripts/generalise_correlations_octave_data.py
python scripts/generalise_correlations_dutch.py
