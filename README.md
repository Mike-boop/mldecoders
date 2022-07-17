# mldecoders

This repository contains all of the code used in our publication "Robust decoding of the speech envelope from EEG recordings through deep neural networks" 
(Thornton, Mandic, and Reichenbach, 2022) [1].

## Overview

- data_preprocessing: these scripts were used to preprocess the raw brainvision (.vhdr) EEG files and stimuli of Dataset 1 and Dataset 2. These datasets 
were collected by Weissbart et al and Etard et al [2,3]. Please see our paper, or those papers, for more details.

- pipeline: this module contains definitions of the deep neural networks, as well as ridge regression. Pytorch Dataset objects are defined for Dataset 
1 and Dataset 2. Training loops and evaluation loops are also defined.

- plotting: these scripts were used to perform statistical analyses and to generate the figures used in the publication.

- studies: these scripts are used to train and evaluate the DNNs and ridge regression.

- scripts: this contains run.sh, which defines paths to Dataset 1 and Dataset 2, as well as the folder in which results should be saved. It then runs the 
scripts in 'studies'.

## Requirements

The preprocessing scripts were run with the following packages:

- h5py == 3.6.0
- mne == 0.24.1
- numpy == 1.22.0
- scipy == 1.7.3

The rest of the package requires the following:

- h5py == 3.6.0
- matplotlib == 3.5.1
- numpy == 1.22.0
- optuna == 2.10.0
- pandas == 1.3.5
- pytorch == 1.10.1 (we also used cuda11.3 to enable hardware acceleration)
- scipy == 1.7.3
- statsmodels == 0.13.1

## Data availability

Preprocessed clean speech data is available at https://figshare.com/articles/dataset/EEG_recordings_and_stimuli/9033983. Please note that the preprocessing
procedure used to produce these data was different to ours (see [2] for details).

Please email Professor Reichenbach at <tobias.j.reichenbach@fau.de> for data requests. Please direct other queries to me at <m.thornton20@imperial.ac.uk>.

## References

[1] M. Thornton, D. P. Mandic and T. Reichenbach, "Robust decoding of the speech envelope from EEG recordings through deep neural networks," _Journal of Neural Engineering_, vol. 19, no. 4, pp. 046007, Jul. 2022.

[2] H. Weissbart, K. D. Kandylaki and T. Reichenbach, “Cortical tracking of surprisal during continuous speech comprehension,” _Journal of Cognitive Neuroscience_, vol. 32, no. 1, pp. 155–166, Jan. 2020.
 
[3] O. Etard and T. Reichenbach, “Neural speech tracking in the theta and in the delta frequency band differentially encode clarity and comprehension 
of speech in noise,” _The Journal of Neuroscience_, vol. 39, no. 29, pp. 5750–5759, May 2019.
