import numpy as np
import h5py
from numpy.core.numeric import normalize_axis_tuple
from torch.functional import norm
from torch.utils.data import Dataset
import torch
import torch.multiprocessing
import mne
import glob
from scipy.io import loadmat
from mne.io import read_raw_brainvision
from scipy.signal import hilbert
from scipy.stats import zscore
import os

torch.multiprocessing.set_sharing_strategy('file_system')

class HugoMapped(Dataset):
    '''
    dataloader for reading Hugo's data
    '''
    def __init__(
        self,
        parts_list,
        data_dir,
        participant=0,
        num_input=50,
        channels = np.arange(63)):

        self.parts_list = parts_list
        self.data_dir=data_dir
        self.num_input=num_input
        self.channels=channels

        if type(participant)==type(int()):
            self.participants=[participant]
        else:
            self.participants=participant

        self._initialise_data()

    def _initialise_data(self):

        eeg = []
        stim = []

        with h5py.File(self.data_dir, "r") as f:
            
            for each_participant in self.participants:
                eeg += [f['eeg/P0{}/part{}/'.format(each_participant, j)][:][self.channels] for j in self.parts_list]
                stim += [f['stim/part{}/'.format(j)][:] for j in self.parts_list]

        self.eeg = np.hstack(eeg)
        self.stim = np.hstack(stim)

    def __getitem__(self, idx):

        return self.eeg[:, idx:idx+self.num_input], self.stim[idx]
    
    def __len__(self):
        return self.stim.size - self.num_input

class OctaveMapped(Dataset):
    '''
    dataloader for reading Octave's two-speaker data
    '''
    def __init__(
        self,
        parts_list=[1],
        data_dir="/media/mdt20/Storage/data/Octave/processed",
        participant="YH00",
        num_input=50,
        channels = np.arange(63),
        condition="clean"):

        self.num_input=num_input

        eegs = []
        stimsa = []
        stimsu = []

        with h5py.File(data_dir, "r") as f:

            for each_part in parts_list:
                eeg = f[f"{condition}/part{each_part}/{participant}"][:][channels]
                S_u = f[f"{condition}/part{each_part}/unattended"][:]
                S_a = f[f"{condition}/part{each_part}/attended"][:]

                if S_a.size < eeg.shape[1]:
                    S_a = np.pad(S_a, (0, eeg.shape[1]-S_a.size))
                if S_u.size < eeg.shape[1]:
                    S_u = np.pad(S_u, (0, eeg.shape[1]-S_u.size))
                if S_a.size > eeg.shape[1]:
                    S_a = S_a[:eeg.shape[1]]
                if S_u.size > eeg.shape[1]:
                    S_u = S_u[:eeg.shape[1]]

                eegs.append(eeg)
                stimsa.append(S_a)
                stimsu.append(S_u)
            
        self.eeg = np.hstack(eegs)
        self.stim_a = np.hstack(stimsa)
        self.stim_u = np.hstack(stimsu)
        self.stim = self.stim_a

    def __getitem__(self, idx):
        return self.eeg[:, idx:idx+self.num_input], self.stim_a[idx]
    
    def __len__(self):
        return self.stim_a.size - self.num_input