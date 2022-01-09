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
        channels = np.arange(64),
        return_null=False,
        normalise=False,
        normalise_window=750):

        self.parts_list = parts_list
        self.data_dir=data_dir
        self.num_input=num_input
        self.channels=channels
        self.return_null=return_null
        self.normalise=normalise
        self.normalise_window=normalise_window

        if type(participant)==type(int()):
            self.participants=[participant]
        else:
            self.participants=participant

        self._initialise_data()

    def _initialise_data(self):

        eeg = []
        stim = []

        f = h5py.File(self.data_dir, "r")
        for each_participant in self.participants:
            eeg += [f['eeg/P0{}/part{}/'.format(each_participant, j)][:][self.channels] for j in self.parts_list]
            stim += [f['stim/part{}/'.format(j)][:] for j in self.parts_list]
        
        f.close()

        self.eeg = np.hstack(eeg)
        self.stim = np.hstack(stim)
        self.null_stim = self.stim[::-1]

    def __getitem__(self, idx):

        if self.normalise:
            std = self.eeg[:, idx:idx+self.normalise_window].std(axis=1)
            #if std==0: std=1
            mean = self.eeg[:, idx:idx+self.normalise_window].mean(axis=1)

            idx_=idx+self.normalise_window
        else:
            std=1
            mean=0

        if self.return_null:
            return self.eeg[:, idx:idx+self.num_input], self.stim[idx], self.null_stim[idx]
        else:
            return self.eeg[:, idx:idx+self.num_input], self.stim[idx]
    
    def __len__(self):
        if self.normalise:
            return self.stim.size - self.num_input - self.normalise_window
        else:
            return self.stim.size - self.num_input

class HugoSeq2Seq(Dataset):
    '''
    dataloader for reading Hugo's data
    '''
    def __init__(
        self,
        parts_list,
        data_dir,
        participant=0,
        num_input=50,
        stimkey='env60Hz',
        eegkey='lowpassed0.3_12Hz',
        channels = np.arange(64)):


        self.parts_list = parts_list
        self.data_dir=data_dir
        self.num_input=num_input
        self.stimkey=stimkey
        self.eegkey=eegkey
        self.channels=channels

        if type(participant)==type(int()):
            self.participants=[participant]
        else:
            self.participants=participant

        self._initialise_data()

    def _initialise_data(self):

        eeg = []
        stim = []

        f = h5py.File(self.data_dir, "r")
        for each_participant in self.participants:
            eeg += [f['eeg/P0{}/part{}/'.format(each_participant, j) + self.eegkey][:][self.channels] for j in self.parts_list]
            stim += [f['stimulus/part{}/'.format(j) + self.stimkey][:] for j in self.parts_list]
        
        f.close()

        self.eeg = np.hstack(eeg)
        self.stim = np.hstack(stim)

    def __getitem__(self, idx):
        return self.eeg[:, idx:idx+self.num_input], self.stim[idx:idx+self.num_input]
    
    def __len__(self):
        return self.stim.size - self.num_input

class OctaveMapped(Dataset):
    '''
    dataloader for reading Octave's two-speaker data
    '''
    def __init__(self, num_input=50, participant="YH00", condition="fW", parts=[1], data_dir="/media/mdt20/Storage/data/Octave/processed", return_null=True):
        self.num_input=num_input
        self.return_null=return_null

        eegs = []
        stimsa = []
        stimsu = []

        with h5py.File(os.path.join(data_dir, "data.h5"), "r") as f:

            for each_part in parts:
                eeg = f[f"{condition}/part{each_part}/{participant}"][:]
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
        self.stim_a_null = np.hstack(stimsa[::-1])
        self.stim_u_null = np.hstack(stimsu[::-1])

    def __getitem__(self, idx):
        if self.return_null:
            return self.eeg[:, idx:idx+self.num_input], [self.stim_u[idx], self.stim_a[idx]], [self.stim_u_null[idx], self.stim_a_null[idx]]
        else:
            return self.eeg[:, idx:idx+self.num_input], [self.stim_u[idx], self.stim_a[idx]]
    
    def __len__(self):
        return self.stim_a.size - self.num_input

    def raw_data(self):
        return self.eeg, [self.stim_u, self.stim_a]