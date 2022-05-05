import numpy as np
import glob
import pathlib
from mne.io import read_raw_brainvision
from scipy.io import loadmat
from scipy.stats import zscore
from scipy.signal import hilbert
import mne
import h5py
import os

try:
    data_file = os.environ['MLDECODERS_OCTAVE_DATA_FILE']
    upbe = float(os.environ['MLDECODERS_EEG_UPBE'])
    lpbe = float(os.environ['MLDECODERS_EEG_LPBE'])
except KeyError:
    print('please configure the environment!')
    exit()

subjects = ["YH{:02d}".format(i) for i in [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]

speakers = {
    "clean":{
        "attended":"story",
        "unattended":None
    },
    "lb":{
        "attended":"story",
        "unattended":"babble"
    },
    "mb":{
        "attended":"story",
        "unattended":"babble"
    },
    "hb":{
        "attended":"story",
        "unattended":"babble"
    },
    "fM":{
        "attended":"story",
        "unattended":"distractor"
    },
    "fW":{
        "attended":"story",
        "unattended":"distractor"
    },
    "cleanDutch":{
        "attended":"dutch",
        "unattended":None
    },
    "lbDutch":{
        "attended":"dutch",
        "unattended":"babble"
    },
    "mbDutch":{
        "attended":"dutch",
        "unattended":"babble"
    },
    "hbDutch":{
        "attended":"dutch",
        "unattended":"babble"
    }
}

def get_conditions(subject):
    files = glob.glob(f"/media/mdt20/Storage/data/Octave/raw-YH/{subject}/*.eeg")
    fnames = [pathlib.Path(file).name for file in files]
    conditions = {fname.split("_")[1] for fname in fnames}

    return conditions

def process_eeg(subject, condition, part):

    if subject == "YH02" and 'Dutch' in condition:
        raw = read_raw_brainvision(f"/media/mdt20/Storage/data/Octave/raw-YH/YH02/Dutch/YH02_{condition}_{part}_dutch.vhdr", preload=True)
    else:
        raw = read_raw_brainvision(f"/media/mdt20/Storage/data/Octave/raw-YH/{subject}/{subject}_{condition}_{part}.vhdr", preload=True)
    raw.drop_channels('Sound')

    raw.filter(None, upbe)
    raw.resample(125)

    raw.filter(lpbe,None)

    start = raw.annotations[1]['onset']
    end = raw.annotations[2]['onset']
    raw.crop(start, end)
    eeg = raw.get_data()

    eeg = zscore(eeg, axis=1)

    return eeg

def process_stim(condition, part):

    fstim = glob.glob(f"/media/mdt20/Storage/data/Octave/stimuli/{condition}/*{part}.mat")[0]
    S = loadmat(fstim)
    key_attended = speakers[condition]['attended']
    key_unattended = speakers[condition]['unattended']

    S_a = S[key_attended].flatten()
    S_a = np.abs(hilbert(S_a))
    
    if key_unattended is None:
        S_u = np.zeros_like(S_a)
    else:
        S_u = S[key_unattended].flatten()
        S_u = np.abs(hilbert(S_u))

    S_u = mne.filter.filter_data(S_u, 44100,None, 50)
    S_a = mne.filter.filter_data(S_a, 44100,None, 50)

    S_u = mne.filter.resample(S_u, 125, 44100)
    S_a = mne.filter.resample(S_a, 125, 44100)

    if condition in ["lb", "mb", "hb"]:
        S_a = np.pad(S_a, (125, 0))

    length = min(S_a.size, S_u.size)

    return S_u[:length], S_a[:length]

if __name__ == '__main__':

    with h5py.File(data_file, "w") as f:

        for condition in ["clean", "lb", "mb", "hb", "fW", "fM", "cleanDutch", "lbDutch", "mbDutch", "hbDutch"]:
            for part in range(1,5):
                S_u, S_a = process_stim(condition, part)
                f.create_dataset(f"{condition}/part{part}/unattended", data=S_u)
                f.create_dataset(f"{condition}/part{part}/attended", data=S_a)

        for participant in ["YH00", "YH01", "YH02", "YH03", "YH04", "YH06", "YH07", "YH08", "YH09", "YH10", "YH11", "YH12", "YH13", "YH14", "YH15", "YH16", "YH17", "YH18", "YH19", "YH20"]:
            for condition in get_conditions(participant):
                for part in range(1,5):
                    eeg = process_eeg(participant, condition, part)
                    f.create_dataset(f"{condition}/part{part}/{participant}", data=eeg)