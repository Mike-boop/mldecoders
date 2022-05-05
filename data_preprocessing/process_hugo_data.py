import os
import h5py
import glob
import numpy as np
from scipy.signal import resample, hilbert
from scipy.stats import zscore
from scipy.io import loadmat, wavfile
from mne.io import read_raw_brainvision
from mne.filter import filter_data, resample

try:
    data_file = os.environ['MLDECODERS_HUGO_DATA_FILE']
    upbe = float(os.environ['MLDECODERS_EEG_UPBE'])
    lpbe = float(os.environ['MLDECODERS_EEG_LPBE'])
except KeyError:
    print('please configure the environment!')
    exit()

participants = [1,2,3,4,5,6,7,8,9,10,12,13,14]
story_parts = list(range(15))

eeg_filepath = lambda participant: glob.glob("/media/mdt20/Storage/data/hugo_data_processing/raw/P{:02d}*/*.vhdr".format(participant))[0]
onsets = loadmat('/media/mdt20/Storage/data/hugo_data_processing/raw/onsets.mat')['onsets']

path_to_data = '/media/mdt20/Storage/data/hugo_data_processing/raw_data'
story_part_names = ["AUNP0{}".format(i) for i in range(1,9)] + ["BROP0{}".format(i) for i in range(1,4)] + ["FLOP0{}".format(i) for i in range(1,5)]
audio_path = lambda s: os.path.join(path_to_data, 'stories', 'story_parts', 'alignement_data', s, s+'.wav')

Fs = 125

def process_stimuli(story_idx):
    '''
    hpf: the high-pass filter window to be used
    lpfs: a list of N low-pass filter windows
    srate: target sampling rate

    returns: N filtered and resampled windowed
    '''

    story_part_name = story_part_names[story_idx]
    srate0, stimulus = wavfile.read(audio_path(story_part_name))
    duration = len(stimulus)/srate0

    out = np.abs(hilbert(stimulus))
    out = filter_data(out, srate0, None, 50)
    out = resample(out, 125, srate0)

    return zscore(out), duration

def process_eeg(participant_idx, lengths):

    participant=participants[participant_idx]
    raw = read_raw_brainvision(eeg_filepath(participant), preload=True)
    raw = raw.drop_channels('Sound')

    raw.filter(None, upbe)
    raw.resample(125)
    raw.filter(lpbe, None)

    eeg = zscore(raw.get_data(), axis=1)

    EEG = []

    for j in story_parts:

        length = lengths[j]
        start_time = onsets[participant_idx, j]

        EEG.append(eeg[:, int(start_time*125):int(start_time*125)+length])
    
    return EEG

if __name__ == '__main__':

    with h5py.File(data_file, 'w') as f:
        target_lengths = []
        for i in story_parts:
            story, duration = process_stimuli(i)
            target_lengths.append(len(story))

            f.create_dataset('stim/part{}'.format(i), data=story)
            
        for j in range(len(participants)):
            eeg = process_eeg(j, target_lengths)
            for i in story_parts:
                f.create_dataset('eeg/P0{}/part{}'.format(j, i), data=eeg[i])
