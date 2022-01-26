import torch
import numpy as np
import h5py

def get_dnn_predictions(dnn, loader, device='cuda'): # move to pipeline

     predictions = []
     dnn.eval()
     dnn = dnn.to(device)

     with torch.no_grad():
          for x, y in loader:
               x = x.to(device, dtype=torch.float)
               y_hat = dnn(x)
               predictions.append(y_hat.cpu().numpy())
     return np.hstack(predictions)

def get_ground_truth(data_file, parts_list, source='hugo', condition='clean', offset=50, speaker='attended'):
     with h5py.File(data_file, 'r') as data:
          if source == 'hugo':
               ground_truth = np.hstack([data[f'stim/part{j}/'][:] for j in parts_list])[:-offset]
          if source == 'octave':
               ground_truth = np.hstack([data[f'{condition}/part{j}/{speaker}'][:] for j in parts_list])[:-offset]

     return ground_truth

def get_conditions(participant=None):

     dutch_conditions = ['cleanDutch', 'lbDutch', 'mbDutch', 'hbDutch']
     english_conditions = ['clean', 'lb', 'mb', 'hb', 'fM', 'fW']

     participant_conditions = []

     if participant is None:
          participant_conditions += dutch_conditions + english_conditions
     
     if participant in ["YH00", "YH01", "YH02", "YH03", "YH04", "YH06", "YH07", "YH08", "YH09", "YH10", "YH11", "YH14", "YH15", "YH16", "YH17", "YH18", "YH19", "YH20"]:
          participant_conditions += english_conditions

     if participant in ["YH07", "YH08", "YH09", "YH10", "YH11", "YH12", "YH13", "YH14", "YH15", "YH16", "YH17", "YH18", "YH19", "YH20"]:
          participant_conditions += dutch_conditions

     return participant_conditions
