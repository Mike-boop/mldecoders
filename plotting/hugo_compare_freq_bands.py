import numpy as np
import json
import os
from pipeline.helpers import get_scores
import matplotlib.pyplot as plt

np.random.seed(0)
study = 'hugo_subject_specific'

plotting_config = json.load(open("plotting/plotting_config.json", "r"))
colors = plotting_config['colors']
models = plotting_config['models']

freq_bands = ['2-8Hz', '0.5-8Hz', '0.5-12Hz', '0.5-16Hz', '0.5-32Hz']

width=1/4

for i, freq_band in enumerate(freq_bands):
    results_dir = os.path.join("results", freq_band)
    path = os.path.join(results_dir, 'predictions', study)

    ridge_predictions = [np.load(os.path.join(path, f"ridge_predictions_P{participant:02d}.npy")) for participant in range(13)]
    cnn_predictions = [np.load(os.path.join(path, f"cnn_predictions_P{participant:02d}.npy")) for participant in range(13)]
    fcnn_predictions = [np.load(os.path.join(path, f"fcnn_predictions_P{participant:02d}.npy")) for participant in range(13)]
    ground_truth = np.load(os.path.join("results/0.5-8Hz", "predictions", "hugo_subject_specific", "ground_truth.npy"))

    scores = {
        'ridge':[get_scores(pred, ground_truth, batch_size=250) for pred in ridge_predictions],
        'cnn'  :[get_scores(pred, ground_truth, batch_size=250) for pred in cnn_predictions],
        'fcnn' :[get_scores(pred, ground_truth, batch_size=250) for pred in fcnn_predictions]
    }

    mean_scores = {
        'ridge':[np.mean(participant_scores) for participant_scores in scores['ridge']],
        'cnn'  :[np.mean(participant_scores) for participant_scores in scores['cnn']],
        'fcnn' :[np.mean(participant_scores) for participant_scores in scores['fcnn']]
    }


    for j, mdl in enumerate(models):

        plt.boxplot(mean_scores[mdl], positions=[i+(j-1)/4], patch_artist=True,
                    boxprops={'facecolor':colors[mdl], 'alpha':1, 'edgecolor':'black'},
                    flierprops={'marker':'x', 'markersize':2.5, 'markeredgecolor':'grey'},
                    whiskerprops={'color':'grey'},
                    capprops={'color':'grey'},
                    medianprops={'color':'yellow'})

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xticks(np.arange(len(freq_bands)), freq_bands)
plt.xlabel("Frequency band")
plt.ylabel("Mean reconstruction score")

labels = ["Ridge", "CNN", "FCNN"]
handles = [plt.Rectangle((0,0),1,1, facecolor=colors[mdl], edgecolor='black') for mdl in models]
plt.legend(handles, labels, frameon=False)

plt.savefig(os.path.join("results", "0.5-8Hz", "plots", "freqs.pdf"))
plt.close()