import numpy as np
import json
import os
from pipeline.helpers import get_scores
import matplotlib.pyplot as plt
import pandas as pd

study = 'hugo_subject_specific'

plotting_config = json.load(open("plotting/plotting_config.json", "r"))
colors = plotting_config['colors']
models = plotting_config['models']

freq_bands = ['2-8Hz', '0.5-8Hz', '0.5-12Hz', '0.5-16Hz', '0.5-32Hz']

width=1/4

for i, freq_band in enumerate(freq_bands):
    results_dir = os.path.join("results", freq_band+'-090522')
    path = os.path.join(results_dir, 'predictions', study)

    predictions = {
        (mdl, participant):
            np.load(os.path.join(path, f"{mdl}_predictions_P{participant:02d}.npy")) for participant in range(13) for mdl in models
    }
    predictions = pd.DataFrame.from_dict(predictions)
    ground_truth = np.load(os.path.join("results/0.5-8Hz-090522", "predictions", "hugo_subject_specific", "ground_truth.npy"))

    scores = predictions.apply(lambda x: get_scores(x, ground_truth, batch_size=250))


    for j, mdl in enumerate(models):

        plt.boxplot(scores[mdl].mean(), positions=[i+(j-1)/4], patch_artist=True,
                    boxprops={'facecolor':colors[mdl], 'alpha':1, 'edgecolor':'black'},
                    flierprops={'marker':'x', 'markersize':2.5, 'markeredgecolor':'grey'},
                    whiskerprops={'color':'grey'},
                    capprops={'color':'grey'},
                    medianprops={'color':'yellow'})

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.title("Decoding performance in different EEG frequency bands")

plt.xticks(np.arange(len(freq_bands)), freq_bands)
plt.xlabel("EEG Frequency band")
plt.ylabel("Mean reconstruction score")

labels = ["Ridge", "CNN", "FCNN"]
handles = [plt.Rectangle((0,0),1,1, facecolor=colors[mdl], edgecolor='black') for mdl in models]
plt.legend(handles, labels, frameon=False)

plt.savefig(os.path.join("results", "0.5-8Hz-090522", "plots", "freqs.pdf"))
plt.close()