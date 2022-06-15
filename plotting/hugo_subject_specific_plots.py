import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json
import os
from pipeline.helpers import get_scores
import pandas as pd

np.random.seed(0)

plotting_config = json.load(open("plotting/plotting_config.json", "r"))
colors = plotting_config['colors']
models = plotting_config['models']
results_dir = "results/0.5-8Hz-090522"

if not os.path.exists(os.path.join(results_dir, 'plots')):
    os.mkdir(os.path.join(results_dir, 'plots'))

path = os.path.join(results_dir, 'predictions', 'hugo_subject_specific')

predictions = pd.DataFrame.from_dict({
    (mdl, participant):
        np.load(os.path.join(path, f"{mdl}_predictions_P{participant:02d}.npy")) for participant in range(13) for mdl in models
})
ground_truth = np.load(os.path.join(path, "ground_truth.npy"))

scores = predictions.apply(lambda x: get_scores(x, ground_truth, batch_size=250))
null_scores = predictions.apply(lambda x: get_scores(x, ground_truth, batch_size=250, null=True))
null_medians = null_scores.median()

####################
# Subject-level plot
####################

ridge_means = scores['ridge'].mean()
sorted_idx = np.argsort(ridge_means)

width=1/5
for i, model in enumerate(models):

    x = np.arange(13) + width*(i-1)

    plt.boxplot([scores[(model, i)] for i in sorted_idx],
                positions = x,
                widths=0.9*width,
                patch_artist=True,
                boxprops={'facecolor':colors[model], 'alpha':1, 'edgecolor':'black'},
                flierprops={'marker':'x', 'markersize':2.5, 'markeredgecolor':'grey'},
                whiskerprops={'color':'grey'},
                capprops={'color':'grey'},
                medianprops={'color':'yellow'})
    plt.scatter(x, null_medians[model], zorder=np.inf, color='#5ce600', marker='_', facecolor=None, s=20)

plt.axhline(0, color='black', linestyle='dotted', linewidth=1, zorder=-np.inf)

labels = ["Ridge", "CNN", "FCNN", "Noise Level"]
handles = [plt.Rectangle((0,0),1,1, facecolor=colors[mdl], edgecolor='black') for mdl in models] + \
            [Line2D([0], [0], c='#5ce600')]
leg = plt.legend(handles, labels, frameon=False, loc=(0.03,0.01), ncol=2, handleheight=.7, labelspacing=0.25, columnspacing=1)

plt.yticks(np.round(np.linspace(-1, 1, 11),1), np.round(np.linspace(-1, 1, 11),1))
plt.xticks(x-0.2, range(1,14))
plt.ylabel("Reconstruction score")
plt.xlabel("Participant")
plt.title("Subject-level reconstruction scores (subject-specific models)")
plt.ylim(-1, .9)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig(os.path.join(results_dir, "plots", "hugo_subject_specific_subject_level.pdf"))
plt.close()

####################
# Mean/std plot
####################


correlation_windows = list(range(3, 16)) + list(range(20, 51, 10)) + list(range(125, 1251, 125))
correlation_windows = np.array(correlation_windows)

means = {'ridge':[], 'cnn':[],'fcnn':[]}
null_means = {'ridge':[], 'cnn':[],'fcnn':[]}
stds = {'ridge':[], 'cnn':[],'fcnn':[]}
null_stds = {'ridge':[], 'cnn':[],'fcnn':[]}

for window in correlation_windows:
    scores = predictions.apply(lambda x: get_scores(x, ground_truth, batch_size=window))
    null_scores = predictions.apply(lambda x: get_scores(x, ground_truth, batch_size=window, null=True))

    for model in models:
        means[model].append(scores[model].to_numpy().mean())
        null_means[model].append(null_scores[model].to_numpy().mean())
        stds[model].append(np.std(scores[model].to_numpy(), ddof=1))
        null_stds[model].append(np.std(null_scores[model].to_numpy(), ddof=1))

for model in models:
    plt.plot(correlation_windows*1/125,means[model], color=colors[model], zorder=np.inf, linestyle='-')
    plt.plot(correlation_windows*1/125,null_means[model], color=colors[model], zorder=1, linestyle='dotted')

plt.xscale("log")

custom_lines = [Line2D([0], [0], color=colors['ridge'], lw=4),
                Line2D([0], [0], color=colors['cnn'], lw=4),
                Line2D([0], [0], color=colors['fcnn'], lw=4)]

plt.legend(custom_lines, ['Ridge', 'CNN', 'FCNN'], loc=(0.08,0.7), frameon=False)
plt.ylabel("Mean reconstruction score")
plt.xlabel("Window size [s]")
plt.title("Mean reconstruction score against window size")
plt.xlim(0.1, None)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig(os.path.join(results_dir, "plots", "hugo_subject_specific_means.pdf"))
plt.close()

for model in models:
    plt.plot(correlation_windows*1/125,stds[model], color=colors[model], zorder=np.inf, linestyle='-')
    plt.plot(correlation_windows*1/125,null_stds[model], color=colors[model], zorder=1, linestyle='dotted')

plt.xscale("log")

custom_lines = [Line2D([0], [0], color=colors['ridge'], lw=4),
                Line2D([0], [0], color=colors['cnn'], lw=4),
                Line2D([0], [0], color=colors['fcnn'], lw=4)]

plt.legend(custom_lines, ['Ridge', 'CNN', 'FCNN'], loc='upper right', frameon=False)
plt.ylabel("Standard deviation of reconstruction score")
plt.xlabel("Window size [s]")
plt.title("Variability of reconstruction score against window size")
plt.xlim(0.1, None)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig(os.path.join(results_dir, "plots", "hugo_subject_specific_stds.pdf"))
plt.close()