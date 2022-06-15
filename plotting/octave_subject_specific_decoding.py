import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pipeline.helpers import get_scores

results_dir = "results/0.5-8Hz-090522"
results_path = os.path.join(results_dir, 'predictions', 'octave_subject_specific')

plotting_config = json.load(open("plotting/plotting_config.json", "r"))
colors = plotting_config['colors']
models = plotting_config['models']
english_participants = plotting_config['octave_english_participants']
dutch_participants = plotting_config['octave_dutch_participants']

correlation_window=250

# collect mean scores

conditions_mean_scores = {
    'competing-attended':{'ridge':[], 'cnn':[], 'fcnn':[]},
    'competing-unattended':{'ridge':[], 'cnn':[], 'fcnn':[]},
    'dutch':{'ridge':[], 'cnn':[], 'fcnn':[]},
    'babble':{'ridge':[], 'cnn':[], 'fcnn':[]}
}

fM_scores = {'ridge':[], 'cnn':[], 'fcnn':[]}
fW_scores = {'ridge':[], 'cnn':[], 'fcnn':[]}
for participant in english_participants:

    attended_fM_ground_truth = np.load(os.path.join(results_path, f'attended_ground_truth_{participant}_fM.npy'))
    attended_fW_ground_truth = np.load(os.path.join(results_path, f'attended_ground_truth_{participant}_fW.npy'))
    unattended_fM_ground_truth = np.load(os.path.join(results_path, f'unattended_ground_truth_{participant}_fM.npy'))
    unattended_fW_ground_truth = np.load(os.path.join(results_path, f'unattended_ground_truth_{participant}_fW.npy'))

    for model in models:
        fM_predictions = np.load(os.path.join(results_path, f'{model}_predictions_{participant}_fM.npy'))
        fW_predictions = np.load(os.path.join(results_path, f'{model}_predictions_{participant}_fW.npy'))

        attended_fM_scores = get_scores(attended_fM_ground_truth, fM_predictions, batch_size=correlation_window)
        attended_fW_scores = get_scores(attended_fW_ground_truth, fW_predictions, batch_size=correlation_window)
        unattended_fM_scores = get_scores(unattended_fM_ground_truth, fM_predictions, batch_size=correlation_window)
        unattended_fW_scores = get_scores(unattended_fW_ground_truth, fW_predictions, batch_size=correlation_window)

        fM_acc = np.sum(attended_fM_scores > unattended_fM_scores).sum()/attended_fM_scores.size
        fW_acc = np.sum(attended_fW_scores > unattended_fW_scores).sum()/attended_fW_scores.size

        fM_scores[model].append(fM_acc)
        fW_scores[model].append(fW_acc)

width = 0.6

for j, model in enumerate(models):
    bp = plt.boxplot(fM_scores[model],
                positions = [(j-1)*width*1.2],
                patch_artist=True,
                boxprops = {'facecolor':colors[model], 'edgecolor':'black'},
                medianprops={'color':'yellow'},
                flierprops={'marker':'x'},
                widths=width)

for j, model in enumerate(models):
    bp = plt.boxplot(fW_scores[model],
                positions = [(j-1)*width*1.2+4],
                patch_artist=True,
                boxprops = {'facecolor':colors[model], 'edgecolor':'black'},
                medianprops={'color':'yellow'},
                flierprops={'marker':'x'},
                widths=width)

plt.xticks([0,4], ["Male speaker attended", "Female speaker attended"])

handles = [plt.Rectangle((0,0),1,1, facecolor=colors['ridge'], edgecolor='black'),
                plt.Rectangle((0,0),1,1, facecolor=colors['cnn'], edgecolor='black'),
                plt.Rectangle((0,0),1,1, facecolor=colors['fcnn'], edgecolor='black')]

plt.legend(handles, ['Ridge', 'CNN', 'FCNN'], frameon=False)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.axhline(0.5, color='black', linestyle='dotted', linewidth=1, zorder=-np.inf)
plt.ylim(0.48,1.02)

plt.ylabel("Attention decoding accuracy")
yticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
plt.yticks(yticks, [f"{int(p*100):2d}%" for p in yticks])

plt.title("Attention decoding accuracies in competing-speaker scenarios")

plt.savefig(os.path.join(results_dir, "plots", "octave_subject_specific_decoding.pdf"))