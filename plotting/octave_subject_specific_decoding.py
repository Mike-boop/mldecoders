import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json
import os
from pipeline.helpers import get_scores, add_sig, get_stars
from scipy.stats import ranksums, wilcoxon, ttest_rel
from statsmodels.stats import multitest

results_dir = "results/0.5-8Hz"
results_path = os.path.join(results_dir, 'predictions', 'octave_subject_specific')

plotting_config = json.load(open("plotting/plotting_config.json", "r"))
colors = plotting_config['colors']
models = plotting_config['models']
english_participants = plotting_config['octave_english_participants']
dutch_participants = plotting_config['octave_dutch_participants']

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

        attended_fM_scores = get_scores(attended_fM_ground_truth, fM_predictions, batch_size=250)
        attended_fW_scores = get_scores(attended_fW_ground_truth, fW_predictions, batch_size=250)
        unattended_fM_scores = get_scores(unattended_fM_ground_truth, fM_predictions, batch_size=250)
        unattended_fW_scores = get_scores(unattended_fW_ground_truth, fW_predictions, batch_size=250)

        fM_acc = np.sum(attended_fM_scores > unattended_fM_scores).sum()/attended_fM_scores.size
        fW_acc = np.sum(attended_fW_scores > unattended_fW_scores).sum()/attended_fW_scores.size

        fM_scores[model].append(fM_acc)
        fW_scores[model].append(fW_acc)

plt.boxplot(fM_scores.values(), positions=[0,1,2])
plt.boxplot(fW_scores.values(), positions=[4,5,6])
plt.savefig("tests/tmp.test_acc.pdf")