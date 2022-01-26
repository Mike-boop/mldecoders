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

for participant in english_participants:

    attended_fM_ground_truth = np.load(os.path.join(results_path, f'attended_ground_truth_{participant}_fM.npy'))
    attended_fW_ground_truth = np.load(os.path.join(results_path, f'attended_ground_truth_{participant}_fW.npy'))
    unattended_fM_ground_truth = np.load(os.path.join(results_path, f'unattended_ground_truth_{participant}_fM.npy'))
    unattended_fW_ground_truth = np.load(os.path.join(results_path, f'unattended_ground_truth_{participant}_fW.npy'))
    mb_ground_truth = np.load(os.path.join(results_path, f'attended_ground_truth_{participant}_mb.npy'))

    if participant in dutch_participants:
        cleanDutch_ground_truth = np.load(os.path.join(results_path, f'attended_ground_truth_{participant}_cleanDutch.npy'))

    for model in models:
        fM_predictions = np.load(os.path.join(results_path, f'{model}_predictions_{participant}_fM.npy'))
        fW_predictions = np.load(os.path.join(results_path, f'{model}_predictions_{participant}_fW.npy'))
        mb_predictions = np.load(os.path.join(results_path, f'{model}_predictions_{participant}_mb.npy'))

        attended_fM_scores = get_scores(attended_fM_ground_truth, fM_predictions, batch_size=250)
        attended_fW_scores = get_scores(attended_fW_ground_truth, fW_predictions, batch_size=250)
        unattended_fM_scores = get_scores(unattended_fM_ground_truth, fM_predictions, batch_size=250)
        unattended_fW_scores = get_scores(unattended_fW_ground_truth, fW_predictions, batch_size=250)

        attended_competing_scores = np.hstack([attended_fM_scores[~np.isnan(attended_fM_scores)], attended_fW_scores[~np.isnan(attended_fW_scores)]])
        unattended_competing_scores = np.hstack([unattended_fM_scores[~np.isnan(unattended_fM_scores)], unattended_fW_scores[~np.isnan(unattended_fW_scores)]])
        mb_scores = get_scores(mb_ground_truth, mb_predictions)
        mb_scores = mb_scores[~np.isnan(mb_scores)]
    
        conditions_mean_scores['competing-attended'][model].append(np.mean(attended_competing_scores))
        conditions_mean_scores['competing-unattended'][model].append(np.mean(unattended_competing_scores))
        conditions_mean_scores['babble'][model].append(np.mean(mb_scores))

        if participant in dutch_participants:
            cleanDutch_predictions = np.load(os.path.join(results_path, f'{model}_predictions_{participant}_cleanDutch.npy'))
            cleanDutch_scores = get_scores(cleanDutch_ground_truth, cleanDutch_predictions, batch_size=250)
            cleanDutch_scores = cleanDutch_scores[~np.isnan(cleanDutch_scores)]
            conditions_mean_scores['dutch'][model].append(np.mean(cleanDutch_scores))

# collect p-vals

p_attended_competing_ridge_cnn = ttest_rel(conditions_mean_scores['competing-attended']['ridge'], conditions_mean_scores['competing-attended']['cnn'], alternative='less')[1]
p_attended_competing_ridge_fcnn = ttest_rel(conditions_mean_scores['competing-attended']['ridge'], conditions_mean_scores['competing-attended']['fcnn'], alternative='less')[1]
p_attended_competing_cnn_fcnn = ttest_rel(conditions_mean_scores['competing-attended']['cnn'], conditions_mean_scores['competing-attended']['fcnn'], alternative='two-sided')[1]

# p_unattended_competing_ridge_cnn = ttest_rel(conditions_mean_scores['competing-unattended']['ridge'], conditions_mean_scores['competing-unattended']['cnn'], alternative='less')[1]
# p_unattended_competing_ridge_fcnn = ttest_rel(conditions_mean_scores['competing-unattended']['ridge'], conditions_mean_scores['competing-unattended']['fcnn'], alternative='less')[1]
# p_unattended_competing_cnn_fcnn = ttest_rel(conditions_mean_scores['competing-unattended']['cnn'], conditions_mean_scores['competing-unattended']['fcnn'], alternative='two-sided')[1]

p_babble_ridge_cnn = ttest_rel(conditions_mean_scores['babble']['ridge'], conditions_mean_scores['babble']['cnn'], alternative='less')[1]
p_babble_ridge_fcnn = ttest_rel(conditions_mean_scores['babble']['ridge'], conditions_mean_scores['babble']['fcnn'], alternative='less')[1]
p_babble_cnn_fcnn = ttest_rel(conditions_mean_scores['babble']['cnn'], conditions_mean_scores['babble']['fcnn'], alternative='two-sided')[1]

p_dutch_ridge_cnn = ttest_rel(conditions_mean_scores['dutch']['ridge'], conditions_mean_scores['dutch']['cnn'], alternative='less')[1]
p_dutch_ridge_fcnn = ttest_rel(conditions_mean_scores['dutch']['ridge'], conditions_mean_scores['dutch']['fcnn'], alternative='less')[1]
p_dutch_cnn_fcnn = ttest_rel(conditions_mean_scores['dutch']['cnn'], conditions_mean_scores['dutch']['fcnn'], alternative='two-sided')[1]

# pvals = [p_unattended_competing_ridge_cnn, p_unattended_competing_ridge_fcnn, p_unattended_competing_cnn_fcnn,
#          p_attended_competing_ridge_cnn, p_attended_competing_ridge_fcnn, p_attended_competing_cnn_fcnn,
#          p_babble_ridge_cnn, p_babble_ridge_fcnn, p_babble_cnn_fcnn,
#          p_dutch_ridge_cnn, p_dutch_ridge_fcnn, p_dutch_cnn_fcnn]

# [p_unattended_competing_ridge_cnn, p_unattended_competing_ridge_fcnn, p_unattended_competing_cnn_fcnn,
#          p_attended_competing_ridge_cnn, p_attended_competing_ridge_fcnn, p_attended_competing_cnn_fcnn,
#          p_babble_ridge_cnn, p_babble_ridge_fcnn, p_babble_cnn_fcnn,
#          p_dutch_ridge_cnn, p_dutch_ridge_fcnn, p_dutch_cnn_fcnn] = multitest.fdrcorrection(pvals)[1]

pvals = [p_attended_competing_ridge_cnn, p_attended_competing_ridge_fcnn, p_attended_competing_cnn_fcnn,
         p_babble_ridge_cnn, p_babble_ridge_fcnn, p_babble_cnn_fcnn,
         p_dutch_ridge_cnn, p_dutch_ridge_fcnn, p_dutch_cnn_fcnn]

[p_attended_competing_ridge_cnn, p_attended_competing_ridge_fcnn, p_attended_competing_cnn_fcnn,
         p_babble_ridge_cnn, p_babble_ridge_fcnn, p_babble_cnn_fcnn,
         p_dutch_ridge_cnn, p_dutch_ridge_fcnn, p_dutch_cnn_fcnn] = multitest.fdrcorrection(pvals)[1]

# plot data

width = 1/5

for i, cond in enumerate(['competing-unattended', 'babble', 'dutch', 'competing-attended']):
    for j, model in enumerate(models):
        bp = plt.boxplot(conditions_mean_scores[cond][model],
                    positions = [i + (j-1)*width],
                    patch_artist=True,
                    boxprops = {'facecolor':colors[model], 'edgecolor':'black'},
                    medianprops={'color':'yellow'},
                    flierprops={'marker':'x'})

plt.savefig(os.path.join(results_dir, "plots", "test.pdf"))

# add significances

h = -0.012
level1 = -0.015
level2 = -0.03

# if p_unattended_competing_ridge_cnn < 0.05:
#     ax = add_sig(plt.gca(),0-width,0,level1)
#     plt.text(0-width/2, level1+h, get_stars(p_unattended_competing_ridge_cnn), ha='center')
    
# if p_unattended_competing_cnn_fcnn < 0.05:
#     ax = add_sig(plt.gca(),0,0+width,level1)
#     plt.text(0+width/2, level1+h, get_stars(p_unattended_competing_cnn_fcnn), ha='center')
    
# if p_unattended_competing_ridge_fcnn < 0.05:
#     ax = add_sig(plt.gca(),0-width,0+width,level2)
#     plt.text(0, level2+h, get_stars(p_unattended_competing_ridge_fcnn), ha='center')


if p_attended_competing_ridge_cnn < 0.05:
    ax = add_sig(plt.gca(),1-width,1,level1)
    plt.text(1-width/2, level1+h, get_stars(p_attended_competing_ridge_cnn), ha='center')
    
if p_attended_competing_cnn_fcnn < 0.05:
    ax = add_sig(plt.gca(),1,1+width,level1)
    plt.text(1+width/2, level1+h, get_stars(p_attended_competing_cnn_fcnn), ha='center')
    
if p_attended_competing_ridge_fcnn < 0.05:
    ax = add_sig(plt.gca(),1-width,1+width,level2)
    plt.text(1, level2+h, get_stars(p_attended_competing_ridge_fcnn), ha='center')

    
if p_babble_ridge_cnn < 0.05:
    ax = add_sig(plt.gca(),2-width,2,level1)
    plt.text(2-width/2, level1+h, get_stars(p_babble_ridge_cnn), ha='center')
    
if p_babble_cnn_fcnn < 0.05:
    ax = add_sig(plt.gca(),2,2+width,level1)
    plt.text(2+width/2, level1+h, get_stars(p_babble_cnn_fcnn), ha='center')
    
if p_babble_ridge_fcnn < 0.05:
    ax = add_sig(plt.gca(),2-width,2+width,level2)
    plt.text(2, level2+h, get_stars(p_babble_ridge_fcnn), ha='center')


if p_dutch_ridge_cnn < 0.05:
    ax = add_sig(plt.gca(),3-width,3,level1)
    plt.text(3-width/2, level1+h, get_stars(p_dutch_ridge_cnn), ha='center')
    
if p_dutch_cnn_fcnn < 0.05:
    ax = add_sig(plt.gca(),3,3+width,level1)
    plt.text(3+width/2, level1+h, get_stars(p_dutch_cnn_fcnn), ha='center')
    
if p_dutch_ridge_fcnn < 0.05:
    ax = add_sig(plt.gca(),3-width,3+width,level2)
    plt.text(3, level2+h, get_stars(p_dutch_ridge_fcnn), ha='center')

# annotations

handles = [plt.Rectangle((0,0),1,1, facecolor=colors['ridge'], edgecolor='black'),
                plt.Rectangle((0,0),1,1, facecolor=colors['cnn'], edgecolor='black'),
                plt.Rectangle((0,0),1,1, facecolor=colors['fcnn'], edgecolor='black')]

plt.legend(handles, ['Ridge', 'CNN', 'FCNN'], frameon=False)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.axhline(0, color='black', linestyle='dotted', linewidth=1, zorder=-np.inf)

plt.ylabel('Mean reconstruction score')
# plt.xlabel('Listening condition')

plt.xticks([0,1,2,3], ["Dual speakers\n(unattended)", "Background\nbabble noise", "Dual speakers\n(attended)", "Clean speech\n(Dutch)"])
# plt.subplots_adjust(bottom=0.15)

plt.savefig(os.path.join(results_dir, "plots", "test.pdf"))
