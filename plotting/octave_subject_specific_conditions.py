import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pipeline.helpers import get_scores, add_sig, get_stars
from scipy.stats import ttest_rel

results_dir = "results/0.5-8Hz-090522"
results_path = os.path.join(results_dir, 'predictions', 'octave_subject_specific')

plotting_config = json.load(open("plotting/plotting_config.json", "r"))
colors = plotting_config['colors']
models = plotting_config['models']
english_participants = plotting_config['octave_english_participants']
dutch_participants = plotting_config['octave_dutch_participants']
batch_size = 250
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

        attended_fM_scores = get_scores(attended_fM_ground_truth, fM_predictions, batch_size=batch_size)
        attended_fW_scores = get_scores(attended_fW_ground_truth, fW_predictions, batch_size=batch_size)
        unattended_fM_scores = get_scores(unattended_fM_ground_truth, fM_predictions, batch_size=batch_size)
        unattended_fW_scores = get_scores(unattended_fW_ground_truth, fW_predictions, batch_size=batch_size)
        mb_scores = get_scores(mb_ground_truth, mb_predictions)
    
        conditions_mean_scores['competing-attended'][model] += [np.nanmean(attended_fM_scores), np.nanmean(attended_fW_scores)]
        conditions_mean_scores['competing-unattended'][model] += [np.nanmean(unattended_fM_scores), np.nanmean(unattended_fW_scores)]
        conditions_mean_scores['babble'][model].append(np.nanmean(mb_scores))

        if participant in dutch_participants:
            cleanDutch_predictions = np.load(os.path.join(results_path, f'{model}_predictions_{participant}_cleanDutch.npy'))
            cleanDutch_scores = get_scores(cleanDutch_ground_truth, cleanDutch_predictions, batch_size=batch_size)
            conditions_mean_scores['dutch'][model].append(np.nanmean(cleanDutch_scores))

# collect p-vals

p_attended_competing_ridge_cnn = ttest_rel(conditions_mean_scores['competing-attended']['ridge'], conditions_mean_scores['competing-attended']['cnn'], alternative='less')[1]
p_attended_competing_ridge_fcnn = ttest_rel(conditions_mean_scores['competing-attended']['ridge'], conditions_mean_scores['competing-attended']['fcnn'], alternative='less')[1]
p_attended_competing_cnn_fcnn = ttest_rel(conditions_mean_scores['competing-attended']['cnn'], conditions_mean_scores['competing-attended']['fcnn'], alternative='two-sided')[1]

p_babble_ridge_cnn = ttest_rel(conditions_mean_scores['babble']['ridge'], conditions_mean_scores['babble']['cnn'], alternative='less')[1]
p_babble_ridge_fcnn = ttest_rel(conditions_mean_scores['babble']['ridge'], conditions_mean_scores['babble']['fcnn'], alternative='less')[1]
p_babble_cnn_fcnn = ttest_rel(conditions_mean_scores['babble']['cnn'], conditions_mean_scores['babble']['fcnn'], alternative='two-sided')[1]

p_dutch_ridge_cnn = ttest_rel(conditions_mean_scores['dutch']['ridge'], conditions_mean_scores['dutch']['cnn'], alternative='less')[1]
p_dutch_ridge_fcnn = ttest_rel(conditions_mean_scores['dutch']['ridge'], conditions_mean_scores['dutch']['fcnn'], alternative='less')[1]
p_dutch_cnn_fcnn = ttest_rel(conditions_mean_scores['dutch']['cnn'], conditions_mean_scores['dutch']['fcnn'], alternative='two-sided')[1]

pvals = np.array([p_attended_competing_ridge_cnn, p_attended_competing_ridge_fcnn, p_attended_competing_cnn_fcnn,
         p_babble_ridge_cnn, p_babble_ridge_fcnn, p_babble_cnn_fcnn,
         p_dutch_ridge_cnn, p_dutch_ridge_fcnn, p_dutch_cnn_fcnn])

[p_attended_competing_ridge_cnn, p_attended_competing_ridge_fcnn, p_attended_competing_cnn_fcnn,
         p_babble_ridge_cnn, p_babble_ridge_fcnn, p_babble_cnn_fcnn,
         p_dutch_ridge_cnn, p_dutch_ridge_fcnn, p_dutch_cnn_fcnn] = len(pvals)*pvals#multitest.fdrcorrection(pvals)[1]

# plot data

width = 1/5

cond_order = ['competing-unattended', 'dutch', 'babble', 'competing-attended']

for i, cond in enumerate(cond_order):
    for j, model in enumerate(models):
        bp = plt.boxplot(conditions_mean_scores[cond][model],
                    positions = [i + (j-1)*width],
                    patch_artist=True,
                    boxprops = {'facecolor':colors[model], 'edgecolor':'black'},
                    medianprops={'color':'yellow'},
                    flierprops={'marker':'x'})

# add significances

h = -0.012
level1 = -0.015
level2 = -0.03

#import pdb;pdb.set_trace()
loc = cond_order.index('competing-attended')
if p_attended_competing_ridge_cnn < 0.05:
    ax = add_sig(plt.gca(),loc-width,loc,level1)
    plt.text(loc-width/2, level1+h, get_stars(p_attended_competing_ridge_cnn), ha='center')
    
if p_attended_competing_cnn_fcnn < 0.05:
    ax = add_sig(plt.gca(),loc,loc+width,level1)
    plt.text(loc+width/2, level1+h, get_stars(p_attended_competing_cnn_fcnn), ha='center')
    
if p_attended_competing_ridge_fcnn < 0.05:
    ax = add_sig(plt.gca(),loc-width,loc+width,level2)
    plt.text(loc, level2+h, get_stars(p_attended_competing_ridge_fcnn), ha='center')

loc = cond_order.index('babble')
if p_babble_ridge_cnn < 0.05:
    ax = add_sig(plt.gca(),loc-width,loc,level1)
    plt.text(loc-width/2, level1+h, get_stars(p_babble_ridge_cnn), ha='center')
    
if p_babble_cnn_fcnn < 0.05:
    ax = add_sig(plt.gca(),loc,loc+width,level1)
    plt.text(loc+width/2, level1+h, get_stars(p_babble_cnn_fcnn), ha='center')
    
if p_babble_ridge_fcnn < 0.05:
    ax = add_sig(plt.gca(),loc-width,loc+width,level2)
    plt.text(loc, level2+h, get_stars(p_babble_ridge_fcnn), ha='center')

loc = cond_order.index('dutch')
if p_dutch_ridge_cnn < 0.05:
    ax = add_sig(plt.gca(),loc-width,loc,level1)
    plt.text(loc-width/2, level1+h, get_stars(p_dutch_ridge_cnn), ha='center')
    
if p_dutch_cnn_fcnn < 0.05:
    ax = add_sig(plt.gca(),loc,loc+width,level1)
    plt.text(loc+width/2, level1+h, get_stars(p_dutch_cnn_fcnn), ha='center')
    
if p_dutch_ridge_fcnn < 0.05:
    ax = add_sig(plt.gca(),loc-width,loc+width,level2)
    plt.text(loc, level2+h, get_stars(p_dutch_ridge_fcnn), ha='center')

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

labels = {
    'competing-unattended':"Dual speakers\n(unattended)",
    'competing-attended':"Dual speakers\n(attended)",
    'dutch':"Clean speech\n(Dutch)",
    'babble':"Background\nbabble noise"
}

xticks = [0,1,2,3]
xticklabels = [labels[cond] for cond in cond_order]

plt.xticks(xticks, xticklabels)
# plt.subplots_adjust(bottom=0.15)

plt.title("Subject-specific models in different listening conditions")
plt.savefig(os.path.join(results_dir, "plots", "octave_subject_specific_conditions.pdf"))
