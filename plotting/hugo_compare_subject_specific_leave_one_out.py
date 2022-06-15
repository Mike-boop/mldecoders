import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pipeline.helpers import get_scores, add_sig, get_stars
from scipy.stats import ttest_rel
import pandas as pd

plotting_config = json.load(open("plotting/plotting_config.json", "r"))
colors = plotting_config['colors']
models = plotting_config['models']
results_dir = "results/0.5-8Hz-090522"

# 1. Load data

path_ss = os.path.join(results_dir, 'predictions', 'hugo_subject_specific')
path_loo = os.path.join(results_dir, 'predictions', 'hugo_leave_one_out')
path_pop = os.path.join(results_dir, 'predictions', 'hugo_population')

ground_truth = np.load(os.path.join(path_ss, "ground_truth.npy"))

predictions_ss = pd.DataFrame.from_dict({
    (mdl, participant):
        np.load(os.path.join(path_ss, f"{mdl}_predictions_P{participant:02d}.npy")) for participant in range(13) for mdl in models
})

predictions_loo = pd.DataFrame.from_dict({
    (mdl, participant):
        np.load(os.path.join(path_loo, f"{mdl}_predictions_P{participant:02d}.npy")) for participant in range(13) for mdl in models
})

predictions_pop = pd.DataFrame.from_dict({
    (mdl, participant):
        np.load(os.path.join(path_pop, f"{mdl}_predictions_P{participant:02d}.npy")) for participant in range(13) for mdl in models
})

mean_scores_ss = predictions_ss.apply(lambda x: get_scores(x, ground_truth, batch_size=250)).mean()
mean_scores_loo = predictions_loo.apply(lambda x: get_scores(x, ground_truth, batch_size=250)).mean()
mean_scores_pop = predictions_pop.apply(lambda x: get_scores(x, ground_truth, batch_size=250)).mean()

# 2. calculate p-vals

p_ridge_cnn_ss = ttest_rel(mean_scores_ss['ridge'], mean_scores_ss['cnn'], alternative='less')[1]
p_ridge_fcnn_ss = ttest_rel(mean_scores_ss['ridge'], mean_scores_ss['fcnn'], alternative='less')[1]
p_cnn_fcnn_ss = ttest_rel(mean_scores_ss['cnn'], mean_scores_ss['fcnn'], alternative='two-sided')[1]

p_ridge_cnn_loo = ttest_rel(mean_scores_loo['ridge'], mean_scores_loo['cnn'], alternative='less')[1]
p_ridge_fcnn_loo = ttest_rel(mean_scores_loo['ridge'], mean_scores_loo['fcnn'], alternative='less')[1]
p_cnn_fcnn_loo = ttest_rel(mean_scores_loo['cnn'], mean_scores_loo['fcnn'], alternative='two-sided')[1]

p_ridge_cnn_pop = ttest_rel(mean_scores_pop['ridge'], mean_scores_pop['cnn'], alternative='less')[1]
p_ridge_fcnn_pop = ttest_rel(mean_scores_pop['ridge'], mean_scores_pop['fcnn'], alternative='less')[1]
p_cnn_fcnn_pop = ttest_rel(mean_scores_pop['cnn'], mean_scores_pop['fcnn'], alternative='two-sided')[1]

[p_ridge_cnn_ss, p_ridge_fcnn_ss, p_cnn_fcnn_ss,
p_ridge_cnn_loo, p_ridge_fcnn_loo, p_cnn_fcnn_loo,
p_ridge_cnn_pop, p_ridge_fcnn_pop, p_cnn_fcnn_pop] =\
    np.array([p_ridge_cnn_ss, p_ridge_fcnn_ss, p_cnn_fcnn_ss,
            p_ridge_cnn_loo, p_ridge_fcnn_loo, p_cnn_fcnn_loo,
            p_ridge_cnn_pop, p_ridge_fcnn_pop, p_cnn_fcnn_pop])*13

# 3. make plot

bplot1 = plt.boxplot([mean_scores_ss[mdl] for mdl in models],
                    positions=[1,2,3],
                    widths=0.7,
                    patch_artist=True,
                    medianprops={'color':'yellow'},
                    flierprops={'marker':'x'},
                    showfliers=True)

bplot2 = plt.boxplot([mean_scores_loo[mdl] for mdl in models],
                    positions=[5,6,7],
                    widths=0.7,
                    patch_artist=True,
                    medianprops={'color':'yellow'},
                    flierprops={'marker':'x'},
                    showfliers=True)

bplot3 = plt.boxplot([mean_scores_pop[mdl] for mdl in models],
                    positions=[9,10,11],
                    widths=0.7,
                    patch_artist=True,
                    medianprops={'color':'yellow'},
                    flierprops={'marker':'x'},
                    showfliers=True)

for patch, color in zip(bplot1['boxes'], list(colors.values())):
    patch.set_facecolor(color)
for patch, color in zip(bplot2['boxes'], list(colors.values())):
    patch.set_facecolor(color)
for patch, color in zip(bplot3['boxes'], list(colors.values())):
    patch.set_facecolor(color)

# 4. add significances

h = -0.02

if p_ridge_cnn_ss < 0.05:
    y = -0.025
    ax = add_sig(plt.gca(),1,2,y)
    plt.text(1.5, y+h, get_stars(p_ridge_cnn_ss), ha='center')
    
if p_cnn_fcnn_ss < 0.05:
    y = -0.025
    ax = add_sig(plt.gca(),2,3,y)
    plt.text(2.5, y+h, get_stars(p_cnn_fcnn_ss), ha='center')
    
if p_ridge_fcnn_ss < 0.05:
    y = -0.05
    ax = add_sig(plt.gca(),1,3,y)
    plt.text(2, y+h, get_stars(p_ridge_fcnn_ss), ha='center')
    
if p_ridge_cnn_loo < 0.05:
    y = -0.025
    ax = add_sig(plt.gca(),5,6,y)
    plt.text(5.5, y+h, get_stars(p_ridge_cnn_loo), ha='center')
    
if p_cnn_fcnn_loo < 0.05:
    y = -0.025
    ax = add_sig(plt.gca(),6,7,y)
    plt.text(6.5, y+h, get_stars(p_cnn_fcnn_loo), ha='center')
    
if p_ridge_fcnn_loo < 0.05:
    y = -0.05
    ax = add_sig(plt.gca(),5,7,y)
    plt.text(6, y+h, get_stars(p_ridge_fcnn_loo), ha='center')

if p_ridge_cnn_pop < 0.05:
    y = -0.025
    ax = add_sig(plt.gca(),9,10,y)
    plt.text(9.5, y+h, get_stars(p_ridge_cnn_loo), ha='center')
    
if p_cnn_fcnn_pop < 0.05:
    y = -0.025
    ax = add_sig(plt.gca(),10,11,y)
    plt.text(10.5, y+h, get_stars(p_cnn_fcnn_loo), ha='center')
    
if p_ridge_fcnn_pop < 0.05:
    y = -0.05
    ax = add_sig(plt.gca(),9,11,y)
    plt.text(10, y+h, get_stars(p_ridge_fcnn_loo), ha='center')

# 5. annotations & style

plt.title("Comparison of subject-specific, subject-independent,\nand population decoders")
plt.ylabel("Mean reconstruction score")

plt.xticks([2,6, 10], ["Subject-specific\nmodels", "Leave-one-out\nexperiment", "Population\nmodels"])

handles = [plt.Rectangle((0,0),1,1, facecolor=colors['ridge'], edgecolor='black'),
                plt.Rectangle((0,0),1,1, facecolor=colors['cnn'], edgecolor='black'),
                plt.Rectangle((0,0),1,1, facecolor=colors['fcnn'], edgecolor='black')]

plt.legend(handles, ['Ridge', 'CNN', 'FCNN'], loc='upper right', frameon=False)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.ylim(-0.08, 0.38)

plt.axhline(0, color='black', linestyle='dotted', linewidth=1, zorder=-np.inf)
plt.gca().set_axisbelow(True)

plt.savefig(os.path.join(results_dir, "plots", "subject-specific-loo-population-comparison.pdf"))

for model in ['ridge', 'cnn', 'fcnn']:
    print(model, np.mean(mean_scores_loo[model])/np.mean(mean_scores_ss[model]))