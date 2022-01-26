import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json
import os
from pipeline.helpers import get_scores, add_sig, get_stars
from scipy.stats import wilcoxon
from statsmodels.stats import multitest

plotting_config = json.load(open("plotting/plotting_config.json", "r"))
colors = plotting_config['colors']
models = plotting_config['models']
results_dir = "results/0.5-8Hz"

# 1. Load data

path_ss = os.path.join(results_dir, 'predictions', 'hugo_subject_specific')
path_loo = os.path.join(results_dir, 'predictions', 'hugo_leave_one_out')

ground_truth = np.load(os.path.join(path_ss, "ground_truth.npy"))

ridge_predictions_ss = [np.load(os.path.join(path_ss, f"ridge_predictions_P{participant:02d}.npy")) for participant in range(13)]
cnn_predictions_ss = [np.load(os.path.join(path_ss, f"cnn_predictions_P{participant:02d}.npy")) for participant in range(13)]
fcnn_predictions_ss = [np.load(os.path.join(path_ss, f"fcnn_predictions_P{participant:02d}.npy")) for participant in range(13)]

ridge_predictions_loo = [np.load(os.path.join(path_loo, f"ridge_predictions_P{participant:02d}.npy")) for participant in range(13)]
cnn_predictions_loo = [np.load(os.path.join(path_loo, f"cnn_predictions_P{participant:02d}.npy")) for participant in range(13)]
fcnn_predictions_loo = [np.load(os.path.join(path_loo, f"fcnn_predictions_P{participant:02d}.npy")) for participant in range(13)]

mean_scores_ss = {
    'ridge':[np.mean(get_scores(pred, ground_truth, batch_size=250)) for pred in ridge_predictions_ss],
    'cnn'  :[np.mean(get_scores(pred, ground_truth, batch_size=250)) for pred in cnn_predictions_ss],
    'fcnn' :[np.mean(get_scores(pred, ground_truth, batch_size=250)) for pred in fcnn_predictions_ss]
}

mean_scores_loo = {
    'ridge':[np.mean(get_scores(pred, ground_truth, batch_size=250)) for pred in ridge_predictions_loo],
    'cnn'  :[np.mean(get_scores(pred, ground_truth, batch_size=250)) for pred in cnn_predictions_loo],
    'fcnn' :[np.mean(get_scores(pred, ground_truth, batch_size=250)) for pred in fcnn_predictions_loo]
}

# 2. calculate p-vals

p_ridge_cnn_ss = wilcoxon(mean_scores_ss['ridge'], mean_scores_ss['cnn'], alternative='less')[1]
p_ridge_fcnn_ss = wilcoxon(mean_scores_ss['ridge'], mean_scores_ss['fcnn'], alternative='less')[1]
p_cnn_fcnn_ss = wilcoxon(mean_scores_ss['cnn'], mean_scores_ss['fcnn'], alternative='two-sided')[1]

p_ridge_cnn_loo = wilcoxon(mean_scores_loo['ridge'], mean_scores_loo['cnn'], alternative='less')[1]
p_ridge_fcnn_loo = wilcoxon(mean_scores_loo['ridge'], mean_scores_loo['fcnn'], alternative='less')[1]
p_cnn_fcnn_loo = wilcoxon(mean_scores_loo['cnn'], mean_scores_loo['fcnn'], alternative='two-sided')[1]

[p_ridge_cnn_ss, p_ridge_fcnn_ss, p_cnn_fcnn_ss, p_ridge_cnn_loo, p_ridge_fcnn_loo, p_cnn_fcnn_loo] =\
    multitest.fdrcorrection(
        [p_ridge_cnn_ss, p_ridge_fcnn_ss, p_cnn_fcnn_ss, p_ridge_cnn_loo, p_ridge_fcnn_loo, p_cnn_fcnn_loo]
    )[1]

# 3. make plot

bplot1 = plt.boxplot(mean_scores_ss.values(),
                    positions=[1,2,3],
                    widths=0.7,
                    patch_artist=True,
                    medianprops={'color':'yellow'},
                    flierprops={'marker':'x'},
                    showfliers=True);

bplot2 = plt.boxplot(mean_scores_loo.values(),
                    positions=[5,6,7],
                    widths=0.7,
                    patch_artist=True,
                    medianprops={'color':'yellow'},
                    flierprops={'marker':'x'},
                    showfliers=True)

for patch, color in zip(bplot1['boxes'], list(colors.values())):
    patch.set_facecolor(color)
for patch, color in zip(bplot2['boxes'], list(colors.values())):
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

# 5. annotations & style

plt.title("Comparison of subject-specific and pre-trained decoders")
plt.ylabel("Mean reconstruction score")

plt.xticks([2,6, 10], ["Subject-specific\nmodels", "Leave-one-out\nexperiment", "Pre-trained\nmodels"])

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

plt.savefig(os.path.join(results_dir, "plots", "test.pdf"))