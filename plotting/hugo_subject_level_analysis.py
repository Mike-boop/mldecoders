import numpy as np
import json
import os
from pipeline.helpers import get_scores
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats import multitest

np.random.seed(0)
study = 'hugo_leave_one_out'

plotting_config = json.load(open("plotting/plotting_config.json", "r"))
colors = plotting_config['colors']
models = plotting_config['models']
results_dir = "results/0.5-8Hz-090522"

path = os.path.join(results_dir, 'predictions', study)

if not os.path.exists(os.path.join(results_dir, 'p_vals')):
    os.mkdir(os.path.join(results_dir, 'p_vals'))

# load scores & order by ridge mean

ridge_predictions = [np.load(os.path.join(path, f"ridge_predictions_P{participant:02d}.npy")) for participant in range(13)]
cnn_predictions = [np.load(os.path.join(path, f"cnn_predictions_P{participant:02d}.npy")) for participant in range(13)]
fcnn_predictions = [np.load(os.path.join(path, f"fcnn_predictions_P{participant:02d}.npy")) for participant in range(13)]
ground_truth = np.load(os.path.join("results/0.5-8Hz", "predictions", "hugo_subject_specific", "ground_truth.npy"))

scores = {
    'ridge':[get_scores(pred, ground_truth, batch_size=250) for pred in ridge_predictions],
    'cnn'  :[get_scores(pred, ground_truth, batch_size=250) for pred in cnn_predictions],
    'fcnn' :[get_scores(pred, ground_truth, batch_size=250) for pred in fcnn_predictions]
}

null_scores = {
    'ridge':[get_scores(pred, ground_truth, batch_size=250, null=True) for pred in ridge_predictions],
    'cnn'  :[get_scores(pred, ground_truth, batch_size=250, null=True) for pred in cnn_predictions],
    'fcnn' :[get_scores(pred, ground_truth, batch_size=250, null=True) for pred in fcnn_predictions]
}

ridge_means = [np.mean(score) for score in scores['ridge']]
sorted_idx = np.argsort(ridge_means)

# test whether model scores are significantly different to null distributions

not_null_p_vals = []

for model in models:
    for participant_idx in sorted_idx:
        not_null_p_vals.append(
            ttest_ind(scores[model][participant_idx], null_scores[model][participant_idx], alternative='greater')[1]
            )

p_vals = multitest.fdrcorrection(not_null_p_vals)[1]
p_ridge_null = p_vals[:13]
p_cnn_null = p_vals[13:13*2]
p_fcnn_null = p_vals[13*2:13*3]

# compare models on the subject level

p_ridge_cnn = []
p_ridge_fcnn = []
p_cnn_fcnn = []

for participant_idx in sorted_idx:
    p_ridge_cnn.append(
        ttest_rel(scores['cnn'][participant_idx], scores['ridge'][participant_idx], alternative='greater')[1]
        )
    p_ridge_fcnn.append(
        ttest_rel(scores['fcnn'][participant_idx], scores['ridge'][participant_idx], alternative='greater')[1]
        )
    p_cnn_fcnn.append(
        ttest_rel(scores['fcnn'][participant_idx], scores['cnn'][participant_idx], alternative='two-sided')[1]
        )

p_vals = multitest.fdrcorrection(p_ridge_cnn + p_ridge_fcnn + p_cnn_fcnn)[1]
p_ridge_cnn = p_vals[:13]
p_ridge_fcnn = p_vals[13:13*2]
p_cnn_fcnn = p_vals[13*2:13*3]

# write results to file

def format_pvalues(p_vals):
    formatted_p_vals = []
    for p_val in p_vals:
        if p_val < 0.05:
            formatted_p_vals.append(f"\\textbf{{{p_val:.2e}}}")
        else:
            formatted_p_vals.append(f"{p_val:.2e}")
    return " & ".join(formatted_p_vals) + " \\\\"

with open(os.path.join(results_dir, 'p_vals', f'{study}.txt'), 'w') as f:
    f.writelines('\nridge not null\n')
    f.writelines(format_pvalues(p_ridge_null))
    f.writelines('\ncnn not null\n')
    f.writelines(format_pvalues(p_cnn_null))
    f.writelines('\nfcnn not null\n')
    f.writelines(format_pvalues(p_fcnn_null))
    f.writelines('\nridge vs cnn\n')
    f.writelines(format_pvalues(p_ridge_cnn))
    f.writelines('\nridge vs fcnn\n')
    f.writelines(format_pvalues(p_ridge_fcnn))
    f.writelines('\ncnn vs fcnn\n')
    f.writelines(format_pvalues(p_cnn_fcnn))

# population analysis

print('population-level-analysis')
mean_scores = {k:[np.mean(scores[k][i]) for i in range(13)] for k in scores.keys()}

p_ridge_cnn = ttest_rel(mean_scores['ridge'], mean_scores['cnn'], alternative='less')[1]
p_ridge_fcnn = ttest_rel(mean_scores['ridge'], mean_scores['fcnn'], alternative='less')[1]
p_fcnn_cnn = ttest_rel(mean_scores['fcnn'], mean_scores['cnn'], alternative='two-sided')[1]

[p_ridge_cnn, p_ridge_fcnn, p_fcnn_cnn] = np.array([p_ridge_cnn, p_ridge_fcnn, p_fcnn_cnn])*3
print([p_ridge_cnn, p_ridge_fcnn, p_fcnn_cnn])

# ratio of means

for window_size in [250, 625, 1250]:
    scores = {
    'ridge':np.mean([get_scores(pred, ground_truth, batch_size=window_size) for pred in ridge_predictions]),
    'cnn'  :np.mean([get_scores(pred, ground_truth, batch_size=window_size) for pred in cnn_predictions]),
    'fcnn' :np.mean([get_scores(pred, ground_truth, batch_size=window_size) for pred in fcnn_predictions])
    }
    print(window_size, (scores['cnn']-scores['ridge'])/scores['ridge'], (scores['fcnn']-scores['ridge'])/scores['ridge'])