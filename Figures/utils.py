import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from IPython.display import display
from scipy.stats import gaussian_kde
from scipy.stats import wilcoxon, probplot, ranksums, ttest_ind, ttest_1samp
import matplotlib.patches as patches
from matplotlib import rc
from matplotlib.patches import FancyArrowPatch
from statsmodels.stats import multitest
import statsmodels.api as sm

_h_participant_names = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P12', 'P13', 'P14']


def _h_add_pvalue(df, test='notnull'):
    p = []
    s = []
    for participant in range(13):

        x = df.loc[str(participant)]['correlations']
        y = df.loc[str(participant)]['null_correlations']

        if test=='notnull':
            #s_, p_ = wilcoxon(x, y, alternative='greater')
            s_, p_ = ttest_ind(x, y, alternative='two-sided', equal_var=True)
        else:
            s_, p_ = ttest_1samp(x, alternative='greater')
        s.append(s_)
        p.append(p_)

    df["p-value"] = p
    df["statistic"] = s
    
def _h_correct_pvalues(dfs):
    
    p = list(dfs['ridge']['p-value']) + list(dfs['eegnet']['p-value']) + list(dfs['detaillez']['p-value'])
    p = multitest.fdrcorrection(p)[1]
    dfs['ridge']['p-corrected'] = p[:13]
    dfs['eegnet']['p-corrected'] = p[13:13*2]
    dfs['detaillez']['p-corrected'] = p[13*2:13*3]
    for model in ['ridge', 'eegnet', 'detaillez']:
        _h_add_stars(dfs[model])
    
def _h_add_stars(df):
    significances = df["p-corrected"]
    stars = []
    for s in significances:
        if s <= 0.00001:
            stars.append("*****")
        elif s <= 0.0001:
            stars.append("****")
        elif s <= 0.001:
            stars.append("***")
        elif s <= 0.01:
            stars.append("**")
        elif s <= 0.05:
            stars.append("*")
        elif s > 0.05:
            stars.append("")
            
    df["stars"] = stars

def load_hugo_data(study="individuals", prediction_window=250, results_path="/home/mdt20/Code/MLDecoders/results/011121/"):
    
    dfs = {}
    for model in ["ridge", "eegnet", "detaillez"]:
        r = json.load(open(os.path.join(results_path, model, study + "_results_{}".format(prediction_window) + ".json"), "r"))
        dfs[model] = pd.DataFrame.from_dict(r).transpose()
        _h_add_pvalue(dfs[model], test='notnull')
        dfs[model].index = dfs[model].index.rename("participant")
        dfs[model]['names'] = _h_participant_names
        dfs[model]["means"] = [np.mean(x) for x in dfs[model]["correlations"]]
        dfs[model]["medians"] = [np.median(x) for x in dfs[model]["correlations"]]
        dfs[model]["stds"] = [np.std(x) for x in dfs[model]["correlations"]]
        dfs[model]["null_means"] = [np.mean(x) for x in dfs[model]["null_correlations"]]
        dfs[model]["null_medians"] = [np.median(x) for x in dfs[model]["null_correlations"]]
        dfs[model]["null_stds"] = [np.std(x) for x in dfs[model]["null_correlations"]]

        dfs[model]["25%"] = [np.percentile(dfs[model].loc[str(i)]["correlations"], 25) for i in range(13)]
        dfs[model]["75%"] = [np.percentile(dfs[model].loc[str(i)]["correlations"], 75) for i in range(13)]
        dfs[model]["null_95%"] = [np.std(x) for x in dfs[model]["null_correlations"]]
        dfs[model]["null_25%"] = [np.percentile(dfs[model].loc[str(i)]["null_correlations"], 25) for i in range(13)] - dfs[model]["null_means"]
        dfs[model]["null_75%"] = [np.percentile(dfs[model].loc[str(i)]["null_correlations"], 75) for i in range(13)] - dfs[model]["null_means"]

    _h_correct_pvalues(dfs)
    
    
    idxs = dfs["ridge"]["means"].argsort().to_numpy()
    h_idxs=idxs

    for model in ["ridge", "eegnet", "detaillez"]:
        dfs[model] = dfs[model].loc[dfs[model].index[idxs]]
    
    return dfs

####################

def _o_get_participant_names(condition, results_path, prediction_window):
    if 'Dutch' in condition:
        participant_names = ["YH07", "YH08", "YH09", "YH10", "YH11", "YH12", "YH13","YH14", "YH15", "YH16", "YH17", "YH18", "YH19", "YH20"]
        data_path = os.path.join(results_path, f"{prediction_window}_dutch_correlations.json")
    else:
        participant_names = ["YH00", "YH01", "YH02", "YH03", "YH06", "YH07", "YH08", "YH09", "YH10", "YH11", "YH14", "YH15", "YH16", "YH17", "YH18", "YH19", "YH20"]
        data_path = os.path.join(results_path, f"{prediction_window}_octave_correlations.json")
    return participant_names, data_path

def get_results_octave(prediction_window, condition, results_path):
    
    names, data_path = _o_get_participant_names(condition, results_path, prediction_window)
    r = json.load(open(data_path, 'r'))[condition]
    df = pd.DataFrame.from_dict(r).transpose()
    df.columns = [condition+"_"+col for col in df.columns]
    
    for col in df.columns:
        for p in df.index:
            df.loc[p][col] = np.nan_to_num(np.array(df.loc[p][col]).flatten())
    
    _o_add_significance(df)
    _o_correct_pvalue(df)
    _o_add_stars(df)
    _o_add_means_stds(df)
    
    idxs = df[f"{condition}_ridge_attended_mean"].argsort().to_numpy()
    index0 = np.array([df.index[i][-2:] for i in range(df.index.size)])
    df = df.loc[df.index[idxs]]
    
    return df, index0

def _o_add_significance(df):
    condition = df.columns[0].split("_")[0]
    
    for model in ['ridge', 'eegnet', 'detaillez']:
        p_att = []
        s_att = []
        p_unatt = []
        s_unatt = []
        for participant in df.index:
            x_att = df.loc[participant][f"{condition}_{model}_attended"]
            y_att = df.loc[participant][f"{condition}_{model}_attended_null"]
            x_unatt = df.loc[participant][f"{condition}_{model}_unattended"]
            y_unatt = df.loc[participant][f"{condition}_{model}_unattended_null"]
            
            s_att_, p_att_ = ttest_ind(x_att,y_att, alternative='greater', nan_policy='omit')
            
            if condition=='clean' or condition=='cleanDutch':
                s_unatt_, p_unatt_ = 1, 1
            else:
                s_unatt_, p_unatt_ = ttest_ind(x_unatt,y_unatt, alternative='greater', nan_policy='omit')
            
            p_att.append(p_att_)
            s_att.append(s_att_)
            p_unatt.append(p_unatt_)
            s_unatt.append(s_unatt_)
            
        df[f"p-value (attended, {model})"] = p_att
        df[f"statistic (attended, {model})"] = s_att
        df[f"p-value (unattended, {model})"] = p_unatt
        df[f"statistic (unattended, {model})"] = s_unatt
        
def _o_correct_pvalue(df):
    condition = df.columns[0].split("_")[0]
    
    for target in ['attended', 'unattended']:
        p_vals = []
        for model in ['ridge', 'eegnet', 'detaillez']:
            col = f'p-value ({target}, {model})'
            p_vals += list(df[col])
        p_vals = multitest.fdrcorrection(p_vals)[1]
        for i, model in enumerate(['ridge', 'eegnet', 'detaillez']):
            col = f'p-corrected ({target}, {model})'
            df[col] = p_vals[i*len(df.index):(i+1)*len(df.index)]
    
#     num_test = len(df.index)
#     cols = [x for x in df.columns if 'p-value' in x]
#     for col in cols:
#         corrected_pvalues = df[col]*num_test
#         df[col.replace("p-value", "bonferroni")] = corrected_pvalues
        
        
def _o_add_stars(df):
    
    for col in [x for x in df.columns if 'p-corrected' in x]:
        significances = df[col]
        stars = []
        for s in significances:
            if s <= 0.00001:
                stars.append("*****")
            elif s <= 0.0001:
                stars.append("****")
            elif s <= 0.001:
                stars.append("***")
            elif s <= 0.01:
                stars.append("**")
            elif s <= 0.05:
                stars.append("*")
            elif s > 0.05:
                stars.append("")
            
        df[col.replace("p-corrected", "stars")] = stars
        
def _o_add_means_stds(df):
    for model in ["ridge", "detaillez", "eegnet"]:
        for column in df.columns:
            if f"_{model}_" in column:
                df[column+"_mean"] = [np.mean(x) for x in df[column]]
                df[column+"_std"] = [np.std(x) for x in df[column]]
                df[column+"_25%"] = [np.percentile(x, 25) for x in df[column]]
                df[column+"_75%"] = [np.percentile(x, 75) for x in df[column]]