import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pipeline.helpers import get_scores, bitrate

results_dir = "results/0.5-8Hz-090522"
results_path = os.path.join(results_dir, 'predictions', 'octave_subject_specific')

plotting_config = json.load(open("plotting/plotting_config.json", "r"))
colors = plotting_config['colors']
models = plotting_config['models']
english_participants = plotting_config['octave_english_participants']
dutch_participants = plotting_config['octave_dutch_participants']

correlation_windows=np.array([62, 125, 250, 625, 1250])
scores = {'ridge':[], 'cnn':[], 'fcnn':[]}
bitrates = {'ridge':[], 'cnn':[], 'fcnn':[]}

for correlation_window in correlation_windows:

    # collect mean scores

    conditions_mean_scores = {
        'competing-attended':{'ridge':[], 'cnn':[], 'fcnn':[]},
        'competing-unattended':{'ridge':[], 'cnn':[], 'fcnn':[]},
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

    for model in models:
        scores[model].append(fM_scores[model]+fW_scores[model])

for model in models:
    bitrates[model] = [bitrate(np.mean(scores[model][i]),window=correlation_windows[i]/125 + 0.4) for i in range(len(correlation_windows))]

labels = {'ridge':'Ridge', 'cnn':'CNN', 'fcnn':'FCNN'}

for model in models:
    plt.plot(correlation_windows/125, bitrates[model], color=colors[model], label=labels[model])
    plt.scatter(correlation_windows/125, bitrates[model], color=colors[model])

plt.xscale('log')
plt.xticks(correlation_windows/125, np.round(correlation_windows/125,1))
plt.minorticks_off()

plt.title("Information transfer rate vs window size")
plt.xlabel("Window size [s]")
plt.ylabel("Bitrate [bits/min]")
plt.legend(frameon=False)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.savefig(os.path.join(results_dir, "plots", "octave_subject_specific_decoding_bitrate.pdf"))

plt.figure()

width = 0.6

for i, correlation_window in enumerate(correlation_windows[2:]):
    for j, model in enumerate(models):
        bp = plt.boxplot(scores[model][2+i],
                    positions = [3*i+(j-1)*width*1.2],
                    patch_artist=True,
                    boxprops = {'facecolor':colors[model], 'edgecolor':'black'},
                    medianprops={'color':'yellow'},
                    flierprops={'marker':'x'},
                    widths=width)

plt.title("Attention decoding accuracy vs window size")
plt.xticks([0, 3, 6], ['2 s window', '5 s window', '10 s window'])
plt.ylabel("Decoding accuracy")

handles = [plt.Rectangle((0,0),1,1, facecolor=colors['ridge'], edgecolor='black'),
                plt.Rectangle((0,0),1,1, facecolor=colors['cnn'], edgecolor='black'),
                plt.Rectangle((0,0),1,1, facecolor=colors['fcnn'], edgecolor='black')]

plt.legend(handles, ['Ridge', 'CNN', 'FCNN'], frameon=False)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.ylim(0.48, 1.02)
plt.savefig(os.path.join(results_dir, "plots", "octave_subject_specific_decoding_pooled.pdf"))

plt.show()
