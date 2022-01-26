import numpy as np
import pathlib
import os
from scipy.stats import pearsonr, wilcoxon, ranksums
import matplotlib.pyplot as plt

results_path = pathlib.Path('results/0.5-8Hz')

def get_scores(y, y_hat, batch_size=1250, sliding=False, sliding_step=1):

    if sliding:
        scores = []
        for i in range(0, y.size-batch_size, sliding_step):
            scores.append(pearsonr(y[i:i+batch_size], y_hat[i:i+batch_size])[0])
        return np.array(scores)

    batches = y.size//batch_size - 1
    scores = []
    for i in range(batches):
        scores.append(pearsonr(y[i*batch_size:(i+1)*batch_size], y_hat[i*batch_size:(i+1)*batch_size])[0])
    return np.array(scores)

ridge_scores = []
cnn_scores = []
fcnn_scores = []

for participant in ["YH00", "YH01", "YH02", "YH03", "YH06",
                        "YH07", "YH08", "YH09", "YH10", "YH11",
                        "YH14", "YH15", "YH16", "YH17", "YH18", 
                        "YH19", "YH20"]:
    try:
        x_cnn_fM = np.load(os.path.join(results_path, 'predictions', 'octave_subject_specific', f'cnn_predictions_{participant}_fM.npy'))
        x_fcnn_fM = np.load(os.path.join(results_path, 'predictions', 'octave_subject_specific', f'fcnn_predictions_{participant}_fM.npy'))
        x_ridge_fM = np.load(os.path.join(results_path, 'predictions', 'octave_subject_specific', f'ridge_predictions_{participant}_fM.npy'))
        y_fM = np.load(os.path.join(results_path, 'predictions', 'octave_subject_specific', f'attended_ground_truth_{participant}_fM.npy'))

        x_cnn_fW = np.load(os.path.join(results_path, 'predictions', 'octave_subject_specific', f'cnn_predictions_{participant}_fW.npy'))
        x_fcnn_fW = np.load(os.path.join(results_path, 'predictions', 'octave_subject_specific', f'fcnn_predictions_{participant}_fW.npy'))
        x_ridge_fW = np.load(os.path.join(results_path, 'predictions', 'octave_subject_specific', f'ridge_predictions_{participant}_fW.npy'))
        y_fW = np.load(os.path.join(results_path, 'predictions', 'octave_subject_specific', f'attended_ground_truth_{participant}_fW.npy'))

        x_cnn = np.hstack([x_cnn_fM, x_cnn_fW])
        x_fcnn = np.hstack([x_fcnn_fM, x_fcnn_fW])
        x_ridge = np.hstack([x_ridge_fM, x_ridge_fW])
        y = np.hstack([y_fM, y_fW])

        print(
            pearsonr(x_ridge, y)[0],
            f'({np.mean(get_scores(x_ridge, y))})\n',
            pearsonr(x_cnn, y)[0],
            f'({np.mean(get_scores(x_cnn, y))})\n',
            pearsonr(x_fcnn, y)[0],
            f'({np.mean(get_scores(x_fcnn, y))})\n'
        )

        ridge_scores.append(np.mean(get_scores(x_ridge, y)))
        cnn_scores.append(np.mean(get_scores(x_cnn, y)))
        fcnn_scores.append(np.mean(get_scores(x_fcnn, y)))
    except FileNotFoundError:
        continue

#plt.boxplot([np.array(cnn_scores)-np.array(ridge_scores), np.array(cnn_scores)-np.array(ridge_scores)])
plt.boxplot([ridge_scores, cnn_scores, fcnn_scores])
plt.axhline(np.mean(ridge_scores))
plt.axhline(np.mean(cnn_scores))
plt.axhline(np.mean(fcnn_scores))
print(wilcoxon(np.array(cnn_scores),np.array(ridge_scores), alternative='greater'))
print(wilcoxon(np.array(fcnn_scores),np.array(ridge_scores), alternative='greater'))
#plt.axhline(0.2)
plt.savefig('tests/tmp/fig.pdf')