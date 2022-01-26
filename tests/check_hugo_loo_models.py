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

for participant in range(13):
    x_cnn = np.load(os.path.join(results_path, 'predictions', 'hugo_leave_one_out', f'cnn_predictions_P{participant:02d}.npy'))
    x_fcnn = np.load(os.path.join(results_path, 'predictions', 'hugo_leave_one_out', f'fcnn_predictions_P{participant:02d}.npy'))
    x_ridge = np.load(os.path.join(results_path, 'predictions', 'hugo_leave_one_out', f'ridge_predictions_P{participant:02d}.npy'))
    y = np.load(os.path.join(results_path, 'predictions', 'hugo_subject_specific', 'ground_truth.npy'))

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

#plt.boxplot([np.array(cnn_scores)-np.array(ridge_scores), np.array(cnn_scores)-np.array(ridge_scores)])
plt.boxplot([ridge_scores, cnn_scores, fcnn_scores])
plt.axhline(np.mean(ridge_scores))
plt.axhline(np.mean(cnn_scores))
plt.axhline(np.mean(fcnn_scores))
print(wilcoxon(np.array(cnn_scores),np.array(ridge_scores), alternative='greater'))
print(wilcoxon(np.array(fcnn_scores),np.array(ridge_scores), alternative='greater'))
#plt.axhline(0.2)
plt.savefig('tests/tmp/fig.pdf')