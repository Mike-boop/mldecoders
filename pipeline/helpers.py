import torch
import numpy as np
from scipy.stats import pearsonr
    
def correlation(x, y, eps=1e-8):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    return corr

def get_scores(y, y_hat, batch_size=1250, sliding=False, sliding_step=1, null=False):

    if sliding:
        scores = []
        for i in range(0, y.size-batch_size, sliding_step):
            scores.append(pearsonr(y[i:i+batch_size], y_hat[i:i+batch_size])[0])
        return np.array(scores)

    batches = y.size//batch_size - 1
    scores = []

    idxs = np.arange(batches)
    if null:
        idxs = np.random.choice(idxs, idxs.size, replace=False)

    for i in range(batches):
        scores.append(pearsonr(y[idxs[i]*batch_size:(idxs[i]+1)*batch_size], y_hat[i*batch_size:(i+1)*batch_size])[0])
    return np.array(scores)

def add_sig(ax, x1, x2, y, headwidth=0.01, headpos='top', width=1):
    ax.plot([x1,x2], [y,y], color='black', lw=width)
    if headpos=='top':
        ax.plot([x1,x1], [y,y+headwidth], color='black', lw=width)
        ax.plot([x2,x2], [y,y+headwidth], color='black', lw=width)
    if headpos=='bottom':
        ax.plot([x1,x1], [y,y-headwidth], color='black', lw=width)
        ax.plot([x2,x2], [y,y-headwidth], color='black', lw=width)
        
    return ax

def get_stars(p):

    if p <= 0.00001:
        return "*****"
    elif p <= 0.0001:
        return "****"
    elif p <= 0.001:
        return "***"
    elif p <= 0.01:
        return "**"
    elif p <= 0.05:
        return "*"
    elif p > 0.05:
        return ""

def bitrate(accuracy, classes=2, window=1):
    bitrate = np.log2(classes) + accuracy*np.log2(accuracy) + (1-accuracy)*np.log2((1-accuracy)/(classes-1))
    return bitrate*60/window #bits/min