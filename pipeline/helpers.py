import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
import numpy as np

def mse_loss(predictions, labels):
    return nn.MSELoss()(predictions, labels)
    
def correlation(x, y, eps=1e-8):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    return corr

def variance(outputs):
    return torch.var(outputs)