
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import auc, roc_auc_score


class NLLLoss(nn.Module):
    def __init__(self, neg_scale_factor=1):
        super(NLLLoss, self).__init__()
        self.neg_scale_factor = neg_scale_factor

    def forward(self, hr, dt, y): # dt = fenetre temporelle (dernier instant ou evenement vu et temps actuel) et hr = hazard rate value
        loss_terms = dt * hr
    
        loss_terms[y == 1] -= hr[y == 1].log()
        loss_terms[y == 0] *= self.neg_scale_factor
        loss = loss_terms.mean()

        assert not loss.isnan()
        assert loss.isfinite()
        return loss
