from __future__ import absolute_import

import numpy as np
import math
import torch
from torch.nn.parameter import Parameter
from metric_learn.base_metric import BaseMetricLearner


def validate_cov_matrix(M):
    M = (M + M.T) * 0.5
    k = 0
    I = np.eye(M.shape[0])
    while True:
        try:
            _ = np.linalg.cholesky(M)
            break
        except np.linalg.LinAlgError:
            # Find the nearest positive definite matrix for M. Modified from
            # http://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
            # Might take several minutes
            k += 1
            w, v = np.linalg.eig(M)
            min_eig = v.min()
            M += (-min_eig * k * k + np.spacing(min_eig)) * I
    return M


class TMM(BaseMetricLearner):
    def __init__(self, metric_triplet):
        self.metric_triplet = metric_triplet
        self.M_ = None

    def metric(self):
        return self.M_

    def fit(self):
        if torch.cuda.is_available():
            self.M_ = self.metric_triplet.M_.cpu().detach().numpy()
        else:
            self.M_ = self.metric_triplet.M_.numpy()
        self.M_ = validate_cov_matrix(self.M_)

