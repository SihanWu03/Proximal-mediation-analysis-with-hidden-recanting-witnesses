"""Estimator for the bridge function h(W, X)"""

import numpy as np
from minimax import kkt_solve, score_nuisance_function
import bridge_base
import sys

LAMBDA_MIN_FACTOR = 1e-3
LAMBDA_MAX_FACTOR = 1e3
LAMBDA_GRID = 25

GAMMA_MIN = 1e-3
GAMMA_MAX = 1e1
GAMMA_GRID = 25
GAMMA_VALUE = 0.1

REG_GAMMA_MIN = 1e-3
REG_GAMMA_MAX = 1e1

ALPHA_MIN = 1e-8
ALPHA_MAX = 1e1

C_MIN = 1e-1
C_MAX = 1e8

DEGREE_MIN = 1
DEGREE_MAX = 5

MIN_PROP_SCORE = 0.01

class KernelBridgeH0(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function h0(W, M2, X)"""

    def extract_data(self, data):
        """Extract the data for fitting the bridge function."""
        loc = data['a'][:, 0] == 0
        g2 = data['y'][loc, 0]
        g1 = -np.ones(len(g2))
        wx = np.hstack((data['w'], data['m2'], data['x']))
        zx = np.hstack((data['z'], data['m2'], data['x']))
        return g1, g2, wx, zx, loc

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization."""
        g1, g2, wx, zx, loc = self.extract_data(fit_data)

        kh1 = self.kernel1(wx[loc], wx)
        kf1 = self.kernel2(zx[loc], zx)
        kh = self.kernel1(wx, wx)
        kf = self.kernel2(zx, zx)

        alpha, beta = kkt_solve(
            kh1, kf1, kf1, kf1, kh, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = wx.copy()
        self.beta = beta
        self.xf = zx.copy()

    def score(self, val_data):
        """Score the bridge function."""
        g1, g2, wx, zx, loc = self.extract_data(val_data)

        kf1 = self.kernel2(zx[loc], zx)
        kf = self.kernel2(zx, zx)
        try:
            return score_nuisance_function(
                self(wx[loc]),
                kf1,
                kf1,
                kf1,
                kf,
                g1,
                g2,
                LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
            )
        except np.linalg.LinAlgError:
            return np.inf

class KernelBridgeH1(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function h1(W, X)"""

    def __init__(self, lambda1, lambda2, gamma1, gamma2, h0_estimator=None, treatment_prob=None):
        super().__init__(lambda1, lambda2, gamma1, gamma2, treatment_prob)
        self.h0_estimator = h0_estimator
    
    def extract_data(self, data):
        """Extract the data for fitting the bridge function."""
        loc = data['a'][:, 0] == 1
        
        if self.h0_estimator is not None:
            wx_h0 = np.hstack((data['w'], data['m2'], data['x']))
            h0_predictions = self.h0_estimator(wx_h0).flatten()
            g2 = h0_predictions[loc]
        else:
            raise ValueError("h0_estimator is required for KernelBridgeH1")
        
        g1 = -np.ones(len(g2))
        
        wx = np.hstack((data['w'], data['x']))
        zx = np.hstack((data['z'], data['x']))
        
        return g1, g2, wx, zx, loc

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization."""
        g1, g2, wx, zx, loc = self.extract_data(fit_data)

        kh1 = self.kernel1(wx[loc], wx)
        kf1 = self.kernel2(zx[loc], zx)
        kh = self.kernel1(wx, wx)
        kf = self.kernel2(zx, zx)

        alpha, beta = kkt_solve(
            kh1, kf1, kf1, kf1, kh, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = wx.copy()
        self.beta = beta
        self.xf = zx.copy()

    def score(self, val_data):
        """Score the bridge function."""
        g1, g2, wx, zx, loc = self.extract_data(val_data)

        kf1 = self.kernel2(zx[loc], zx)
        kf = self.kernel2(zx, zx)
        try:
            return score_nuisance_function(
                self(wx[loc]),
                kf1,
                kf1,
                kf1,
                kf,
                g1,
                g2,
                LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
            )
        except np.linalg.LinAlgError:
            return np.inf        
        
