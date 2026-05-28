import numpy as np
import bridge_base
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from minimax import kkt_solve, score_nuisance_function
from sklearn.metrics.pairwise import pairwise_kernels
from econml.grf.classes import RegressionForest

LAMBDA_MIN_FACTOR = 1e-3
LAMBDA_MAX_FACTOR = 1e3
LAMBDA_GRID = 25

GAMMA_MIN = 1e-3 # only for 5b: use 1e-4
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

class KernelBridgeQ1(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function Q1(Z, X)"""

    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
        probs = self.treatment_prob.predict_proba(data['x'])
        g1 = (data['a'][:, 0] == 1).astype(float) / probs[:, 1]
        g2 = -(data['a'][:, 0] == 0).astype(float) / probs[:, 0]
        wx = np.hstack((data['w'], data['x']))
        zx = np.hstack((data['z'], data['x']))
        return g1, g2, wx, zx

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        g1, g2, wx, zx = self.extract_data(fit_data)

        kq = self.kernel1(zx, zx)
        kf = self.kernel2(wx, wx)

        alpha, beta = kkt_solve(
            kq, kf, kf, kf, kq, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = zx.copy()
        self.beta = beta
        self.xf = wx.copy()

    def score(self, val_data):
        """Score the bridge function"""
        g1, g2, wx, zx = self.extract_data(val_data)

        kf = self.kernel2(wx, wx)
        try:
            return score_nuisance_function(
                self(zx),
                kf,
                kf,
                kf,
                kf,
                g1,
                g2,
                LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
            )
        except np.linalg.LinAlgError:
            return np.inf
        
class treatment_prob:
    def __init__(self, calibrated=True):
        self.calibrated = calibrated
    
    def fit(self, X, A):
        # Base logistic regression model
        base_model = LogisticRegression(
            penalty='l2',  # L2 regularization to prevent overfitting
            C=1.0,         # Regularization strength
            solver='lbfgs',
            max_iter=1000,
            random_state=1234
        )
        
        if self.calibrated:
            # Use probability calibration (theoretically not required, but beneficial in practice)
            self.model = CalibratedClassifierCV(
                base_model,
                method='sigmoid',  # Use sigmoid calibration
                cv=5
            )
        else:
            self.model = base_model
        
        self.model.fit(X, A.flatten() if A.ndim > 1 else A)
        
    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        # [Critical fix]: Clip probabilities to prevent division by extremely small values
        proba = np.clip(proba, 0.05, 0.95)
        return proba
    
class KernelBridgeQ0(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function Q0(W, M2, X)"""
    
    def __init__(self, lambda1, lambda2, gamma1, gamma2, q1_estimator=None, treatment_prob=None):
        super().__init__(lambda1, lambda2, gamma1, gamma2, treatment_prob)
        self.q1_estimator = q1_estimator  # Save the q1 estimator
    
    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
        # Calculate propensity scores
        probs = self.treatment_prob.predict_proba(data['x'])
        
        # Calculate q1 predictions (using the previously trained q1 estimator)
        if self.q1_estimator is not None:
            # Construct features for q1: Z, X
            zx_q1 = np.hstack((data['z'], data['x']))
            # Predict all samples using the q1 estimator
            q1_predictions = self.q1_estimator(zx_q1)
        else:
            raise ValueError("q1_estimator is required for KernelBridgeQ0")
        
        # Calculate g1 and g2
        # g1 = I(A=0) / P(A=0|X)
        g1 = (data['a'][:, 0] == 0).astype(float) 
        
        # g2 = -I(A=1) * q1(Z,X) / P(A=0|X)
        g2 = -(data['a'][:, 0] == 1).astype(float) * q1_predictions 
        
        # Feature combinations: Q0 uses W, M2, X and Z, M2, X
        wx = np.hstack((data['w'], data['m2'], data['x']))
        zx = np.hstack((data['z'], data['m2'], data['x']))
        
        return g1, g2, wx, zx

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
        g1, g2, wx, zx = self.extract_data(fit_data)

        kq = self.kernel1(zx, zx)
        kf = self.kernel2(wx, wx)

        alpha, beta = kkt_solve(
            kq, kf, kf, kf, kq, kf, g1, g2, self.lambda1, self.lambda2
        )

        self.alpha = alpha
        self.x = zx.copy()
        self.beta = beta
        self.xf = wx.copy()

    def score(self, val_data):
        """Score the bridge function"""
        g1, g2, wx, zx = self.extract_data(val_data)

        kf = self.kernel2(wx, wx)
        try:
            return score_nuisance_function(
                self(zx),
                kf,
                kf,
                kf,
                kf,
                g1,
                g2,
                LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
            )
        except np.linalg.LinAlgError:
            return np.inf

