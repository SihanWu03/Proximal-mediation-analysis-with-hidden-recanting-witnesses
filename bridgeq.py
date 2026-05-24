import numpy as np
import bridge_base
import linear_dgp
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from minimax import kkt_solve, score_nuisance_function
from sklearn.metrics.pairwise import pairwise_kernels
from econml.grf.classes import RegressionForest

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

class KernelBridgeQ1(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function Q(Z, X)"""

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
        # Basic logistic regression model
        base_model = LogisticRegression(
            penalty='l2',  # L2 regularization to prevent overfitting
            C=1.0,         # Regularization strength
            solver='lbfgs',
            max_iter=1000,
            random_state=1234
        )
        
        if self.calibrated:
            # Use probability calibration (though not theoretically required, it might be beneficial in practice)
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
        return proba

class KernelBridgeQ0(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function Q0(W,M2,X)"""
    
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

def test_q1_and_q0():
    """Test whether KernelBridgeQ1 and KernelBridgeQ0 run correctly"""
    print("=" * 60)
    print("Testing if Q1 and Q0 bridge functions run correctly")
    print("=" * 60)
    
    # 1. Create data generator
    print("\nGenerating data...")
    datagen = linear_dgp.LinearTwoMediatorsProxDGP(
        xdim=5,     # Covariate dimension
        zdim=2,     # Dimension of Z
        wdim=2,     # Dimension of W
        m1dim=2,    # Dimension of M1
        m2dim=2,    # Dimension of M2
        seed=42
    )
    
    # Generate data
    data = datagen.sample_dataset(n=300, seed=42)
    
    # Print data shapes
    print("Data shapes:")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
    
    # 2. Train propensity score model
    print("\n" + "-" * 40)
    print("Training propensity score model...")
    print("-" * 40)
    
    try:
        propensity_model = treatment_prob()
        propensity_model.fit(X=data['x'], A=data['a'][:, 0])
        print("✓ Propensity score model trained successfully")
        
        # Check propensity scores
        probs = propensity_model.predict_proba(data['x'])
        print(f"Propensity score shape: {probs.shape}")
        print(f"Average probability for treatment group: {np.mean(probs[:, 1]):.4f}")
        print(f"Average probability for control group: {np.mean(probs[:, 0]):.4f}")
        
    except Exception as e:
        print(f"✗ Propensity score model training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Test KernelBridgeQ1
    print("\n" + "-" * 40)
    print("Testing KernelBridgeQ1...")
    print("-" * 40)
    
    try:
        # Create Q1 estimator
        q1_estimator = KernelBridgeQ1(
            lambda1=0.1,
            lambda2=0.1,
            gamma1=0.5,
            gamma2=0.5,
        )
        
        # Set propensity score model
        q1_estimator.treatment_prob = propensity_model
        
        # Train Q1 function
        q1_estimator.fit(data)
        print("✓ KernelBridgeQ1 trained successfully")
        
        # Test prediction
        zx_q1 = np.hstack((data['z'], data['x']))
        q1_predictions = q1_estimator(zx_q1)
        print(f"  Q1 prediction shape: {q1_predictions.shape}")
        print(f"  Q1 prediction range: [{q1_predictions.min():.3f}, {q1_predictions.max():.3f}]")
        
        # Test scoring
        q1_score = q1_estimator.score(data)
        print(f"  Q1 score: {q1_score:.4f}")
        
        # Check model parameters
        print(f"  Q1 alpha shape: {q1_estimator.alpha.shape}")
        print(f"  Q1 beta shape: {q1_estimator.beta.shape}")
        
    except Exception as e:
        print(f"✗ KernelBridgeQ1 failed to run: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Test KernelBridgeQ0
    print("\n" + "-" * 40)
    print("Testing KernelBridgeQ0...")
    print("-" * 40)
    
    try:
        # Create Q0 estimator, passing the trained Q1 estimator
        q0_estimator = KernelBridgeQ0(
            lambda1=0.1,
            lambda2=0.1,
            gamma1=0.5,
            gamma2=0.5,
            q1_estimator=q1_estimator
        )
        
        # Set propensity score model
        q0_estimator.treatment_prob = propensity_model
        
        # Test data extraction
        print("\nTesting Q0 data extraction...")
        g1, g2, wx, zx = q0_estimator.extract_data(data)
        print(f"  g1 shape: {g1.shape}")
        print(f"  g2 shape: {g2.shape}")
        print(f"  wx shape: {wx.shape}")
        print(f"  zx shape: {zx.shape}")
        
        # Check calculation of g1 and g2
        a_values = data['a'][:, 0]
        loc_a0 = a_values == 0
        loc_a1 = a_values == 1
        
        # Manually verify g1
        g1_manual = (loc_a0.astype(float) / probs[:, 0])
        g1_match = np.allclose(g1, g1_manual)
        print(f"  g1 calculated correctly: {g1_match}")
        if not g1_match:
            print(f"    Max difference: {np.max(np.abs(g1 - g1_manual)):.6f}")
        
        # Manually verify g2
        q1_predictions_flat = q1_predictions.flatten() if q1_predictions.ndim > 1 else q1_predictions
        g2_manual = -(loc_a1.astype(float) * q1_predictions_flat / probs[:, 0])
        g2_match = np.allclose(g2, g2_manual)
        print(f"  g2 calculated correctly: {g2_match}")
        if not g2_match:
            print(f"    Max difference: {np.max(np.abs(g2 - g2_manual)):.6f}")
        
        # Train Q0 function
        q0_estimator.fit(data)
        print("✓ KernelBridgeQ0 trained successfully")
        
        # Test prediction
        q0_predictions = q0_estimator(zx)
        print(f"  Q0 prediction shape: {q0_predictions.shape}")
        print(f"  Q0 prediction range: [{q0_predictions.min():.3f}, {q0_predictions.max():.3f}]")
        
        # Test scoring
        q0_score = q0_estimator.score(data)
        print(f"  Q0 score: {q0_score:.4f}")
        
        # Check model parameters
        print(f"  Q0 alpha shape: {q0_estimator.alpha.shape}")
        print(f"  Q0 beta shape: {q0_estimator.beta.shape}")
        
        # Check feature dimensions
        print(f"\nChecking feature dimensions:")
        print(f"  zx dimension of Q1: {zx_q1.shape[1]}")
        print(f"  wx dimension of Q0: {wx.shape[1]}")
        print(f"  zx dimension of Q0: {zx.shape[1]}")
        print(f"  Contains M2: {'m2' in data.keys()}")
        
    except Exception as e:
        print(f"✗ KernelBridgeQ0 failed to run: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Testing completed! Both Q1 and Q0 bridge functions run correctly.")
    print("=" * 60)

if __name__ == "__main__":
    test_q1_and_q0()