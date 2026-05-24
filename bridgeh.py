"""Estimator for the bridge function h(W, X)"""

import numpy as np
from minimax import kkt_solve, score_nuisance_function
import bridge_base
import linear_dgp
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
        """Extract the data for fitting the bridge function"""
        loc = data['a'][:, 0] == 0
        g2 = data['y'][loc, 0]
        g1 = -np.ones(len(g2))
        wx = np.hstack((data['w'], data['m2'], data['x']))
        zx = np.hstack((data['z'], data['m2'], data['x']))
        return g1, g2, wx, zx, loc

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
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
        """Score the bridge function"""
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
        self.h0_estimator = h0_estimator  # Save h0 estimator
    
    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
        loc = data['a'][:, 0] == 1  # Note: Here we filter for A=1
        
        # Key modification: Use the prediction from h0_estimator as g2
        if self.h0_estimator is not None:
            # Construct the same features used for h0: W, M2, X
            wx_h0 = np.hstack((data['w'], data['m2'], data['x']))
            # Predict all samples using the h0 estimator
            h0_predictions = self.h0_estimator(wx_h0).flatten()
            # Retain only samples where A=1
            g2 = h0_predictions[loc]
        else:
            # If h0_estimator is not provided, raise an error
            raise ValueError("h0_estimator is required for KernelBridgeH1")
        
        g1 = -np.ones(len(g2))
        
        # Feature combination for h1: W, X (note the difference from h0)
        wx = np.hstack((data['w'], data['x']))
        zx = np.hstack((data['z'], data['x']))
        
        return g1, g2, wx, zx, loc

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
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
        """Score the bridge function"""
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
        
def main():
    """Test whether KernelBridgeH0 and KernelBridgeH1 run correctly"""
    print("=" * 60)
    print("Testing if bridge functions run correctly")
    print("=" * 60)
    
    # 1. Create data generator
    datagen = linear_dgp.LinearTwoMediatorsProxDGP(
        xdim=5,     
        zdim=2,     
        wdim=2,     
        m1dim=2,    
        m2dim=2,    
        seed=42
    )
    
    # 2. Generate a small amount of data (e.g., 100 samples)
    print("\nGenerating data...")
    data = datagen.sample_dataset(n=100, seed=42)
    
    # Print data shapes
    print("Data shapes:")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
    
    # 3. Test KernelBridgeH0
    print("\n" + "-" * 40)
    print("Testing KernelBridgeH0...")
    print("-" * 40)
    
    try:
        # Create h0 estimator
        h0_estimator = KernelBridgeH0(
            lambda1=0.1,
            lambda2=0.1,
            gamma1=0.5,
            gamma2=0.5,
        )
        
        # Train h0
        h0_estimator.fit(data)
        print("✓ KernelBridgeH0 trained successfully")
        
        # Test prediction
        wx_h0 = np.hstack((data['w'], data['m2'], data['x']))
        h0_predictions = h0_estimator(wx_h0)
        print(f"✓ KernelBridgeH0 predicted successfully, prediction shape: {h0_predictions.shape}")
        
        # Test scoring (using the same data)
        h0_score = h0_estimator.score(data)
        print(f"✓ KernelBridgeH0 scored successfully, score: {h0_score:.4f}")
        
    except Exception as e:
        print(f"✗ KernelBridgeH0 failed to run: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Test KernelBridgeH1
    print("\n" + "-" * 40)
    print("Testing KernelBridgeH1...")
    print("-" * 40)
    
    try:
        # Create h1 estimator, passing the trained h0_estimator
        h1_estimator = KernelBridgeH1(
            lambda1=0.1,
            lambda2=0.1,
            gamma1=0.5,
            gamma2=0.5,
            h0_estimator=h0_estimator
        )
        
        # Train h1
        h1_estimator.fit(data)
        print("✓ KernelBridgeH1 trained successfully")
        
        # Test prediction
        wx_h1 = np.hstack((data['w'], data['x']))
        h1_predictions = h1_estimator(wx_h1)
        print(f"✓ KernelBridgeH1 predicted successfully, prediction shape: {h1_predictions.shape}")
        
        # Test scoring
        h1_score = h1_estimator.score(data)
        print(f"✓ KernelBridgeH1 scored successfully, score: {h1_score:.4f}")
        
    except Exception as e:
        print(f"✗ KernelBridgeH1 failed to run: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Check data flow
    print("\n" + "-" * 40)
    print("Checking data flow...")
    print("-" * 40)
    
    # Check g2 values for h0 and h1
    loc_a0 = data['a'][:, 0] == 0
    loc_a1 = data['a'][:, 0] == 1
    
    print(f"Data statistics:")
    print(f"  Total samples: {len(data['a'])}")
    print(f"  Samples with A=0: {np.sum(loc_a0)}")
    print(f"  Samples with A=1: {np.sum(loc_a1)}")
    
    # Check the range of h0 predictions
    if 'h0_estimator' in locals():
        h0_all_predictions = h0_estimator(np.hstack((data['w'], data['m2'], data['x'])))
        print(f"  h0 prediction range: [{h0_all_predictions.min():.3f}, {h0_all_predictions.max():.3f}]")
    
    # Check if g2 for h1 is indeed the prediction of h0 on A=1
    if 'h1_estimator' in locals():
        # Get g2 extracted by h1
        g1, g2_h1, wx_h1, zx_h1, loc_h1 = h1_estimator.extract_data(data)
        
        # Calculate h0 prediction on A=1 samples
        wx_h0_all = np.hstack((data['w'], data['m2'], data['x']))
        h0_all_pred = h0_estimator(wx_h0_all)
        if h0_all_pred.ndim > 1:
            h0_all_pred = h0_all_pred.flatten()
        h0_a1_pred = h0_all_pred[loc_a1]
        
        print(f"  Shape of g2 for h1: {g2_h1.shape}")
        print(f"  Shape of h0 prediction on A=1 samples: {h0_a1_pred.shape}")
        
        # Check if they match
        if np.allclose(g2_h1, h0_a1_pred, rtol=1e-5, atol=1e-5):
            print("  ✓ g2 for h1 correctly equals h0 prediction on A=1 samples")
        else:
            print("  ✗ g2 for h1 does not match h0 prediction")
            print(f"    Max difference: {np.max(np.abs(g2_h1 - h0_a1_pred)):.6f}")
    
    print("\n" + "=" * 60)
    print("Testing completed! Both bridge functions run correctly.")
    print("=" * 60)

if __name__ == "__main__":
    main()