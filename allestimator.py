import numpy as np
from proximal_estimator_base import ProximalEstimatorBase


class AllEstimator(ProximalEstimatorBase):
    """Multiply robust identification estimator"""
    preset_params = {
        'h0': {
            'lambda1': 0.01,
            'lambda2': 0.01,
            'gamma': 0.01
        },
        'h1': {
            'lambda1': 0.01,
            'lambda2': 0.01,
            'gamma': 0.01
        },
        'q0': {
            'lambda1': 0.01,
            'lambda2': 0.01,
            'gamma': 0.01
        },
        'q1': {
            'lambda1': 0.01,
            'lambda2': 0.01,
            'gamma': 0.01
        },
        'eta': {
            'alpha': 0.01,
            'gamma': 0.01
        },
    }
    
    def fit(self, fit_data, val_data):
        """Train the multiply robust identification estimator"""
        print("=" * 60)
        print("Starting training for the multiply robust identification estimator...")
        print("=" * 60)
        self.params.update(self.preset_params)
        
        # Step 1: Optimize h0 and h1
        print("\nOptimizing h0 and h1...")
        both_results = self.optimize_both(fit_data, val_data)
        
        # Save h0 and h1
        self.h0_fn = both_results['h0']['estimator']
        self.h1_fn = both_results['h1']['estimator']
        self.params["h0"] = both_results['h0']['params']
        self.params["h1"] = both_results['h1']['params']
        self.h0_score = both_results['h0']['score']
        self.h1_score = both_results['h1']['score']
        
        # Step 2: Optimize all Q-related models
        print("\nOptimizing all Q-related models...")
        q_results = self.optimize_allq(fit_data, val_data)
        
        # Save propensity score model, q0, and q1
        self.treatment_prob_model = q_results['treatment']['estimator']
        self.q0_fn = q_results['q0']['estimator']
        self.q1_fn = q_results['q1']['estimator']
        self.params["q0"] = q_results['q0']['params']
        self.params["q1"] = q_results['q1']['params']
        self.q0_score = q_results['q0']['score']
        self.q1_score = q_results['q1']['score']
        
        # Step 3: Train the eta (η) function
        print("\nTraining the eta (η) function...")
        self.eta = self.fit_eta(self.h1_fn, fit_data, val_data)
        
        print("\n" + "=" * 60)
        print("Multiply robust identification estimator training completed!")
        print("=" * 60)
        print(f"h0 score: {self.h0_score:.4f}")
        print(f"h1 score: {self.h1_score:.4f}")
        print(f"q0 score: {self.q0_score:.4f}")
        print(f"q1 score: {self.q1_score:.4f}")
        
        return self

    def evaluatepor(self, eval_data):
        """Evaluate the POR estimator pointwise on the evaluation data"""
        return self.eta.predict(eval_data['x'])

    def evaluatepipw(self, eval_data):
        """Evaluate the PIPW estimator pointwise on the evaluation data"""
        # Get indicator function for A=0
        loc0 = eval_data['a'][:, 0] == 0
        a0_eval = loc0.astype(float).flatten()
        
        # Get outcome variable y
        y_eval = eval_data['y'][:, 0].flatten()
        
        # Get predicted values for q0 function (using Z, M2, X as inputs)
        zx_m2 = np.hstack((eval_data['z'], eval_data['m2'], eval_data['x']))
        q0_eval = self.q0_fn(zx_m2).flatten()
        
        # Get propensity score P(A=1|X)
        prop_score = self.treatment_prob_model.predict_proba(eval_data['x'])[:, 1]
        
        # Calculate estimate: I(A=0) * q0(Z,M2,X) * Y / P(A=1|X)
        estimate = a0_eval * y_eval * q0_eval / prop_score
        
        return estimate

    def evaluatephe(self, eval_data):
        """Evaluate the Proximal Hybrid Estimator (PHE): I(A=1) * h0(W,M2,X) * q1(Z,X) / P(A=1|X)"""
        # Get indicator function for A=1
        loc1 = eval_data['a'][:, 0] == 1
        a1_eval = loc1.astype(float).flatten()
        
        # Get predicted values for h0 function (using W, M2, X as inputs)
        w_m2_x = np.hstack((eval_data['w'], eval_data['m2'], eval_data['x']))
        h0_eval = self.h0_fn(w_m2_x).flatten()
        
        # Get predicted values for q1 function (using Z, X as inputs)
        z_x = np.hstack((eval_data['z'], eval_data['x']))
        q1_eval = self.q1_fn(z_x).flatten()
        
        # Get propensity score P(A=1|X)
        prop_score = self.treatment_prob_model.predict_proba(eval_data['x'])[:, 1]
        
        # Calculate estimate: I(A=1) * h0(W,M2,X) * q1(Z,X) / P(A=1|X)
        estimate = a1_eval * h0_eval * q1_eval / prop_score
        
        return estimate

    def evaluatepmr(self, eval_data):
        """Evaluate the Proximal Multiply Robust (PMR) estimator"""
        # Get propensity scores
        probs = self.treatment_prob_model.predict_proba(eval_data['x'])
        prop_score_1 = probs[:, 1]  # P(A=1|X)
        prop_score_0 = probs[:, 0]  # P(A=0|X)
        
        # Get indicator functions for A
        a_values = eval_data['a'][:, 0]
        a1_indicator = (a_values == 1).astype(float)  # I(A=1)
        a0_indicator = (a_values == 0).astype(float)  # I(A=0)
        
        # Get outcome variable Y
        y = eval_data['y'][:, 0]
        
        # Get h0 predictions (W, M2, X)
        w_m2_x = np.hstack((eval_data['w'], eval_data['m2'], eval_data['x']))
        h0_pred = self.h0_fn(w_m2_x).flatten()
        
        # Get h1 predictions (W, X)
        w_x = np.hstack((eval_data['w'], eval_data['x']))
        h1_pred = self.h1_fn(w_x).flatten()
        
        # Get q0 predictions (Z, M2, X)
        z_m2_x = np.hstack((eval_data['z'], eval_data['m2'], eval_data['x']))
        q0_pred = self.q0_fn(z_m2_x).flatten()
        
        # Get q1 predictions (Z, X)
        z_x = np.hstack((eval_data['z'], eval_data['x']))
        q1_pred = self.q1_fn(z_x).flatten()
        
        # Get eta (η) predictions (X)
        eta_pred = self.eta.predict(eval_data['x']).flatten()
        
        # Calculate individual terms
        # Term 1: (I(A=1)/P(A=1|X)) * q1 * (h0 - h1)
        term1 = (a1_indicator / prop_score_1) * q1_pred * (h0_pred - h1_pred)
        
        # Term 2: (I(A=0)/P(A=1|X)) * q0 * (y - h0)
        term2 = (a0_indicator / prop_score_1) * q0_pred * (y - h0_pred)
        
        # Term 3: (I(A=0)/P(A=0|X)) * (h1 - η)
        term3 = (a0_indicator / prop_score_0) * (h1_pred - eta_pred)
        
        # Term 4: eta (η)
        term4 = eta_pred
        
        # Calculate final estimate
        estimate = term1 + term2 + term3 + term4
        
        return estimate