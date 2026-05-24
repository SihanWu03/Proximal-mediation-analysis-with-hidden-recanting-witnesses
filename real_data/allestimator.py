import numpy as np
from proximal_estimator_base import ProximalEstimatorBase


class AllEstimator(ProximalEstimatorBase):
    """Multiply Robust Identification Estimator."""
    
    preset_params = {
        # Outcome bridge functions
        'h0': {
            'lambda1': 0.1, 
            'lambda2': 0.1, 
            'gamma': 0.01
        },
        'h1': {
            'lambda1': 0.1, 
            'lambda2': 0.1, 
            'gamma': 0.01
        },
        
        # Weight bridge functions
        'q0': {
            'lambda1': 0.5,
            'lambda2': 0.5,
            'gamma': 0.01
        },
        'q1': {
            'lambda1': 0.5, 
            'lambda2': 0.5, 
            'gamma': 0.01
        },
        
        # Baseline regression term
        'eta': {
            'alpha': 0.1, 
            'gamma': 0.01
        },
    }
        
    def fit(self, fit_data, val_data):
        """Train the multiply robust identification estimator."""
        print("=" * 60)
        print("Training the multiply robust estimator...")
        print("=" * 60)
        
        self.params.update(self.preset_params)
        
        # Step 1: Optimize h0 and h1
        print("\nOptimizing h0 and h1...")
        both_results = self.optimize_both(fit_data, val_data)
        
        self.h0_fn = both_results['h0']['estimator']
        self.h1_fn = both_results['h1']['estimator']
        self.params["h0"] = both_results['h0']['params']
        self.params["h1"] = both_results['h1']['params']
        self.h0_score = both_results['h0']['score']
        self.h1_score = both_results['h1']['score']
        
        # Step 2: Optimize all Q-related models
        print("\nOptimizing all Q-related models...")
        q_results = self.optimize_allq(fit_data, val_data)
        
        self.treatment_prob_model = q_results['treatment']['estimator']
        self.q0_fn = q_results['q0']['estimator']
        self.q1_fn = q_results['q1']['estimator']
        self.params["q0"] = q_results['q0']['params']
        self.params["q1"] = q_results['q1']['params']
        self.q0_score = q_results['q0']['score']
        self.q1_score = q_results['q1']['score']
        
        # Step 3: Train the eta function
        print("\nTraining the eta function...")
        self.eta = self.fit_eta(self.h1_fn, fit_data, val_data)
        
        print("\n" + "=" * 60)
        print("Multiply robust estimator training completed!")
        print("=" * 60)
        print(f"h0 score: {self.h0_score:.4f}")
        print(f"h1 score: {self.h1_score:.4f}")
        print(f"q0 score: {self.q0_score:.4f}")
        print(f"q1 score: {self.q1_score:.4f}")
        
        return self

    def evaluatepor(self, eval_data):
        """Evaluate the POR estimator pointwise on the evaluation data."""
        return self.eta.predict(eval_data['x'])

    def evaluatepipw(self, eval_data):
        """Evaluate the PIPW estimator pointwise on the evaluation data."""
        # Indicator for A=0
        loc0 = eval_data['a'][:, 0] == 0
        a0_eval = loc0.astype(float).flatten()
        
        # Outcome variable Y
        y_eval = eval_data['y'][:, 0].flatten()
        
        # Predictions for q0 (using Z, M2, X)
        zx_m2 = np.hstack((eval_data['z'], eval_data['m2'], eval_data['x']))
        q0_eval = self.q0_fn(zx_m2).flatten()
        
        # Propensity score P(A=1|X)
        prop_score = self.treatment_prob_model.predict_proba(eval_data['x'])[:, 1]
        
        # Calculate estimate: I(A=0) * q0(Z,M2,X) * Y / P(A=1|X)
        estimate = a0_eval * y_eval * q0_eval / prop_score
        
        return estimate

    def evaluatephe(self, eval_data):
        """Evaluate the PHE estimator pointwise on the evaluation data."""
        # Indicator for A=1
        loc1 = eval_data['a'][:, 0] == 1
        a1_eval = loc1.astype(float).flatten()
        
        # Predictions for h0 (using W, M2, X)
        w_m2_x = np.hstack((eval_data['w'], eval_data['m2'], eval_data['x']))
        h0_eval = self.h0_fn(w_m2_x).flatten()
        
        # Predictions for q1 (using Z, X)
        z_x = np.hstack((eval_data['z'], eval_data['x']))
        q1_eval = self.q1_fn(z_x).flatten()
        
        # Propensity score P(A=1|X)
        prop_score = self.treatment_prob_model.predict_proba(eval_data['x'])[:, 1]
        
        # Calculate estimate: I(A=1) * h0(W,M2,X) * q1(Z,X) / P(A=1|X)
        estimate = a1_eval * h0_eval * q1_eval / prop_score
        
        return estimate

    def evaluatepmr(self, eval_data):
        """Evaluate the PMR estimator pointwise on the evaluation data."""
        # Propensity scores
        probs = self.treatment_prob_model.predict_proba(eval_data['x'])
        prop_score_1 = probs[:, 1]  # P(A=1|X)
        prop_score_0 = probs[:, 0]  # P(A=0|X)
        
        # Treatment indicators
        a_values = eval_data['a'][:, 0]
        a1_indicator = (a_values == 1).astype(float)  # I(A=1)
        a0_indicator = (a_values == 0).astype(float)  # I(A=0)
        
        # Outcome Y
        y = eval_data['y'][:, 0]
        
        # Predictions for h0 (W, M2, X)
        w_m2_x = np.hstack((eval_data['w'], eval_data['m2'], eval_data['x']))
        h0_pred = self.h0_fn(w_m2_x).flatten()
        
        # Predictions for h1 (W, X)
        w_x = np.hstack((eval_data['w'], eval_data['x']))
        h1_pred = self.h1_fn(w_x).flatten()
        
        # Predictions for q0 (Z, M2, X)
        z_m2_x = np.hstack((eval_data['z'], eval_data['m2'], eval_data['x']))
        q0_pred = self.q0_fn(z_m2_x).flatten()
        
        # Predictions for q1 (Z, X)
        z_x = np.hstack((eval_data['z'], eval_data['x']))
        q1_pred = self.q1_fn(z_x).flatten()
        
        # Predictions for eta (X)
        eta_pred = self.eta.predict(eval_data['x']).flatten()
        
        # Calculate individual terms
        # Term 1: (I(A=1)/P(A=1|X)) * q1 * (h0 - h1)
        term1 = (a1_indicator / prop_score_1) * q1_pred * (h0_pred - h1_pred)
        
        # Term 2: (I(A=0)/P(A=1|X)) * q0 * (y - h0)
        term2 = (a0_indicator / prop_score_1) * q0_pred * (y - h0_pred)
        
        # Term 3: (I(A=0)/P(A=0|X)) * (h1 - eta)
        term3 = (a0_indicator / prop_score_0) * (h1_pred - eta_pred)
        
        # Term 4: eta
        term4 = eta_pred
        
        # Calculate final estimate
        estimate = term1 + term2 + term3 + term4
        
        return estimate