import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
import linear_dgp
from bridgeh import KernelBridgeH0, KernelBridgeH1
from bridgeq import KernelBridgeQ0, KernelBridgeQ1, treatment_prob
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

class ProximalEstimatorBase:
    """Base class for proximal causal inference estimators."""

    def __init__(
        self,
        kernel_metric="rbf",
        num_runs=200,
        num_jobs=1,
        verbose=True,
        **kwargs,
    ):
        assert kernel_metric in [
            "rbf",
            "laplacian",
        ], "Invalid kernel metric. Choose from ['rbf', 'laplacian']"
        
        self.kernel_metric = kernel_metric
        self.num_runs = num_runs
        self.num_jobs = num_jobs
        self.verbose = verbose

        # Set any provided parameters
        self.params = {}
        for key in ["treatment", "h", "q", "eta"]:
            if key in kwargs:
                self.params[key] = kwargs[key]

        # Pass in dgp directly for debugging
        self.dgp = kwargs.get("dgp", None)

    def fit(self, fit_data, val_data):
        """Fit the estimator"""
        raise NotImplementedError

    def evaluate(self, eval_data):
        """Evaluate the estimator pointwise on the evaluation data"""
        raise NotImplementedError

    def optimize_h0(self, fit_data, val_data):
        """Optimize hyperparameters for h0"""
        print("=" * 60)
        print("Starting hyperparameter optimization for h0...")
        print("=" * 60)
        
        # Check for predefined parameters
        if all(param in self.params.get("h0", {}) for param in ['lambda1', 'lambda2', 'gamma']):
            lambda1 = self.params["h0"]['lambda1']
            lambda2 = self.params["h0"]['lambda2']
            gamma = self.params["h0"]['gamma']
            print(f"Using predefined parameters: lambda1={lambda1}, lambda2={lambda2}, gamma={gamma}")
            best_score = None
            best_params = self.params["h0"]
        else:
            # Use Optuna for hyperparameter optimization
            def _optimize_h0(trial):
                lambda1 = trial.suggest_loguniform(
                    'lambda1', 
                    LAMBDA_MIN_FACTOR * len(fit_data) ** 0.2,
                    LAMBDA_MAX_FACTOR * len(fit_data) ** 0.2
                )
                lambda2 = trial.suggest_loguniform(
                    'lambda2',
                    LAMBDA_MIN_FACTOR * len(fit_data) ** 0.2,
                    LAMBDA_MAX_FACTOR * len(fit_data) ** 0.2
                )
                gamma = trial.suggest_loguniform('gamma', GAMMA_MIN, GAMMA_MAX)
                
                # Create and train h0 estimator
                est = KernelBridgeH0(lambda1, lambda2, gamma, gamma)
                est.fit(fit_data)
                
                # Return validation score (lower is better)
                return est.score(val_data)
            
            study = optuna.create_study(direction="minimize")
            study.optimize(_optimize_h0, n_trials=self.num_runs)
            
            best_params = study.best_params
            best_score = study.best_value
            lambda1 = best_params['lambda1']
            lambda2 = best_params['lambda2']
            gamma = best_params['gamma']
            
            if self.verbose:
                print(f"h0 best parameters: {best_params}, score: {best_score:.4f}")
        
        # Train final h0 estimator with best parameters
        h0_best = KernelBridgeH0(lambda1, lambda2, gamma, gamma)
        h0_best.fit(fit_data)
        
        self.params["h0"] = best_params
        self.h0_estimator = h0_best
        best_score = h0_best.score(val_data)
        
        return h0_best, best_params, best_score
    
    def optimize_h1(self, fit_data, val_data, h0_estimator):
        """Optimize hyperparameters for h1 (requires trained h0_estimator)"""
        print("\n" + "=" * 60)
        print("Starting hyperparameter optimization for h1...")
        print("=" * 60)
        
        if all(param in self.params.get("h1", {}) for param in ['lambda1', 'lambda2', 'gamma']):
            lambda1 = self.params["h1"]['lambda1']
            lambda2 = self.params["h1"]['lambda2']
            gamma = self.params["h1"]['gamma']
            print(f"Using predefined parameters: lambda1={lambda1}, lambda2={lambda2}, gamma={gamma}")
            best_score = None
            best_params = self.params["h1"]
        else:
            def _optimize_h1(trial):
                lambda1 = trial.suggest_loguniform(
                    'lambda1',
                    LAMBDA_MIN_FACTOR * len(fit_data) ** 0.2,
                    LAMBDA_MAX_FACTOR * len(fit_data) ** 0.2
                )
                lambda2 = trial.suggest_loguniform(
                    'lambda2',
                    LAMBDA_MIN_FACTOR * len(fit_data) ** 0.2,
                    LAMBDA_MAX_FACTOR * len(fit_data) ** 0.2
                )
                gamma = trial.suggest_loguniform('gamma', GAMMA_MIN, GAMMA_MAX)
                
                est = KernelBridgeH1(
                    lambda1, lambda2, gamma, gamma,
                    h0_estimator=h0_estimator
                )
                est.fit(fit_data)
                
                return est.score(val_data)
            
            study = optuna.create_study(direction="minimize")
            study.optimize(_optimize_h1, n_trials=self.num_runs)
            
            best_params = study.best_params
            best_score = study.best_value
            lambda1 = best_params['lambda1']
            lambda2 = best_params['lambda2']
            gamma = best_params['gamma']
            
            if self.verbose:
                print(f"h1 best parameters: {best_params}, score: {best_score:.4f}")
        
        h1_best = KernelBridgeH1(
            lambda1, lambda2, gamma, gamma,
            h0_estimator=h0_estimator
        )
        h1_best.fit(fit_data)
        
        self.params["h1"] = best_params
        self.h1_estimator = h1_best
        best_score = h1_best.score(val_data)
        
        return h1_best, best_params, best_score
    
    def optimize_both(self, fit_data, val_data):
        """Optimize h0 and h1 sequentially"""
        h0_best, h0_params, h0_score = self.optimize_h0(fit_data, val_data)
        h1_best, h1_params, h1_score = self.optimize_h1(fit_data, val_data, h0_best)
        
        results = {
            'h0': {'estimator': h0_best, 'params': h0_params, 'score': h0_score},
            'h1': {'estimator': h1_best, 'params': h1_params, 'score': h1_score}
        }
        return results

    def optimize_p(self, fit_data, val_data):
        """Train treatment probability model - merging train and validation sets"""
        combined_x = np.vstack([fit_data['x'], val_data['x']])
        combined_a = np.hstack([fit_data['a'][:, 0], val_data['a'][:, 0]])
        
        self.treatment_prob_model = treatment_prob()
        self.treatment_prob_model.fit(X=combined_x, A=combined_a)
        
        return self.treatment_prob_model, {}

    def optimize_q1(self, fit_data, val_data):
        """Optimize hyperparameters for q1"""
        print("\n" + "=" * 60)
        print("Optimizing hyperparameters for q1...")
        print("=" * 60)
        
        if not hasattr(self, 'treatment_prob_model') or self.treatment_prob_model is None:
          raise ValueError("Propensity model not found. Please call optimize_p() or optimize_allq() first.")
        
        if all(param in self.params.get("q1", {}) for param in ['lambda1', 'lambda2', 'gamma']):
            lambda1 = self.params["q1"]['lambda1']
            lambda2 = self.params["q1"]['lambda2']
            gamma = self.params["q1"]['gamma']
            print(f"Using predefined parameters: lambda1={lambda1}, lambda2={lambda2}, gamma={gamma}")
            best_score = None
            best_params = self.params["q1"]
        else:
            def _optimize_q1(trial):
                lambda1 = trial.suggest_loguniform(
                    'lambda1', 
                    LAMBDA_MIN_FACTOR * len(fit_data) ** 0.2,
                    LAMBDA_MAX_FACTOR * len(fit_data) ** 0.2
                )
                lambda2 = trial.suggest_loguniform(
                    'lambda2',
                    LAMBDA_MIN_FACTOR * len(fit_data) ** 0.2,
                    LAMBDA_MAX_FACTOR * len(fit_data) ** 0.2
                )
                gamma = trial.suggest_loguniform('gamma', GAMMA_MIN, GAMMA_MAX)
                
                est = KernelBridgeQ1(lambda1, lambda2, gamma, gamma)
                est.treatment_prob = self.treatment_prob_model
                est.fit(fit_data)
                
                return est.score(val_data)
            
            study = optuna.create_study(direction="minimize")
            study.optimize(_optimize_q1, n_trials=self.num_runs)
            
            best_params = study.best_params
            best_score = study.best_value
            lambda1 = best_params['lambda1']
            lambda2 = best_params['lambda2']
            gamma = best_params['gamma']
            
            if self.verbose:
                print(f"q1 best parameters: {best_params}, score: {best_score:.4f}")
        
        q1_best = KernelBridgeQ1(lambda1, lambda2, gamma, gamma)
        q1_best.treatment_prob = self.treatment_prob_model
        q1_best.fit(fit_data)
        
        self.params["q1"] = best_params
        self.q1_estimator = q1_best
        best_score = q1_best.score(val_data)
        
        return q1_best, best_params, best_score

    def optimize_q0(self, fit_data, val_data, q1_estimator):
        """Optimize hyperparameters for q0 (requires trained q1_estimator)"""
        print("\n" + "=" * 60)
        print("Optimizing hyperparameters for q0...")
        print("=" * 60)
        
        if not hasattr(self, 'treatment_prob_model') or self.treatment_prob_model is None:
          raise ValueError("Propensity model not found. Please call optimize_p() or optimize_allq() first.")
        
        if all(param in self.params.get("q0", {}) for param in ['lambda1', 'lambda2', 'gamma']):
            lambda1 = self.params["q0"]['lambda1']
            lambda2 = self.params["q0"]['lambda2']
            gamma = self.params["q0"]['gamma']
            print(f"Using predefined parameters: lambda1={lambda1}, lambda2={lambda2}, gamma={gamma}")
            best_score = None
            best_params = self.params["q0"]
        else:
            def _optimize_q0(trial):
                lambda1 = trial.suggest_loguniform(
                    'lambda1',
                    LAMBDA_MIN_FACTOR * len(fit_data) ** 0.2,
                    LAMBDA_MAX_FACTOR * len(fit_data) ** 0.2
                )
                lambda2 = trial.suggest_loguniform(
                    'lambda2',
                    LAMBDA_MIN_FACTOR * len(fit_data) ** 0.2,
                    LAMBDA_MAX_FACTOR * len(fit_data) ** 0.2
                )
                gamma = trial.suggest_loguniform('gamma', GAMMA_MIN, GAMMA_MAX)
                
                est = KernelBridgeQ0(
                    lambda1, lambda2, gamma, gamma,
                    q1_estimator=q1_estimator
                )
                est.treatment_prob = self.treatment_prob_model
                est.fit(fit_data)
                
                return est.score(val_data)
            
            study = optuna.create_study(direction="minimize")
            study.optimize(_optimize_q0, n_trials=self.num_runs)
            
            best_params = study.best_params
            best_score = study.best_value
            lambda1 = best_params['lambda1']
            lambda2 = best_params['lambda2']
            gamma = best_params['gamma']
            
            if self.verbose:
                print(f"q0 best parameters: {best_params}, score: {best_score:.4f}")
        
        q0_best = KernelBridgeQ0(
            lambda1, lambda2, gamma, gamma,
            q1_estimator=q1_estimator
        )
        q0_best.treatment_prob = self.treatment_prob_model
        q0_best.fit(fit_data)
        
        self.params["q0"] = best_params
        self.q0_estimator = q0_best
        best_score = q0_best.score(val_data)
        
        return q0_best, best_params, best_score

    def optimize_allq(self, fit_data, val_data):
        """Train/Optimize all Q-related models sequentially"""
        print("=" * 60)
        print("Training all Q-related models...")
        print("=" * 60)
        
        treatment_model, _ = self.optimize_p(fit_data, val_data)
        q1_best, q1_params, q1_score = self.optimize_q1(fit_data, val_data)
        q0_best, q0_params, q0_score = self.optimize_q0(fit_data, val_data, q1_best)
        
        results = {
            'treatment': {'estimator': treatment_model},
            'q1': {'estimator': q1_best, 'params': q1_params, 'score': q1_score},
            'q0': {'estimator': q0_best, 'params': q0_params, 'score': q0_score}
        }
        
        print("\n" + "=" * 60)
        print("All Q-related models completed!")
        print("=" * 60)
        print(f"q1 score: {q1_score:.4f}")
        print(f"q0 score: {q0_score:.4f}")
        
        return results

    def fit_eta(self, h1_fn, fit_data, val_data):
        """Estimate h1_fn(w,x) at A=0 using Kernel Ridge Regression with hyperparameter optimization"""
        
        def extract_data(data):
            loc = data['a'][:, 0] == 0
            x0 = data['x'][loc]
            targets = h1_fn(np.hstack((data['w'][loc], data['x'][loc])))
            return x0, targets
        
        def build_reg(alpha=1, gamma=None):
            return Pipeline([
                ("scaler", StandardScaler()),
                (
                    "kernel_ridge",
                    KernelRidge(
                        kernel=self.kernel_metric, 
                        alpha=alpha, 
                        gamma=gamma
                    ),
                ),
            ])
        
        x_fit, y_fit = extract_data(fit_data)
        x_val, y_val = extract_data(val_data)
        
        print("=" * 60)
        print("Starting hyperparameter optimization for Kernel Ridge Regression...")
        print("=" * 60)
        
        if all(param in self.params.get("eta", {}) for param in ['alpha', 'gamma']):
            alpha = self.params["eta"]['alpha']
            gamma = self.params["eta"]['gamma']
            print(f"Using predefined parameters: alpha={alpha}, gamma={gamma}")
            best_params = self.params["eta"]
            best_score = None
        else:
            def _optimize_eta(trial):
                alpha = trial.suggest_loguniform('alpha', ALPHA_MIN, ALPHA_MAX)
                gamma = trial.suggest_loguniform('gamma', REG_GAMMA_MIN, REG_GAMMA_MAX)
                
                model = build_reg(alpha=alpha, gamma=gamma)
                model.fit(x_fit, y_fit)
                
                y_pred = model.predict(x_val)
                mse = np.mean((y_val - y_pred) ** 2)
                return mse
            
            study = optuna.create_study(
                direction="minimize",
                study_name="kernel_ridge_optimization"
            )
            
            n_trials = self.num_runs
            study.optimize(_optimize_eta, n_trials=n_trials)
            
            best_params = study.best_params
            best_score = study.best_value
            
            alpha = best_params['alpha']
            gamma = best_params['gamma']
            
            if self.verbose:
                print(f"Kernel Ridge best parameters: {best_params}")
                print(f"Best MSE: {best_score:.6f}")
        
        cond_mean = build_reg(alpha=alpha, gamma=gamma)
        cond_mean.fit(x_fit, y_fit)
        
        self.params["eta"] = best_params
        self.eta_model = cond_mean
        
        return cond_mean
