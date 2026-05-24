import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
import linear_dgp as linear_dgp
from datetime import datetime
import os
import sys
import argparse
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
warnings.filterwarnings('ignore')

class CorrectMRCrossFittedEstimator:
    """Multiply robust estimator implemented according to the formulas (PMR only)"""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def fit_and_predict_all_nuisance(self, data, a_prime=0, a=1):
        """Fit all nuisance functions: B, B', B'', propensity score, and density ratios"""
        n = len(data['y'])
        
        b_preds = np.zeros(n)
        b_prime_preds = np.zeros(n)
        b_double_prime_preds = np.zeros(n)
        propensities = np.zeros(n)
        m1_ratios = np.zeros(n)
        m2_ratios = np.zeros(n)
        
        for train_idx, test_idx in self.kf.split(np.arange(n)):
            train_data = {k: v[train_idx] for k, v in data.items()}
            test_data = {k: v[test_idx] for k, v in data.items()}
            
            # 1. Estimate propensity score f(A|X)
            ps_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=self.random_state, n_jobs=1)
            ps_model.fit(train_data['x'], train_data['a'].ravel())
            prob_test = ps_model.predict_proba(test_data['x'])[:, 1]
            propensities[test_idx] = np.clip(prob_test, 0.01, 0.99)
            
            # 2. Estimate f(A|Z,X)
            XZ_train = np.hstack([train_data['z'], train_data['x']])
            XZ_test = np.hstack([test_data['z'], test_data['x']])
            ps_zx_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=self.random_state, n_jobs=1)
            ps_zx_model.fit(XZ_train, train_data['a'].ravel())
            prob_zx_test = np.clip(ps_zx_model.predict_proba(XZ_test)[:, 1], 0.01, 0.99)
            
            # 3. Estimate f(A|M2,Z,X)
            XM2Z_train = np.hstack([train_data['m2'], train_data['z'], train_data['x']])
            XM2Z_test = np.hstack([test_data['m2'], test_data['z'], test_data['x']])
            ps_m2zx_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=self.random_state, n_jobs=1)
            ps_m2zx_model.fit(XM2Z_train, train_data['a'].ravel())
            prob_m2zx_test = np.clip(ps_m2zx_model.predict_proba(XM2Z_test)[:, 1], 0.01, 0.99)
            
            # 4. Calculate density ratios
            f_a_given_x_test = propensities[test_idx]
            f_a_prime_given_x_test = 1 - f_a_given_x_test
            
            m1_ratios[test_idx] = np.clip(((1 - prob_zx_test) / prob_zx_test) * (f_a_given_x_test / f_a_prime_given_x_test), 0.1, 10)
            m2_ratios[test_idx] = np.clip((prob_m2zx_test / (1 - prob_m2zx_test)) * ((1 - prob_zx_test) / prob_zx_test), 0.1, 10)
            
            # 5. Fit model B: E(Y | m2, z, a_prime, x)
            mask_b = (train_data['a'] == a_prime).flatten()
            if np.sum(mask_b) >= 5:
                X_b_train = np.hstack([train_data['m2'][mask_b], train_data['z'][mask_b], train_data['x'][mask_b]])
                y_b_train = train_data['y'][mask_b].ravel()
                b_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=self.random_state, n_jobs=1)
                b_model.fit(X_b_train, y_b_train)
                
                X_b_test = np.hstack([test_data['m2'], test_data['z'], test_data['x']])
                b_preds[test_idx] = b_model.predict(X_b_test)
                
                # 6. Fit model B': E[B(M2,z,a',X) | z, a, x]
                mask_b_prime = (train_data['a'] == a).flatten()
                if np.sum(mask_b_prime) >= 5:
                    X_b_train_for_prime = np.hstack([train_data['m2'][mask_b_prime], train_data['z'][mask_b_prime], train_data['x'][mask_b_prime]])
                    pseudo_b_prime_train = b_model.predict(X_b_train_for_prime)
                    X_b_prime_train = np.hstack([train_data['z'][mask_b_prime], train_data['x'][mask_b_prime]])
                    b_prime_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=self.random_state, n_jobs=1)
                    b_prime_model.fit(X_b_prime_train, pseudo_b_prime_train)
                    
                    X_b_prime_test = np.hstack([test_data['z'], test_data['x']])
                    b_prime_preds[test_idx] = b_prime_model.predict(X_b_prime_test)
                    
                    # 7. Fit model B'': E[B'(z,a',a,X) | a', x]
                    mask_b_double = (train_data['a'] == a_prime).flatten()
                    if np.sum(mask_b_double) >= 5:
                        X_b_prime_train_for_double = np.hstack([train_data['z'][mask_b_double], train_data['x'][mask_b_double]])
                        pseudo_b_double_train = b_prime_model.predict(X_b_prime_train_for_double)
                        b_double_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=self.random_state, n_jobs=1)
                        b_double_model.fit(train_data['x'][mask_b_double], pseudo_b_double_train)
                        
                        b_double_prime_preds[test_idx] = b_double_model.predict(test_data['x'])
        
        return {
            'b': b_preds,
            'b_prime': b_prime_preds,
            'b_double_prime': b_double_prime_preds,
            'propensities': propensities,
            'm1_ratios': m1_ratios,
            'm2_ratios': m2_ratios,
            'f_a_prime': 1 - propensities,
            'f_a': propensities
        }
    
    def estimate_effect(self, data, proxy_setting='z', a_prime=0, a=1):
        """Calculate PMR estimate and its confidence interval"""
        if proxy_setting == 'z':
            data_for_est = data.copy()
        elif proxy_setting == 'w':
            data_for_est = data.copy()
            data_for_est['z'] = data['w']
        elif proxy_setting == 'zw':
            data_for_est = data.copy()
            data_for_est['z'] = np.hstack([data['z'], data['w']])
        else:
            raise ValueError("proxy_setting must be 'z', 'w', or 'zw'")
        
        nuisance = self.fit_and_predict_all_nuisance(data_for_est, a_prime, a)
        n = len(data['y'])
        a_values = data_for_est['a'].flatten()
        
        term1 = np.zeros(n)
        term2 = np.zeros(n)
        term3 = np.zeros(n)
        
        mask_a_prime = (a_values == a_prime)
        mask_a = (a_values == a)
        
        if np.sum(mask_a_prime) > 0:
            term1[mask_a_prime] = (
                1 / nuisance['f_a_prime'][mask_a_prime].flatten() *
                nuisance['m2_ratios'][mask_a_prime].flatten() *
                (data_for_est['y'][mask_a_prime].flatten() - nuisance['b'][mask_a_prime])
            )
        
        if np.sum(mask_a) > 0:
            term2[mask_a] = (
                1 / nuisance['f_a'][mask_a].flatten() *
                nuisance['m1_ratios'][mask_a].flatten() *
                (nuisance['b'][mask_a] - nuisance['b_prime'][mask_a])
            )
        
        if np.sum(mask_a_prime) > 0:
            term3[mask_a_prime] = (
                1 / nuisance['f_a_prime'][mask_a_prime].flatten() *
                (nuisance['b_prime'][mask_a_prime] - nuisance['b_double_prime'][mask_a_prime])
            )
        
        if_values = term1 + term2 + term3 + nuisance['b_double_prime']
        
        beta_mr = np.mean(if_values)
        se_mr = np.std(if_values, ddof=1) / np.sqrt(n)
        
        ci_lower = beta_mr - 1.96 * se_mr
        ci_upper = beta_mr + 1.96 * se_mr
        
        return {
            'beta_mr': beta_mr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }


def run_single_experiment(dims_name, dims, proxy_setting, run_idx, n_samples=1000):
    """Run a single Monte Carlo simulation iteration"""
    try:
        seed = 42 + hash(f"{dims_name}_{proxy_setting}_{run_idx}") % 100000
        xdim, zdim, wdim, m1dim, m2dim = dims
        
        datagen = linear_dgp.LinearTwoMediatorsProxDGP(
            xdim=xdim, zdim=zdim, wdim=wdim, 
            m1dim=m1dim, m2dim=m2dim, seed=seed
        )
        
        data = datagen.sample_dataset(n=n_samples, seed=seed)
        true_psi = datagen.true_psi(data)
        
        estimator = CorrectMRCrossFittedEstimator(random_state=seed)
        estimates = estimator.estimate_effect(data, proxy_setting=proxy_setting)
        
        beta_mr = estimates['beta_mr']
        ci_lower = estimates['ci_lower']
        ci_upper = estimates['ci_upper']
        ci_length = ci_upper - ci_lower
        coverage = 1 if (ci_lower <= true_psi <= ci_upper) else 0
        
        return {
            'seed': seed,
            'proxy_setting': proxy_setting,
            'dims': dims_name,
            'sample_size': n_samples,
            'true_value': true_psi,
            'estimate': beta_mr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_length': ci_length,
            'coverage': coverage
        }
    except Exception as e:
        print(f"Run {run_idx} failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="PMR Monte Carlo Parallel Simulation")
    parser.add_argument('--dims', type=str, required=True, choices=['53311', '32211'], help='Specified dimension configuration')
    parser.add_argument('--proxy', type=str, required=True, choices=['z', 'w', 'zw'], help='Specified proxy variable')
    parser.add_argument('--n_cores', type=int, default=3, help='Number of CPU cores for parallelization')
    parser.add_argument('--repeats', type=int, default=300, help='Total number of Monte Carlo iterations')
    args = parser.parse_args()

    dims_configs = {
        '53311': (5, 3, 3, 1, 1),
        '32211': (3, 2, 2, 1, 1)
    }
    
    dims_name = args.dims
    dims = dims_configs[dims_name]
    proxy = args.proxy
    n_cores = args.n_cores
    n_repeats = args.repeats
    sample_size = 1000
    chunk_size = 10  
    
    print("="*60)
    print(f"Starting Experiment | Dims: {dims_name} | Proxy: {proxy} | Cores: {n_cores}")
    print("="*60)

    save_dir = "simulation_results" 
    os.makedirs(save_dir, exist_ok=True)
    
    output_file = os.path.join(save_dir, f"pmr_results_{dims_name}_{proxy}.csv")

    all_results = []
    
    for chunk_start in range(0, n_repeats, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_repeats)
        run_indices = range(chunk_start, chunk_end)
        
        chunk_results = Parallel(n_jobs=n_cores)(
            delayed(run_single_experiment)(dims_name, dims, proxy, run_idx, sample_size) 
            for run_idx in run_indices
        )
        
        valid_results = [res for res in chunk_results if res is not None]
        all_results.extend(valid_results)
        
        if all_results:
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(output_file, index=False)
            
            current_coverage = temp_df['coverage'].mean()
            current_bias = (temp_df['estimate'] - temp_df['true_value']).mean()
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {len(temp_df)}/{n_repeats} | "
                  f"Coverage: {current_coverage*100:.1f}% | Bias: {current_bias:.4f} | Table updated...")

    print("="*60)
    print(f"Experiment completed! Final results saved at: {output_file}")


if __name__ == "__main__":
    main()