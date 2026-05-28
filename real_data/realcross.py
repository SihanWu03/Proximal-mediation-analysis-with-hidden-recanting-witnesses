import numpy as np
import pickle
import os
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Import custom modules
import allestimator

estimator = allestimator.AllEstimator

def run_naive_dr_fold(data_fit, data_test):
    """
    Calculate the naive DR estimator and return the point-wise score for each sample
    to construct the DR influence function (EIF) externally.
    """
    X_train = data_fit['x']
    A_train = data_fit['a'].flatten()
    Y_train = data_fit['y'].flatten()
    
    X_test = data_test['x']
    A_test = data_test['a'].flatten()
    Y_test = data_test['y'].flatten()
    
    # 1. Propensity score model P(A|X) 
    pi_model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=100)
    pi_model.fit(X_train, A_train)
    p_a0_test = np.clip(pi_model.predict_proba(X_test)[:, 0], 0.05, 0.95)
    
    # 2. Outcome regression model E(Y|A=0, X)
    mu_model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=100)
    mu_model.fit(X_train[A_train == 0], Y_train[A_train == 0])
    mu0_test = mu_model.predict(X_test)
    
    # 3. Execute DR formula to calculate the point-wise score of Y(0) (i.e., \psi(O_i))
    a0_ind = (A_test == 0).astype(float)
    dr_y0_pointwise = (a0_ind * (Y_test - mu0_test) / p_a0_test) + mu0_test
    
    # Return the score for each sample and the mean for this fold
    return dr_y0_pointwise, np.mean(dr_y0_pointwise)

def compute_influence_function_pmr(model, data, psi_hat):
    """Compute the PMR influence function."""
    a, y, x, w, z, m2 = data['a'][:, 0], data['y'][:, 0], data['x'], data['w'], data['z'], data['m2']
    probs = model.treatment_prob_model.predict_proba(x)
    p_a1, p_a0 = probs[:, 1], probs[:, 0]
    
    h0_pred = model.h0_fn(np.hstack((w, m2, x))).flatten()
    h1_pred = model.h1_fn(np.hstack((w, x))).flatten()
    q0_pred = model.q0_fn(np.hstack((z, m2, x))).flatten()
    q1_pred = model.q1_fn(np.hstack((z, x))).flatten()
    eta_pred = model.eta.predict(x).flatten()
    
    eif = ((a == 1).astype(float) / p_a1) * q1_pred * (h0_pred - h1_pred) + \
          ((a == 0).astype(float) / p_a1) * q0_pred * (y - h0_pred) + \
          ((a == 0).astype(float) / p_a0) * (h1_pred - eta_pred) + \
          eta_pred - psi_hat
    return eif

def cross_fitting_estimate_realdata(data_dict, n_splits=5, num_runs=10, seed=100):
    n = len(data_dict['x'])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    fold_results = {'por': [], 'pipw': [], 'phe': [], 'pmr': [], 'dr_y0': []}
    
    # Simultaneously record the sample-level influence functions (EIF) for PMR and DR
    sample_if_pmr = np.zeros(n)
    sample_if_dr = np.zeros(n)

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        print(f"Fitting fold {fold_idx + 1}/{n_splits}...")
        
        tv_data = {k: v[train_val_idx] for k, v in data_dict.items()}
        test_data = {k: v[test_idx] for k, v in data_dict.items()}
        
        fit_data = tv_data
        val_data = tv_data

        # 1. PMR model fitting
        model = estimator(num_runs=num_runs)
        model.fit(fit_data=fit_data, val_data=val_data)

        # 2. Calculate various path estimators and the PMR EIF
        fold_results['por'].append(np.mean(model.evaluatepor(test_data)))
        fold_results['pipw'].append(np.mean(model.evaluatepipw(test_data)))
        fold_results['phe'].append(np.mean(model.evaluatephe(test_data)))
        
        pmr_est = np.mean(model.evaluatepmr(test_data))
        fold_results['pmr'].append(pmr_est)
        sample_if_pmr[test_idx] = compute_influence_function_pmr(model, test_data, pmr_est)

        # 3. Calculate the DR estimator and the DR EIF
        dr_scores, dr_est = run_naive_dr_fold(fit_data, test_data)
        fold_results['dr_y0'].append(dr_est)
        
        # The DR EIF is the point-wise score minus the mean
        sample_if_dr[test_idx] = dr_scores - dr_est

    # Summarize point estimates
    res = {f'{k}_estimate': np.mean(v) for k, v in fold_results.items()}
    
    # -------------------------------------------------------------
    # Core calculation: Calculate inference statistics for the target difference (PMR - DR)
    # -------------------------------------------------------------
    res['diff_estimate'] = res['pmr_estimate'] - res['dr_y0_estimate']
    
    # Linear combination of influence functions: EIF(A - B) = EIF(A) - EIF(B)
    sample_if_diff = sample_if_pmr - sample_if_dr
    
    # Calculate the variance and standard error of the difference
    diff_var = np.var(sample_if_diff, ddof=1) / n
    diff_se = np.sqrt(diff_var)
    z_val = stats.norm.ppf(0.975)
    
    res.update({
        'diff_var': diff_var,
        'diff_se': diff_se,
        'diff_ci': [res['diff_estimate'] - z_val * diff_se, res['diff_estimate'] + z_val * diff_se]
    })
    
    # Calculate inference for the original PMR (for reference only)
    pmr_se = np.sqrt(np.var(sample_if_pmr, ddof=1) / n)
    res.update({
        'pmr_se': pmr_se,
        'pmr_ci': [res['pmr_estimate'] - z_val*pmr_se, res['pmr_estimate'] + z_val*pmr_se]
    })
    
    return res

def main():
    # Dynamically locate the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'processed_data_dict.pkl')
    
    # Note: If you choose to put the data in a dedicated 'data' subfolder in your repo, 
    # you can change the above line to:
    # data_path = os.path.join(current_dir, 'data', 'processed_data_dict.pkl')

    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)

    print("Starting analysis...")
    res = cross_fitting_estimate_realdata(data_dict)

    print("\n" + "="*60)
    print("📋 Causal Path Effect Comparative Analysis (PMR Effect vs. Naive Baseline E[Y(0)])")
    print("="*60)
    print(f"[1] Baseline Expectation E[Y(0)] (DR Estimate):          {res['dr_y0_estimate']:.4f}")
    print(f"[2] Path Effect E[Y(M2(1), M1(0), 0)] (PMR):             {res['pmr_estimate']:.4f}")
    print("-" * 60)
    print(f"🎯 Core Target Difference [2] - [1]:                     {res['diff_estimate']:.4f}")
    print(f"   => Difference Variance:                               {res['diff_var']:.6f}")
    print(f"   => Difference Standard Error (SE):                    {res['diff_se']:.4f}")
    print(f"   => Difference 95% Confidence Interval (CI):           [{res['diff_ci'][0]:.4f}, {res['diff_ci'][1]:.4f}]")
    print("="*60)
    print(f"Other supplementary information:")
    print(f"PMR isolated 95% CI: [{res['pmr_ci'][0]:.4f}, {res['pmr_ci'][1]:.4f}]")
    print(f"POR={res['por_estimate']:.2f}, PIPW={res['pipw_estimate']:.2f}, PHE={res['phe_estimate']:.2f}")

if __name__ == "__main__":
    main()