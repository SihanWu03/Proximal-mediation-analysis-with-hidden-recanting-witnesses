import numpy as np
from tqdm import tqdm
import os
import linear_dgp
import allestimator
from sklearn.model_selection import KFold
from scipy import stats

estimator = allestimator.AllEstimator

def compute_influence_function_pmr(model, data, psi_hat):
    """
    Calculate the Influence Function (IF) for the PMR estimator.
    Calculates the IF for each sample based on the EIF formula of PMR.
    """
    # Get data
    a = data['a'][:, 0]
    y = data['y'][:, 0]
    x = data['x']
    w = data['w']
    z = data['z']
    m2 = data['m2']
    n = len(x)
    
    # Get propensity scores
    try:
        probs = model.treatment_prob_model.predict_proba(x)
        p_a1 = probs[:, 1]  # P(A=1|X)
        p_a0 = probs[:, 0]  # P(A=0|X)
    except:
        # Fallback if the model lacks predict_proba method
        p_a1 = model.treatment_prob_model.predict(x).flatten()
        p_a0 = 1 - p_a1
    
    # Predict h functions
    w_m2_x = np.hstack((w, m2, x))
    h0_pred = model.h0_fn(w_m2_x).flatten()
    
    w_x = np.hstack((w, x))
    h1_pred = model.h1_fn(w_x).flatten()
    
    # Predict q functions
    z_m2_x = np.hstack((z, m2, x))
    q0_pred = model.q0_fn(z_m2_x).flatten()
    
    z_x = np.hstack((z, x))
    q1_pred = model.q1_fn(z_x).flatten()
    
    # Predict eta (η) function
    eta_pred = model.eta.predict(x).flatten()
    
    # Indicator functions
    a1_ind = (a == 1).astype(float)
    a0_ind = (a == 0).astype(float)
    
    # Calculate PMR's EIF
    # Based on Robins et al. (2008) EIF formula
    eif = (
        (a1_ind / p_a1) * q1_pred * (h0_pred - h1_pred) +
        (a0_ind / p_a1) * q0_pred * (y - h0_pred) +
        (a0_ind / p_a0) * (h1_pred - eta_pred) +
        eta_pred - psi_hat
    )
    
    return eif

def cross_fitting_estimate(datagen, data_all, n_splits=5, num_runs=10):
    """Estimate potential outcomes using cross-fitting and compute IF-based confidence intervals"""
    
    # Calculate the true value
    true_psi = datagen.true_psi(data=data_all)
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store estimates for each fold
    fold_estimates_por = []
    fold_estimates_pipw = []
    fold_estimates_phe = []
    fold_estimates_pmr = []
    
    # Store sample-level estimates for each fold (for variance calculation)
    fold_sample_estimates_pmr = []
    
    # Store sample-level IF values (for variance calculation)
    sample_if_pmr = np.zeros(len(data_all['x']))
    
    # Get total number of samples
    n_total = len(data_all['x'])
    
    # Get all data indices
    indices = np.arange(n_total)
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
        # Prepare train+val data (4/5 of total data)
        train_val_data = {k: v[train_val_idx] for k, v in data_all.items()}
        
        # Further split train+val data into train and val sets (4:1 ratio)
        n_train_val = len(train_val_idx)
        split_point = int(1 * n_train_val)
        
        # Randomly shuffle the train+val indices
        np.random.seed(fold_idx)
        train_val_indices = np.random.permutation(n_train_val)
        
        train_idx_fold = train_val_indices[:split_point]
        val_idx_fold = train_val_indices[:split_point]
        
        # Split data
        data_fit = {k: v[train_idx_fold] for k, v in train_val_data.items()}
        data_val = {k: v[val_idx_fold] for k, v in train_val_data.items()}
        
        # Test data (1/5 of total data)
        data_test = {k: v[test_idx] for k, v in data_all.items()}
        
        try:
            # Train the model
            model = estimator(num_runs=num_runs)
            model.fit(fit_data=data_fit, val_data=data_val)
            
            # Evaluate all four estimators on the test set
            por_estimates = np.array(model.evaluatepor(eval_data=data_test))
            pipw_estimates = np.array(model.evaluatepipw(eval_data=data_test))
            phe_estimates = np.array(model.evaluatephe(eval_data=data_test))
            pmr_estimates = np.array(model.evaluatepmr(eval_data=data_test))
            
            # Calculate PMR point estimate on the test set
            fold_pmr_estimate = np.mean(pmr_estimates)
            
            # Store all sample PMR estimates for this fold
            fold_sample_estimates_pmr.append(pmr_estimates)
            
            # Calculate influence functions for the test set
            test_if = compute_influence_function_pmr(model, data_test, fold_pmr_estimate)
            
            # Store IF values into corresponding sample indices
            sample_if_pmr[test_idx] = test_if
            
            # Calculate the mean for each estimator
            fold_estimate_por = np.mean(por_estimates)
            fold_estimate_pipw = np.mean(pipw_estimates)
            fold_estimate_phe = np.mean(phe_estimates)
            
            # Store estimates for each fold
            fold_estimates_por.append(fold_estimate_por)
            fold_estimates_pipw.append(fold_estimate_pipw)
            fold_estimates_phe.append(fold_estimate_phe)
            fold_estimates_pmr.append(fold_pmr_estimate)
            
        except Exception as e:
            error_msg = f"Fold {fold_idx+1} failed: {str(e)}"
            print(f"Error details: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    # Calculate the final cross-fitted estimate (average over all folds)
    if len(fold_estimates_por) > 0:
        cross_fit_estimate_por = np.mean(fold_estimates_por)
        cross_fit_estimate_pipw = np.mean(fold_estimates_pipw)
        cross_fit_estimate_phe = np.mean(fold_estimates_phe)
        cross_fit_estimate_pmr = np.mean(fold_estimates_pmr)
    else:
        cross_fit_estimate_por = 0.0
        cross_fit_estimate_pipw = 0.0
        cross_fit_estimate_phe = 0.0
        cross_fit_estimate_pmr = 0.0
    
    # Calculate PMR confidence interval - based on the IF method
    if len(fold_estimates_pmr) > 1:
        # Method 1: IF-based variance estimation (Recommended)
        # Calculate variance of the influence function
        if_variance = np.var(sample_if_pmr, ddof=1)
        
        # Variance of the estimator = IF variance / sample size
        pmr_variance_if = if_variance / n_total
        pmr_se_if = np.sqrt(pmr_variance_if)
        
        # Use normal approximation (based on asymptotic theory)
        z_critical = stats.norm.ppf(0.975)  # 95% confidence level
        
        # Calculate 95% confidence interval
        pmr_ci_lower = cross_fit_estimate_pmr - z_critical * pmr_se_if
        pmr_ci_upper = cross_fit_estimate_pmr + z_critical * pmr_se_if
        
        # Check if true value is covered by the CI
        pmr_ci_cover = (pmr_ci_lower <= true_psi <= pmr_ci_upper)
        
        # For comparison, calculate the fold-level variance (original method)
        n_folds = len(fold_estimates_pmr)
        fold_estimates_array = np.array(fold_estimates_pmr)
        fold_variance = np.var(fold_estimates_array, ddof=1)
        pmr_variance_fold = fold_variance / n_folds
        pmr_se_fold = np.sqrt(pmr_variance_fold)
        
        # Store standard errors from both methods
        pmr_se = pmr_se_if  # Use IF method as primary SE
        pmr_se_original = pmr_se_fold  # Original method for comparison
        
        # Print diagnostic information (optional)
        print(f"Diagnostics: n={n_total}, folds={n_folds}, IF_SE={pmr_se_if:.6f}, Fold_SE={pmr_se_fold:.6f}")
        
    else:
        pmr_se = 0.0
        pmr_se_original = 0.0
        pmr_ci_lower = cross_fit_estimate_pmr
        pmr_ci_upper = cross_fit_estimate_pmr
        pmr_ci_cover = True  # Degenerates to point estimate when only 1 fold is available
    
    # Return estimates for all four estimators and the PMR CI
    return {
        'por': cross_fit_estimate_por,
        'pipw': cross_fit_estimate_pipw,
        'phe': cross_fit_estimate_phe,
        'pmr': cross_fit_estimate_pmr,
        'pmr_se': pmr_se,  # IF method SE
        'pmr_se_original': pmr_se_original,  # Original method SE
        'pmr_ci_lower': pmr_ci_lower,
        'pmr_ci_upper': pmr_ci_upper,
        'pmr_ci_cover': pmr_ci_cover,
        'pmr_ci_width': pmr_ci_upper - pmr_ci_lower,
        'pmr_if_variance': if_variance if len(fold_estimates_pmr) > 1 else 0.0
    }, true_psi

def main():
    import os
    import pandas as pd
    from datetime import datetime
    
    # Create results directory
    results_dir = "experiment_results11111"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Define experiment configurations
    experiment_configs = [
        {
            'name': 'group3_11111_dims',
            'dimensions': (1, 1, 1, 1, 1),  # xdim, zdim, wdim, m1dim, m2dim
            'sample_sizes': [200, 500, 1000, 2000],
            'n_experiments': 300,
            'num_runs': 25,
            'n_splits': 5
        },
        
        # Other configurations...
    ]
    
    # Total experiment statistics
    total_experiments = 0
    completed_experiments = 0
    
    # Calculate total number of experiments
    for config in experiment_configs:
        total_experiments += len(config['sample_sizes']) * config['n_experiments']
    
    print("="*80)
    print(f"Starting large-scale experiments (with CI calculation)")
    print(f"Total experiments: {total_experiments}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Iterate over all configurations
    for config_idx, config in enumerate(experiment_configs, 1):
        group_name = config['name']
        xdim, zdim, wdim, m1dim, m2dim = config['dimensions']
        
        print(f"\n{'='*60}")
        print(f"Starting group {config_idx} experiments: {group_name}")
        print(f"Dimension config: xdim={xdim}, zdim={zdim}, wdim={wdim}, m1dim={m1dim}, m2dim={m2dim}")
        print(f"{'='*60}")
        
        # Create directory for this group
        group_dir = os.path.join(results_dir, group_name)
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
        
        # Iterate over all sample sizes
        for sample_size in config['sample_sizes']:
            print(f"\n--- Sample size n = {sample_size} ---")
            
            # Create CSV file for this sample size
            csv_filename = os.path.join(group_dir, f"n{sample_size}_results.csv")
            
            # Prepare list to store data
            all_results = []
            
            # Generate fixed random seed range
            base_seed = config_idx * 32000 + sample_size * 100
            
            # Use progress bar
            for exp_idx in tqdm(range(config['n_experiments']), desc=f"n={sample_size}"):
                # Calculate seed for the current experiment
                seed = base_seed + exp_idx
                
                try:
                    # Create data generator
                    datagen = linear_dgp.LinearTwoMediatorsProxDGP(
                        xdim, zdim, wdim, m1dim, m2dim, seed=seed
                    )
                    
                    # Generate full dataset
                    data_all = datagen.sample_dataset(sample_size, seed=seed)
                    
                    # Estimate using cross-fitting (with CI)
                    estimates_dict, true_psi = cross_fitting_estimate(
                        datagen=datagen,
                        data_all=data_all,
                        n_splits=config['n_splits'],
                        num_runs=config['num_runs']
                    )
                    
                    # Create result record
                    result_record = {
                        'group': group_name,
                        'xdim': xdim,
                        'zdim': zdim,
                        'wdim': wdim,
                        'm1dim': m1dim,
                        'm2dim': m2dim,
                        'sample_size': sample_size,
                        'seed': seed,
                        'true_psi': float(true_psi),
                        'por_estimate': float(estimates_dict['por']),
                        'pipw_estimate': float(estimates_dict['pipw']),
                        'phe_estimate': float(estimates_dict['phe']),
                        'pmr_estimate': float(estimates_dict['pmr']),
                        'pmr_se': float(estimates_dict['pmr_se']),
                        'pmr_ci_lower': float(estimates_dict['pmr_ci_lower']),
                        'pmr_ci_upper': float(estimates_dict['pmr_ci_upper']),
                        'pmr_ci_cover': bool(estimates_dict['pmr_ci_cover']),
                        'pmr_ci_width': float(estimates_dict['pmr_ci_width']),
                        'por_mse': float((estimates_dict['por'] - true_psi) ** 2),
                        'pipw_mse': float((estimates_dict['pipw'] - true_psi) ** 2),
                        'phe_mse': float((estimates_dict['phe'] - true_psi) ** 2),
                        'pmr_mse': float((estimates_dict['pmr'] - true_psi) ** 2),
                        'experiment_index': exp_idx,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    all_results.append(result_record)
                    completed_experiments += 1
                    
                    # Save periodically
                    if (exp_idx + 1) % 10 == 0 or (exp_idx + 1) == config['n_experiments']:
                        df = pd.DataFrame(all_results)
                        df.to_csv(csv_filename, index=False)
                        print(f"  Saved {len(all_results)} records to {csv_filename}")
                
                except Exception as e:
                    error_record = {
                        'group': group_name,
                        'sample_size': sample_size,
                        'seed': seed,
                        'experiment_index': exp_idx,
                        'error': str(e),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    print(f"Warning: Experiment {exp_idx} (seed={seed}) failed: {str(e)}")
                    
                    error_filename = os.path.join(group_dir, f"n{sample_size}_errors.csv")
                    error_df = pd.DataFrame([error_record])
                    if os.path.exists(error_filename):
                        error_df.to_csv(error_filename, mode='a', header=False, index=False)
                    else:
                        error_df.to_csv(error_filename, index=False)
            
            # Final save and statistics
            if all_results:
                df = pd.DataFrame(all_results)
                df.to_csv(csv_filename, index=False)
                
                print(f"\nSample size {sample_size} completed!")
                print(f"Successful experiments: {len(all_results)}")
                print(f"Saved file: {csv_filename}")
                
                # Print CI coverage statistics
                if 'pmr_ci_cover' in df.columns:
                    coverage_rate = df['pmr_ci_cover'].mean() * 100
                    avg_ci_width = df['pmr_ci_width'].mean()
                    avg_se = df['pmr_se'].mean()
                    
                    print(f"PMR 95% CI coverage rate: {coverage_rate:.2f}%")
                    print(f"Average CI width: {avg_ci_width:.6f}")
                    print(f"Average SE: {avg_se:.6f}")
                    
                    # Calculate average MSE for each estimator
                    for estimator_name in ['por', 'pipw', 'phe', 'pmr']:
                        mse_col = f'{estimator_name}_mse'
                        if mse_col in df.columns:
                            mean_mse = df[mse_col].mean()
                            std_mse = df[mse_col].std()
                            print(f"  {estimator_name.upper()} MSE: Mean={mean_mse:.6f}, Std={std_mse:.6f}")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Completed experiments: {completed_experiments}/{total_experiments}")
    print("="*80)
    
    # Create summary report
    create_summary_report_with_ci(results_dir)

def create_summary_report_with_ci(results_dir):
    """Create summary report of experiments with CI statistics"""
    import pandas as pd
    import glob
    
    print("\n" + "="*80)
    print("Create summary report of experiments (with CI statistics)")
    print("="*80)
    
    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, "**", "*_results.csv"), recursive=True)
    
    if not result_files:
        print("No result files found")
        return
    
    all_dfs = []
    
    for file in result_files:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
            print(f"Reading file: {file} ({len(df)} records)")
        except Exception as e:
            print(f"Failed to read file {file}: {e}")
    
    if all_dfs:
        # Merge all data
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save merged data
        combined_file = os.path.join(results_dir, "all_experiments_combined.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"\nMerged data saved to: {combined_file}")
        print(f"Total records: {len(combined_df)}")
        
        # Create summary statistics grouped by group and sample size
        summary_stats = []
        
        # Group by group and sample size
        grouped = combined_df.groupby(['group', 'sample_size'])
        
        for (group, sample_size), group_df in grouped:
            summary = {
                'group': group,
                'sample_size': sample_size,
                'n_experiments': len(group_df),
                'true_mean': group_df['true_psi'].mean(),
                'true_std': group_df['true_psi'].std(),
            }
            
            # Confidence interval statistics
            if 'pmr_ci_cover' in group_df.columns:
                summary['pmr_ci_coverage'] = group_df['pmr_ci_cover'].mean() * 100
                summary['pmr_ci_width_mean'] = group_df['pmr_ci_width'].mean()
                summary['pmr_ci_width_std'] = group_df['pmr_ci_width'].std()
                summary['pmr_se_mean'] = group_df['pmr_se'].mean()
                summary['pmr_se_std'] = group_df['pmr_se'].std()
            
            # Calculate MSE statistics for each estimator
            for estimator in ['por', 'pipw', 'phe', 'pmr']:
                mse_col = f'{estimator}_mse'
                if mse_col in group_df.columns:
                    summary[f'{estimator}_mse_mean'] = group_df[mse_col].mean()
                    summary[f'{estimator}_mse_std'] = group_df[mse_col].std()
                    summary[f'{estimator}_mse_median'] = group_df[mse_col].median()
                    
                # Calculate bias
                est_col = f'{estimator}_estimate'
                if est_col in group_df.columns:
                    bias = group_df[est_col] - group_df['true_psi']
                    summary[f'{estimator}_bias_mean'] = bias.mean()
                    summary[f'{estimator}_bias_std'] = bias.std()
            
            summary_stats.append(summary)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_stats)
        
        # Save summary statistics
        summary_file = os.path.join(results_dir, "experiment_summary_with_ci.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary statistics saved to: {summary_file}")
        
        # Print summary statistics
        print("\nSummary statistics (with CI):")
        print(summary_df.to_string())
        
        # Create detailed comparison by group
        print("\n" + "="*80)
        print("Detailed statistical comparison")
        print("="*80)
        
        for group in summary_df['group'].unique():
            group_data = summary_df[summary_df['group'] == group]
            print(f"\n{group}:")
            for _, row in group_data.iterrows():
                print(f"\n  n={row['sample_size']} (experiments={row['n_experiments']}):")
                print(f"    True value: {row['true_mean']:.4f} ± {row['true_std']:.4f}")
                
                if 'pmr_ci_coverage' in row:
                    print(f"    PMR CI coverage: {row['pmr_ci_coverage']:.2f}%")
                    print(f"    PMR CI width: {row['pmr_ci_width_mean']:.6f} ± {row['pmr_ci_width_std']:.6f}")
                    print(f"    PMR SE: {row['pmr_se_mean']:.6f} ± {row['pmr_se_std']:.6f}")
                
                for estimator in ['por', 'pipw', 'phe', 'pmr']:
                    mse_mean = row.get(f'{estimator}_mse_mean', None)
                    bias_mean = row.get(f'{estimator}_bias_mean', None)
                    if mse_mean is not None and bias_mean is not None:
                        print(f"    {estimator.upper()}: Bias={bias_mean:.6f}, MSE={mse_mean:.6f}")


if __name__ == "__main__":
    main()