import numpy as np
import pickle
import os
import json
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 导入自定义模块
import por, pipw, phe, pmr, allestimator

estimator = allestimator.AllEstimator

def run_naive_dr_fold(data_fit, data_test):
    """
    【修改】：不仅返回 DR 的均值估计量，还要返回每个样本的 point-wise score，
    从而可以在外部构建 DR 的影响函数 (EIF)
    """
    X_train = data_fit['x']
    A_train = data_fit['a'].flatten()
    Y_train = data_fit['y'].flatten()
    
    X_test = data_test['x']
    A_test = data_test['a'].flatten()
    Y_test = data_test['y'].flatten()
    
    # 1. 倾向得分模型 P(A|X) 
    pi_model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=100)
    pi_model.fit(X_train, A_train)
    p_a0_test = np.clip(pi_model.predict_proba(X_test)[:, 0], 0.05, 0.95)
    
    # 2. 结果回归模型 E(Y|A=0, X)
    mu_model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=100)
    mu_model.fit(X_train[A_train == 0], Y_train[A_train == 0])
    mu0_test = mu_model.predict(X_test)
    
    # 3. 执行 DR 公式计算 Y(0) 的 point-wise score (即 \psi(O_i))
    a0_ind = (A_test == 0).astype(float)
    dr_y0_pointwise = (a0_ind * (Y_test - mu0_test) / p_a0_test) + mu0_test
    
    # 返回每个样本的 score 和 该折的均值
    return dr_y0_pointwise, np.mean(dr_y0_pointwise)

def compute_influence_function_pmr(model, data, psi_hat):
    """原有 PMR 影响函数逻辑保持不变"""
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
    
    # 【新增】：同时记录 PMR 和 DR 的样本级影响函数 (EIF)
    sample_if_pmr = np.zeros(n)
    sample_if_dr = np.zeros(n)

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        print(f"正在拟合第 {fold_idx + 1}/{n_splits} 折...")
        
        tv_data = {k: v[train_val_idx] for k, v in data_dict.items()}
        test_data = {k: v[test_idx] for k, v in data_dict.items()}
        
        fit_data = tv_data
        val_data = tv_data

        # 1. PMR 模型拟合
        model = estimator(num_runs=num_runs)
        model.fit(fit_data=fit_data, val_data=val_data)

        # 2. 计算各个路径估计量与 PMR EIF
        fold_results['por'].append(np.mean(model.evaluatepor(test_data)))
        fold_results['pipw'].append(np.mean(model.evaluatepipw(test_data)))
        fold_results['phe'].append(np.mean(model.evaluatephe(test_data)))
        
        pmr_est = np.mean(model.evaluatepmr(test_data))
        fold_results['pmr'].append(pmr_est)
        sample_if_pmr[test_idx] = compute_influence_function_pmr(model, test_data, pmr_est)

        # 3. 计算 DR 估计量与 DR EIF
        dr_scores, dr_est = run_naive_dr_fold(fit_data, test_data)
        fold_results['dr_y0'].append(dr_est)
        # DR 的 EIF 就是 point-wise score 减去均值
        sample_if_dr[test_idx] = dr_scores - dr_est

    # 汇总点估计
    res = {f'{k}_estimate': np.mean(v) for k, v in fold_results.items()}
    
    # -------------------------------------------------------------
    # 【核心计算】：计算 目标差值 (PMR - DR) 的推断统计量
    # -------------------------------------------------------------
    res['diff_estimate'] = res['pmr_estimate'] - res['dr_y0_estimate']
    
    # 影响函数的线性组合: EIF(A - B) = EIF(A) - EIF(B)
    sample_if_diff = sample_if_pmr - sample_if_dr
    
    # 计算差异的方差与标准误
    diff_var = np.var(sample_if_diff, ddof=1) / n
    diff_se = np.sqrt(diff_var)
    z_val = stats.norm.ppf(0.975)
    
    res.update({
        'diff_var': diff_var,
        'diff_se': diff_se,
        'diff_ci': [res['diff_estimate'] - z_val * diff_se, res['diff_estimate'] + z_val * diff_se]
    })
    
    # 附带计算原 PMR 的推断 (仅作参考)
    pmr_se = np.sqrt(np.var(sample_if_pmr, ddof=1) / n)
    res.update({
        'pmr_se': pmr_se,
        'pmr_ci': [res['pmr_estimate'] - z_val*pmr_se, res['pmr_estimate'] + z_val*pmr_se]
    })
    
    return res

def main():
    base_dir = r'D:\code\simulation\simulation'
    with open(os.path.join(base_dir, 'processed_data_dict.pkl'), 'rb') as f:
        data_dict = pickle.load(f)

    print("开始分析...")
    res = cross_fitting_estimate_realdata(data_dict)

    print("\n" + "="*60)
    print("📋 因果路径效应对比分析 (PMR 效应 vs. 朴素基准 E[Y(0)])")
    print("="*60)
    print(f"[1] 基准期望 E[Y(0)] (DR估计):          {res['dr_y0_estimate']:.4f}")
    print(f"[2] 路径效应 E[Y(M2(1), M1(0), 0)] (PMR): {res['pmr_estimate']:.4f}")
    print("-" * 60)
    print(f"🎯 核心目标差异值 [2] - [1]:             {res['diff_estimate']:.4f}")
    print(f"   => 差异方差 (Variance):               {res['diff_var']:.6f}")
    print(f"   => 差异标准误 (SE):                   {res['diff_se']:.4f}")
    print(f"   => 差异 95% 置信区间 (CI):            [{res['diff_ci'][0]:.4f}, {res['diff_ci'][1]:.4f}]")
    print("="*60)
    print(f"其他补充信息:")
    print(f"PMR 单独 95% CI: [{res['pmr_ci'][0]:.4f}, {res['pmr_ci'][1]:.4f}]")
    print(f"POR={res['por_estimate']:.2f}, PIPW={res['pipw_estimate']:.2f}, PHE={res['phe_estimate']:.2f}")

if __name__ == "__main__":
    main()