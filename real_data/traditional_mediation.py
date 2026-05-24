import numpy as np
import pickle
import os
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def cross_fit_traditional_mediation(data_dict, n_splits=5, seed=100):
    n = len(data_dict['x'])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # 存储各项期望的折均值
    results = {'theta_11': [], 'theta_00': [], 'theta_10': [], 'theta_01': []}
    
    # 💥【核心修改】：全局记录四个基础参数的样本级 point-wise score，用于后续严格构建差值的 EIF
    score_11_all = np.zeros(n)
    score_00_all = np.zeros(n)
    score_10_all = np.zeros(n)
    score_01_all = np.zeros(n)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        print(f"正在拟合第 {fold_idx + 1}/{n_splits} 折...")
        
        # --- 1. 数据切分与展平 ---
        X_tr, A_tr, M_tr, Y_tr = data_dict['x'][train_idx], data_dict['a'][train_idx].flatten(), data_dict['m2'][train_idx].flatten(), data_dict['y'][train_idx].flatten()
        X_te, A_te, M_te, Y_te = data_dict['x'][test_idx], data_dict['a'][test_idx].flatten(), data_dict['m2'][test_idx].flatten(), data_dict['y'][test_idx].flatten()
        
        a1_ind = (A_te == 1).astype(float)
        a0_ind = (A_te == 0).astype(float)

        # ==========================================
        # 步骤 1 & 2: 完全对齐 Naive DR 的边缘效应估计 (调高树精度 300, 10, 2)
        # ==========================================
        pi_x_model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=seed)
        pi_x_model.fit(X_tr, A_tr)
        p_a0_te = np.clip(pi_x_model.predict_proba(X_te)[:, 0], 0.05, 0.95)
        p_a1_te = np.clip(pi_x_model.predict_proba(X_te)[:, 1], 0.05, 0.95)

        mu0_model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=seed)
        mu0_model.fit(X_tr[A_tr == 0], Y_tr[A_tr == 0])
        mu_0_x_te = mu0_model.predict(X_te)

        mu1_model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=seed)
        mu1_model.fit(X_tr[A_tr == 1], Y_tr[A_tr == 1])
        mu_1_x_te = mu1_model.predict(X_te)

        eif_00 = (a0_ind * (Y_te - mu_0_x_te) / p_a0_te) + mu_0_x_te
        eif_11 = (a1_ind * (Y_te - mu_1_x_te) / p_a1_te) + mu_1_x_te

        # ==========================================
        # 步骤 3 & 4: 中介路径专用模型 (调高树精度)
        # ==========================================
        XM_tr = np.hstack((X_tr, M_tr.reshape(-1, 1)))
        XM_te = np.hstack((X_te, M_te.reshape(-1, 1)))
        
        pi_xm_model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=seed)
        pi_xm_model.fit(XM_tr, A_tr)
        p_am1_te = np.clip(pi_xm_model.predict_proba(XM_te)[:, 1], 0.05, 0.95)
        p_am0_te = np.clip(pi_xm_model.predict_proba(XM_te)[:, 0], 0.05, 0.95)

        # ---- 计算 theta_10 的中间量 E[Y|A=1, M, X] ----
        mu1_mx_model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=seed)
        mu1_mx_model.fit(XM_tr[A_tr == 1], Y_tr[A_tr == 1])
        Q_Y_1_te = mu1_mx_model.predict(XM_te)
        Q_Y_1_tr = mu1_mx_model.predict(XM_tr)

        q_ym_model_10 = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=seed)
        q_ym_model_10.fit(X_tr[A_tr == 0], Q_Y_1_tr[A_tr == 0])
        Q_YM_te_10 = q_ym_model_10.predict(X_te)

        # ---- 计算 theta_01 的中间量 E[Y|A=0, M, X] ----
        mu0_mx_model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=seed)
        mu0_mx_model.fit(XM_tr[A_tr == 0], Y_tr[A_tr == 0])
        Q_Y_0_te = mu0_mx_model.predict(XM_te)
        Q_Y_0_tr = mu0_mx_model.predict(XM_tr)

        q_ym_model_01 = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=seed)
        q_ym_model_01.fit(X_tr[A_tr == 1], Q_Y_0_tr[A_tr == 1])
        Q_YM_te_01 = q_ym_model_01.predict(X_te)

        # ==========================================
        # 步骤 5: DR 核心公式计算 theta_10 和 theta_01
        # ==========================================
        w_ratio_10 = (p_am0_te / p_am1_te) * (p_a1_te / p_a0_te)
        term1_10 = (a0_ind / p_a0_te) * (Q_Y_1_te - Q_YM_te_10)
        term2_10 = (a1_ind / p_a1_te) * w_ratio_10 * (Y_te - Q_Y_1_te)
        eif_10 = term1_10 + term2_10 + Q_YM_te_10

        w_ratio_01 = (p_am1_te / p_am0_te) * (p_a0_te / p_a1_te)
        term1_01 = (a1_ind / p_a1_te) * (Q_Y_0_te - Q_YM_te_01)
        term2_01 = (a0_ind / p_a0_te) * w_ratio_01 * (Y_te - Q_Y_0_te)
        eif_01 = term1_01 + term2_01 + Q_YM_te_01

        # 保存均值
        results['theta_11'].append(np.mean(eif_11))
        results['theta_00'].append(np.mean(eif_00))
        results['theta_10'].append(np.mean(eif_10))
        results['theta_01'].append(np.mean(eif_01))
        
        # 💥 记录该折样本的 point-wise score 到全局
        score_11_all[test_idx] = eif_11
        score_00_all[test_idx] = eif_00
        score_10_all[test_idx] = eif_10
        score_01_all[test_idx] = eif_01

    # 汇总各折的点估计结果
    final_res = {k: np.mean(v) for k, v in results.items()}
    
    # 计算各因果参数的点估计
    final_res['TE']  = final_res['theta_11'] - final_res['theta_00']
    final_res['NDE'] = final_res['theta_10'] - final_res['theta_00']
    final_res['NIE'] = final_res['theta_11'] - final_res['theta_10'] 
    final_res['PDE'] = final_res['theta_11'] - final_res['theta_01']
    final_res['TIE'] = final_res['theta_01'] - final_res['theta_00']
    
    # ====================================================================
    # 💥 严谨的推断部分: 基于差值 EIF 的方差计算所有因果效应的 SE 和 CI
    # ====================================================================
    z_val = stats.norm.ppf(0.975)
    
    # 1. 总效应 TE = theta_11 - theta_00
    if_te = (score_11_all - final_res['theta_11']) - (score_00_all - final_res['theta_00'])
    final_res['TE_se'] = np.sqrt(np.var(if_te, ddof=1) / n)
    final_res['TE_ci'] = [final_res['TE'] - z_val*final_res['TE_se'], final_res['TE'] + z_val*final_res['TE_se']]
    
    # 2. 自然直接效应 NDE = theta_10 - theta_00
    if_nde = (score_10_all - final_res['theta_10']) - (score_00_all - final_res['theta_00'])
    final_res['NDE_se'] = np.sqrt(np.var(if_nde, ddof=1) / n)
    final_res['NDE_ci'] = [final_res['NDE'] - z_val*final_res['NDE_se'], final_res['NDE'] + z_val*final_res['NDE_se']]
    
    # 3. 自然间接效应 NIE = theta_11 - theta_10
    if_nie = (score_11_all - final_res['theta_11']) - (score_10_all - final_res['theta_10'])
    final_res['NIE_se'] = np.sqrt(np.var(if_nie, ddof=1) / n)
    final_res['NIE_ci'] = [final_res['NIE'] - z_val*final_res['NIE_se'], final_res['NIE'] + z_val*final_res['NIE_se']]
    
    # 4. 纯直接效应 PDE = theta_11 - theta_01
    if_pde = (score_11_all - final_res['theta_11']) - (score_01_all - final_res['theta_01'])
    final_res['PDE_se'] = np.sqrt(np.var(if_pde, ddof=1) / n)
    final_res['PDE_ci'] = [final_res['PDE'] - z_val*final_res['PDE_se'], final_res['PDE'] + z_val*final_res['PDE_se']]
    
    # 5. 总间接效应 TIE = theta_01 - theta_00
    if_tie = (score_01_all - final_res['theta_01']) - (score_00_all - final_res['theta_00'])
    final_res['TIE_se'] = np.sqrt(np.var(if_tie, ddof=1) / n)
    final_res['TIE_ci'] = [final_res['TIE'] - z_val*final_res['TIE_se'], final_res['TIE'] + z_val*final_res['TIE_se']]
    
    return final_res

def main():
    base_dir = r'D:\code\simulation\simulation'
    with open(os.path.join(base_dir, 'processed_data_dict.pkl'), 'rb') as f:
        data_dict = pickle.load(f)

    print("启动全面对齐版传统 DR 中介分析...")
    res = cross_fit_traditional_mediation(data_dict)

    print("\n" + "="*65)
    print("📊 传统中介分析结果报告 (严格 EIF 推断)")
    print("="*65)
    print(f"E[Y(0, M(0))] (普高基准):          {res['theta_00']:.4f} 美元")
    print(f"E[Y(1, M(1))] (预科总期望):          {res['theta_11']:.4f} 美元")
    print(f"E[Y(1, M(0))] (交叉期望 1):          {res['theta_10']:.4f} 美元")
    print(f"E[Y(0, M(1))] (交叉期望 2):          {res['theta_01']:.4f} 美元")
    print("-" * 65)
    print(f"总效应 (Total Effect):             {res['TE']:.4f} 美元")
    print(f"   => 95% CI: [{res['TE_ci'][0]:.4f}, {res['TE_ci'][1]:.4f}], SE: {res['TE_se']:.4f}")
    
    print("\n--- 分解方式 1 (基于 M(0)) ---")
    print(f"自然直接效应 (NDE):                {res['NDE']:.4f} 美元")
    print(f"   => 95% CI: [{res['NDE_ci'][0]:.4f}, {res['NDE_ci'][1]:.4f}], SE: {res['NDE_se']:.4f}")
    print(f"自然间接效应 (NIE):                {res['NIE']:.4f} 美元")
    print(f"   => 95% CI: [{res['NIE_ci'][0]:.4f}, {res['NIE_ci'][1]:.4f}], SE: {res['NIE_se']:.4f}")
    
    print("\n--- 分解方式 2 (基于 M(1)) ---")
    print(f"纯直接效应 (PDE):                  {res['PDE']:.4f} 美元")
    print(f"   => 95% CI: [{res['PDE_ci'][0]:.4f}, {res['PDE_ci'][1]:.4f}], SE: {res['PDE_se']:.4f}")
    print(f"总间接效应 (TIE):                  {res['TIE']:.4f} 美元")
    print(f"   => 95% CI: [{res['TIE_ci'][0]:.4f}, {res['TIE_ci'][1]:.4f}], SE: {res['TIE_se']:.4f}")
    print("="*65)

if __name__ == "__main__":
    main()