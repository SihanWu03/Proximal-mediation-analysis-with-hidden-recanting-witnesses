import numpy as np
import bridge_base
import linear_dgp
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from minimax import kkt_solve, score_nuisance_function
# from bridge_base import KernelBridgeBase
# from parameters import LAMBDA_MIN_FACTOR, LAMBDA_MAX_FACTOR
from sklearn.metrics.pairwise import pairwise_kernels
from econml.grf.classes import RegressionForest
LAMBDA_MIN_FACTOR = 1e-3
LAMBDA_MAX_FACTOR = 1e3
LAMBDA_GRID = 25

GAMMA_MIN = 1e-3 # only for 5b: use 1e-4
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
#         loc_0 = data.a[:, 0] == 0
#         loc_1 = data.a[:, 0] == 1
#         n0 = np.sum(loc_0)
#         n1 = np.sum(loc_1)
#         g1 = np.ones(n1) * len(data) / n1
#         g2 = -1.0 * np.ones(n0) * len(data) / n0
#         zx = np.hstack((data.z, data.x))
#         wx = np.hstack((data.w, data.x))
#         return g1, g2, wx, zx, loc_0, loc_1

        probs = self.treatment_prob.predict_proba(data['x'])
        g1 = (data['a'][:, 0] == 1).astype(float) / probs[:, 1]
        g2 = -(data['a'][:, 0] == 0).astype(float) / probs[:, 0]
        wx = np.hstack((data['w'], data['x']))
        zx = np.hstack((data['z'], data['x']))
        return g1, g2, wx, zx

    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
#         g1, g2, wx, zx, loc_0, loc_1 = self.extract_data(fit_data)

#         kq1 = self.kernel1(zx[loc_1], zx)
#         kf0 = self.kernel2(wx[loc_0], wx)
#         kf1 = self.kernel2(wx[loc_1], wx)

#         kq = self.kernel1(zx, zx)
#         kf = self.kernel2(wx, wx)

#         alpha, beta = kkt_solve(
#             kq1, kf0, kf1, kf, kq, kf, g1, g2, self.lambda1, self.lambda2
#         )

#         self.alpha = alpha
#         self.x = zx.copy()
#         self.beta = beta
#         self.xf = wx.copy()

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
#         g1, g2, wx, zx, loc_0, loc_1 = self.extract_data(val_data)

#         q1 = self(zx[loc_1])
#         kf0 = self.kernel2(wx[loc_0], wx)
#         kf1 = self.kernel2(wx[loc_1], wx)
#         kf = self.kernel2(wx, wx)
#         try:
#             return score_nuisance_function(
#                 q1,
#                 kf0,
#                 kf1,
#                 kf,
#                 kf,
#                 g1,
#                 g2,
# #                 self.lambda2,
#                 LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
#             )
#         except np.linalg.LinAlgError:
#             return np.inf

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
#                 self.lambda2,
#                 np.sqrt(LAMBDA_MIN_FACTOR * LAMBDA_MAX_FACTOR) * kf.shape[0] ** 0.2,
                 LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
            )
        except np.linalg.LinAlgError:
            return np.inf
        
class treatment_prob:
    def __init__(self, calibrated=True):
        self.calibrated = calibrated
    
    def fit(self, X, A):
        # 基础逻辑斯蒂回归模型
        base_model = LogisticRegression(
            penalty='l2',  # L2正则化防止过拟合
            C=1.0,         # 正则化强度
            solver='lbfgs',
            max_iter=1000,
            random_state=1234
        )
        
        if self.calibrated:
            # 使用概率校准（虽然理论不需要，但实际中可能有益）
            self.model = CalibratedClassifierCV(
                base_model,
                method='sigmoid',  # 使用sigmoid校准
                cv=5
            )
        else:
            self.model = base_model
        
        self.model.fit(X, A.flatten() if A.ndim > 1 else A)
        
    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        # 【关键修复】：强制裁剪概率，防止除以极小值暴雷！
        proba = np.clip(proba, 0.05, 0.95)
        return proba
# class treatment_prob:
#     def fit(self, X, A):
#         # 使用逻辑斯蒂回归
#         mu = LogisticRegression(
#             max_iter=1000,
#             random_state=1234
#         )
#         mu.fit(X=X, y=A)
#         self.mu = mu
        
    # def predict_proba(self, X):
    #     # 获取预测概率，形状为(n,2)
    #     point_mu = self.mu.predict_proba(X)
    #     return point_mu
    
class KernelBridgeQ0(bridge_base.KernelBridgeBase):
    """Estimator for the bridge function Q0(W,M2,X)"""
    
    def __init__(self, lambda1, lambda2, gamma1, gamma2, q1_estimator=None, treatment_prob=None):
        super().__init__(lambda1, lambda2, gamma1, gamma2, treatment_prob)
        self.q1_estimator = q1_estimator  # 保存 q1 估计器
    
    def extract_data(self, data):
        """Extract the data for fitting the bridge function"""
        # 计算倾向得分
        probs = self.treatment_prob.predict_proba(data['x'])
        
        # 计算 q1 的预测值（使用之前训练好的 q1 估计器）
        if self.q1_estimator is not None:
            # 构建 q1 的特征：Z, X
            zx_q1 = np.hstack((data['z'], data['x']))
            # 使用 q1 估计器预测所有样本
            q1_predictions = self.q1_estimator(zx_q1)
        else:
            raise ValueError("q1_estimator is required for KernelBridgeQ0")
        
        # 计算 g1 和 g2
        # g1 = I(A=0) / P(A=0|X)
        g1 = (data['a'][:, 0] == 0).astype(float) 
        
        # g2 = -I(A=1) * q1(Z,X) / P(A=0|X)
        g2 = -(data['a'][:, 0] == 1).astype(float) * q1_predictions 
        
        # 特征组合：Q0 使用 W, M2, X 和 Z, M2, X
        wx = np.hstack((data['w'], data['m2'], data['x']))
        zx = np.hstack((data['z'], data['m2'], data['x']))
        
        return g1, g2, wx, zx
    def fit(self, fit_data):
        """Fit the bridge function using minimax optimization"""
#         g1, g2, wx, zx, loc_0, loc_1 = self.extract_data(fit_data)

#         kq1 = self.kernel1(zx[loc_1], zx)
#         kf0 = self.kernel2(wx[loc_0], wx)
#         kf1 = self.kernel2(wx[loc_1], wx)

#         kq = self.kernel1(zx, zx)
#         kf = self.kernel2(wx, wx)

#         alpha, beta = kkt_solve(
#             kq1, kf0, kf1, kf, kq, kf, g1, g2, self.lambda1, self.lambda2
#         )

#         self.alpha = alpha
#         self.x = zx.copy()
#         self.beta = beta
#         self.xf = wx.copy()

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
#         g1, g2, wx, zx, loc_0, loc_1 = self.extract_data(val_data)

#         q1 = self(zx[loc_1])
#         kf0 = self.kernel2(wx[loc_0], wx)
#         kf1 = self.kernel2(wx[loc_1], wx)
#         kf = self.kernel2(wx, wx)
#         try:
#             return score_nuisance_function(
#                 q1,
#                 kf0,
#                 kf1,
#                 kf,
#                 kf,
#                 g1,
#                 g2,
# #                 self.lambda2,
#                 LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
#             )
#         except np.linalg.LinAlgError:
#             return np.inf

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
#                 self.lambda2,
#                 np.sqrt(LAMBDA_MIN_FACTOR * LAMBDA_MAX_FACTOR) * kf.shape[0] ** 0.2,
                 LAMBDA_MIN_FACTOR * kf.shape[0] ** 0.2,
            )
        except np.linalg.LinAlgError:
            return np.inf

def test_q1_and_q0():
    """测试 KernelBridgeQ1 和 KernelBridgeQ0 能否正常运行"""
    print("=" * 60)
    print("测试 Q1 和 Q0 桥函数是否能正常运行")
    print("=" * 60)
    
    # 1. 创建数据生成器
    print("\n生成数据...")
    datagen = linear_dgp.LinearTwoMediatorsProxDGP(
        xdim=5,     # 协变量维度
        zdim=2,     # Z的维度
        wdim=2,     # W的维度
        m1dim=2,    # M1的维度
        m2dim=2,    # M2的维度
        seed=42
    )
    
    # 生成数据
    data = datagen.sample_dataset(n=300, seed=42)
    
    # 输出数据形状
    print("数据形状:")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
    
    # 2. 训练倾向得分模型
    print("\n" + "-" * 40)
    print("训练倾向得分模型...")
    print("-" * 40)
    
    try:
        propensity_model = treatment_prob()
        propensity_model.fit(X=data['x'], A=data['a'][:, 0])
        print("✓ 倾向得分模型训练成功")
        
        # 检查倾向得分
        probs = propensity_model.predict_proba(data['x'])
        print(f"倾向得分形状: {probs.shape}")
        print(f"治疗组平均概率: {np.mean(probs[:, 1]):.4f}")
        print(f"对照组平均概率: {np.mean(probs[:, 0]):.4f}")
        
    except Exception as e:
        print(f"✗ 倾向得分模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 测试 KernelBridgeQ1
    print("\n" + "-" * 40)
    print("测试 KernelBridgeQ1...")
    print("-" * 40)
    
    try:
        # 创建 Q1 估计器
        q1_estimator = KernelBridgeQ1(
            lambda1=0.1,
            lambda2=0.1,
            gamma1=0.5,
            gamma2=0.5,
        )
        
        # 设置倾向得分模型
        q1_estimator.treatment_prob = propensity_model
        
        # 训练 Q1 函数
        q1_estimator.fit(data)
        print("✓ KernelBridgeQ1 训练成功")
        
        # 测试预测
        zx_q1 = np.hstack((data['z'], data['x']))
        q1_predictions = q1_estimator(zx_q1)
        print(f"  Q1 预测形状: {q1_predictions.shape}")
        print(f"  Q1 预测值范围: [{q1_predictions.min():.3f}, {q1_predictions.max():.3f}]")
        
        # 测试评分
        q1_score = q1_estimator.score(data)
        print(f"  Q1 评分: {q1_score:.4f}")
        
        # 检查模型参数
        print(f"  Q1 alpha 形状: {q1_estimator.alpha.shape}")
        print(f"  Q1 beta 形状: {q1_estimator.beta.shape}")
        
    except Exception as e:
        print(f"✗ KernelBridgeQ1 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 测试 KernelBridgeQ0
    print("\n" + "-" * 40)
    print("测试 KernelBridgeQ0...")
    print("-" * 40)
    
    try:
        # 创建 Q0 估计器，传入训练好的 Q1 估计器
        q0_estimator = KernelBridgeQ0(
            lambda1=0.1,
            lambda2=0.1,
            gamma1=0.5,
            gamma2=0.5,
            q1_estimator=q1_estimator
        )
        
        # 设置倾向得分模型
        q0_estimator.treatment_prob = propensity_model
        
        # 测试数据提取
        print("\n测试 Q0 数据提取...")
        g1, g2, wx, zx = q0_estimator.extract_data(data)
        print(f"  g1 形状: {g1.shape}")
        print(f"  g2 形状: {g2.shape}")
        print(f"  wx 形状: {wx.shape}")
        print(f"  zx 形状: {zx.shape}")
        
        # 检查 g1 和 g2 的计算
        a_values = data['a'][:, 0]
        loc_a0 = a_values == 0
        loc_a1 = a_values == 1
        
        # 手动验证 g1
        g1_manual = (loc_a0.astype(float) / probs[:, 0])
        g1_match = np.allclose(g1, g1_manual)
        print(f"  g1 计算正确: {g1_match}")
        if not g1_match:
            print(f"    最大差异: {np.max(np.abs(g1 - g1_manual)):.6f}")
        
        # 手动验证 g2
        q1_predictions_flat = q1_predictions.flatten() if q1_predictions.ndim > 1 else q1_predictions
        g2_manual = -(loc_a1.astype(float) * q1_predictions_flat / probs[:, 0])
        g2_match = np.allclose(g2, g2_manual)
        print(f"  g2 计算正确: {g2_match}")
        if not g2_match:
            print(f"    最大差异: {np.max(np.abs(g2 - g2_manual)):.6f}")
        
        # 训练 Q0 函数
        q0_estimator.fit(data)
        print("✓ KernelBridgeQ0 训练成功")
        
        # 测试预测
        q0_predictions = q0_estimator(zx)
        print(f"  Q0 预测形状: {q0_predictions.shape}")
        print(f"  Q0 预测值范围: [{q0_predictions.min():.3f}, {q0_predictions.max():.3f}]")
        
        # 测试评分
        q0_score = q0_estimator.score(data)
        print(f"  Q0 评分: {q0_score:.4f}")
        
        # 检查模型参数
        print(f"  Q0 alpha 形状: {q0_estimator.alpha.shape}")
        print(f"  Q0 beta 形状: {q0_estimator.beta.shape}")
        
        # 检查特征维度
        print(f"\n检查特征维度:")
        print(f"  Q1 的 zx 维度: {zx_q1.shape[1]}")
        print(f"  Q0 的 wx 维度: {wx.shape[1]}")
        print(f"  Q0 的 zx 维度: {zx.shape[1]}")
        print(f"  是否包含 M2: {'m2' in data.keys()}")
        
    except Exception as e:
        print(f"✗ KernelBridgeQ0 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    

    
    print("\n" + "=" * 60)
    print("测试完成！Q1 和 Q0 桥函数都能正常运行。")
    print("=" * 60)


if __name__ == "__main__":
    test_q1_and_q0()
    

    

