import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
import linear_dgp
from bridgeh import KernelBridgeH0,KernelBridgeH1
from bridgeq import KernelBridgeQ0,KernelBridgeQ1,treatment_prob
import sys

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

class ProximalEstimatorBase:

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

        # set any provided parameters
        self.params = {}
        for key in ["treatment", "h", "q", "eta"]:
            if key in kwargs:
                self.params[key] = kwargs[key]

        # pass in dgp directly for debug
        self.dgp = kwargs.get("dgp", None)

    def fit(self, fit_data, val_data):
        """Fit the estimator"""
        raise NotImplementedError

    def evaluate(self, eval_data):
        """Evaluate the estimator after fitting pointwise on the evaluation data"""
        raise NotImplementedError

    def optimize_h0(self, fit_data, val_data):
        """优化 h0 的超参数"""
        print("=" * 60)
        print("开始优化 h0 的超参数...")
        print("=" * 60)
        
        # 检查是否有预定义的参数
        if all(param in self.params.get("h0", {}) for param in ['lambda1', 'lambda2', 'gamma']):
            lambda1 = self.params["h0"]['lambda1']
            lambda2 = self.params["h0"]['lambda2']
            gamma = self.params["h0"]['gamma']
            print(f"使用预定义参数: lambda1={lambda1}, lambda2={lambda2}, gamma={gamma}")
            best_score = None
            best_params = self.params["h0"]
        else:
            # 使用 Optuna 优化超参数
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
                
                # 创建并训练 h0 估计器
                est = KernelBridgeH0(lambda1, lambda2, gamma, gamma)
                est.fit(fit_data)
                
                # 返回验证集评分（越低越好）
                return est.score(val_data)
            
            # 运行优化
            study = optuna.create_study(direction="minimize")
            study.optimize(_optimize_h0, n_trials=self.num_runs)
            
            # 获取最佳参数
            best_params = study.best_params
            best_score = study.best_value
            lambda1 = best_params['lambda1']
            lambda2 = best_params['lambda2']
            gamma = best_params['gamma']
            
            if self.verbose:
                print(f"h0 最佳参数: {best_params}, 评分: {best_score:.4f}")
        
        # 使用最佳参数训练最终的 h0 估计器
        h0_best = KernelBridgeH0(lambda1, lambda2, gamma, gamma)
        h0_best.fit(fit_data)
        
        # 保存参数
        self.params["h0"] = best_params
        self.h0_estimator = h0_best
        best_score = h0_best.score(val_data)
        
        return h0_best, best_params, best_score
    
    def optimize_h1(self, fit_data, val_data, h0_estimator):
        """优化 h1 的超参数（需要已训练的 h0_estimator）"""
        print("\n" + "=" * 60)
        print("开始优化 h1 的超参数...")
        print("=" * 60)
        
        # 检查是否有预定义的参数
        if all(param in self.params.get("h1", {}) for param in ['lambda1', 'lambda2', 'gamma']):
            lambda1 = self.params["h1"]['lambda1']
            lambda2 = self.params["h1"]['lambda2']
            gamma = self.params["h1"]['gamma']
            print(f"使用预定义参数: lambda1={lambda1}, lambda2={lambda2}, gamma={gamma}")
            best_score = None
            best_params = self.params["h1"]
        else:
            # 使用 Optuna 优化超参数
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
                
                # 创建 h1 估计器（传入 h0_estimator）
                est = KernelBridgeH1(
                    lambda1, lambda2, gamma, gamma,
                    h0_estimator=h0_estimator
                )
                est.fit(fit_data)
                
                # 返回验证集评分
                return est.score(val_data)
            
            # 运行优化
            study = optuna.create_study(direction="minimize")
            study.optimize(_optimize_h1, n_trials=self.num_runs)
            
            # 获取最佳参数
            best_params = study.best_params
            best_score = study.best_value
            lambda1 = best_params['lambda1']
            lambda2 = best_params['lambda2']
            gamma = best_params['gamma']
            
            if self.verbose:
                print(f"h1 最佳参数: {best_params}, 评分: {best_score:.4f}")
        
        # 使用最佳参数训练最终的 h1 估计器
        h1_best = KernelBridgeH1(
            lambda1, lambda2, gamma, gamma,
            h0_estimator=h0_estimator
        )
        h1_best.fit(fit_data)
        
        # 保存参数
        self.params["h1"] = best_params
        self.h1_estimator = h1_best
        best_score = h1_best.score(val_data)
        
        return h1_best, best_params, best_score
    
    def optimize_both(self, fit_data, val_data):
        """依次优化 h0 和 h1"""
        # 1. 优化 h0
        h0_best, h0_params, h0_score = self.optimize_h0(fit_data, val_data)
        
        # 2. 优化 h1（使用优化好的 h0）
        h1_best, h1_params, h1_score = self.optimize_h1(fit_data, val_data, h0_best)
        
        # 3. 汇总结果
        results = {
            'h0': {
                'estimator': h0_best,
                'params': h0_params,
                'score': h0_score
            },
            'h1': {
                'estimator': h1_best,
                'params': h1_params,
                'score': h1_score
            }
        }
        
        return results
    

    def optimize_p(self, fit_data, val_data):
        """训练倾向得分模型 - 合并训练集和验证集"""
        # 合并所有数据
        combined_x = np.vstack([fit_data['x'], val_data['x']])
        combined_a = np.hstack([fit_data['a'][:, 0], val_data['a'][:, 0]])
        
        # 训练模型
        self.treatment_prob_model = treatment_prob()
        self.treatment_prob_model.fit(X=combined_x, A=combined_a)
        
        return self.treatment_prob_model, {}

    def optimize_q1(self, fit_data, val_data):
        """优化 q1 的超参数"""
        print("\n" + "=" * 60)
        print("优化 q1 的超参数...")
        print("=" * 60)
        
        # 确保倾向得分模型已经训练
        if not hasattr(self, 'treatment_prob_model') or self.treatment_prob_model is None:
          raise ValueError("未找到倾向得分模型，请先调用 optimize_p() 或 optimize_allq() 训练倾向得分模型")
        
        # 检查是否有预定义的参数
        if all(param in self.params.get("q1", {}) for param in ['lambda1', 'lambda2', 'gamma']):
            lambda1 = self.params["q1"]['lambda1']
            lambda2 = self.params["q1"]['lambda2']
            gamma = self.params["q1"]['gamma']
            print(f"使用预定义参数: lambda1={lambda1}, lambda2={lambda2}, gamma={gamma}")
            best_score = None
            best_params = self.params["q1"]
        else:
            # 使用 Optuna 优化超参数
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
                
                # 创建并训练 q1 估计器
                est = KernelBridgeQ1(lambda1, lambda2, gamma, gamma)
                est.treatment_prob = self.treatment_prob_model
                est.fit(fit_data)
                
                # 返回验证集评分
                return est.score(val_data)
            
            # 运行优化
            study = optuna.create_study(direction="minimize")
            study.optimize(_optimize_q1, n_trials=self.num_runs)
            
            # 获取最佳参数
            best_params = study.best_params
            best_score = study.best_value
            lambda1 = best_params['lambda1']
            lambda2 = best_params['lambda2']
            gamma = best_params['gamma']
            
            if self.verbose:
                print(f"q1 最佳参数: {best_params}, 评分: {best_score:.4f}")
        
        # 使用最佳参数训练最终的 q1 估计器
        q1_best = KernelBridgeQ1(lambda1, lambda2, gamma, gamma)
        q1_best.treatment_prob = self.treatment_prob_model
        q1_best.fit(fit_data)
        
        # 保存参数
        self.params["q1"] = best_params
        self.q1_estimator = q1_best
        best_score = q1_best.score(val_data)
        
        return q1_best, best_params, best_score
    

    def optimize_q0(self, fit_data, val_data, q1_estimator):
        """优化 q0 的超参数（需要已训练的 q1_estimator）"""
        print("\n" + "=" * 60)
        print("优化 q0 的超参数...")
        print("=" * 60)
        
        # 确保倾向得分模型已经训练
        if not hasattr(self, 'treatment_prob_model') or self.treatment_prob_model is None:
          raise ValueError("未找到倾向得分模型，请先调用 optimize_p() 或 optimize_allq() 训练倾向得分模型")
        
        # 检查是否有预定义的参数
        if all(param in self.params.get("q0", {}) for param in ['lambda1', 'lambda2', 'gamma']):
            lambda1 = self.params["q0"]['lambda1']
            lambda2 = self.params["q0"]['lambda2']
            gamma = self.params["q0"]['gamma']
            print(f"使用预定义参数: lambda1={lambda1}, lambda2={lambda2}, gamma={gamma}")
            best_score = None
            best_params = self.params["q0"]
        else:
            # 使用 Optuna 优化超参数
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
                
                # 创建 q0 估计器（传入 q1_estimator）
                est = KernelBridgeQ0(
                    lambda1, lambda2, gamma, gamma,
                    q1_estimator=q1_estimator
                )
                est.treatment_prob = self.treatment_prob_model
                est.fit(fit_data)
                
                # 返回验证集评分
                return est.score(val_data)
            
            # 运行优化
            study = optuna.create_study(direction="minimize")
            study.optimize(_optimize_q0, n_trials=self.num_runs)
            
            # 获取最佳参数
            best_params = study.best_params
            best_score = study.best_value
            lambda1 = best_params['lambda1']
            lambda2 = best_params['lambda2']
            gamma = best_params['gamma']
            
            if self.verbose:
                print(f"q0 最佳参数: {best_params}, 评分: {best_score:.4f}")
        
        # 使用最佳参数训练最终的 q0 估计器
        q0_best = KernelBridgeQ0(
            lambda1, lambda2, gamma, gamma,
            q1_estimator=q1_estimator
        )
        q0_best.treatment_prob = self.treatment_prob_model
        q0_best.fit(fit_data)
        
        # 保存参数
        self.params["q0"] = best_params
        self.q0_estimator = q0_best
        best_score = q0_best.score(val_data)
        
        return q0_best, best_params, best_score

    def optimize_allq(self, fit_data, val_data):
        """依次训练/优化所有 Q 相关的模型"""
        print("=" * 60)
        print("训练所有 Q 相关模型...")
        print("=" * 60)
        
        # 1. 训练倾向得分模型
        treatment_model, _ = self.optimize_p(fit_data, val_data)
        
        # 2. 优化 q1
        q1_best, q1_params, q1_score = self.optimize_q1(fit_data, val_data)
        
        # 3. 优化 q0
        q0_best, q0_params, q0_score = self.optimize_q0(fit_data, val_data, q1_best)
        
        # 4. 汇总结果
        results = {
            'treatment': {
                'estimator': treatment_model
            },
            'q1': {
                'estimator': q1_best,
                'params': q1_params,
                'score': q1_score
            },
            'q0': {
                'estimator': q0_best,
                'params': q0_params,
                'score': q0_score
            }
        }
        
        print("\n" + "=" * 60)
        print("所有 Q 相关模型完成!")
        print("=" * 60)
        print(f"q1 评分: {q1_score:.4f}")
        print(f"q0 评分: {q0_score:.4f}")
        
        return results





    def fit_eta(self, h1_fn, fit_data, val_data):
        """使用岭核回归估计 h1_fn(w,x) 在 A=0 时的值，并进行超参数优化"""
        
        def extract_data(data):
            """从数据中提取特征和目标"""
            loc = data['a'][:, 0] == 0  # 只使用 A=0 的样本
            x0 = data['x'][loc]  # 特征：X
            # 目标：h1_fn(w,x) 在 A=0 样本上的预测值
            targets = h1_fn(np.hstack((data['w'][loc], data['x'][loc])))
            return x0, targets
        
        def build_reg(alpha=1, gamma=None):
            """构建岭核回归管道"""
            return Pipeline([
                ("scaler", StandardScaler()),  # 标准化特征
                (
                    "kernel_ridge",
                    KernelRidge(
                        kernel=self.kernel_metric, 
                        alpha=alpha, 
                        gamma=gamma
                    ),
                ),
            ])
        
        # 提取训练集和验证集数据
        x_fit, y_fit = extract_data(fit_data)
        x_val, y_val = extract_data(val_data)
        
        print("=" * 60)
        print("开始优化岭核回归的超参数...")
        print("=" * 60)
        
        # 如果有预定义的参数，直接使用
        if all(param in self.params.get("eta", {}) for param in ['alpha', 'gamma']):
            alpha = self.params["eta"]['alpha']
            gamma = self.params["eta"]['gamma']
            print(f"使用预定义参数: alpha={alpha}, gamma={gamma}")
            best_params = self.params["eta"]
            best_score = None
        else:
            # 使用 Optuna 优化超参数
            def _optimize_eta(trial):
                # 定义超参数搜索范围
                alpha = trial.suggest_loguniform('alpha', ALPHA_MIN, ALPHA_MAX)
                gamma = trial.suggest_loguniform('gamma', REG_GAMMA_MIN, REG_GAMMA_MAX)
                
                # 创建模型
                model = build_reg(alpha=alpha, gamma=gamma)
                
                # 训练模型
                model.fit(x_fit, y_fit)
                
                # 在验证集上评分（使用负的均方误差，因为 optuna 默认最小化）
                y_pred = model.predict(x_val)
                mse = np.mean((y_val - y_pred) ** 2)
                return mse
            
            # 创建并运行 Optuna 研究
            study = optuna.create_study(
                direction="minimize",
                study_name="kernel_ridge_optimization"
            )
            
            # 设置优化的试验次数
            n_trials = self.num_runs # 最多100次试验
            study.optimize(_optimize_eta, n_trials=n_trials)
            
            # 获取最佳参数
            best_params = study.best_params
            best_score = study.best_value
            
            alpha = best_params['alpha']
            gamma = best_params['gamma']
            
            if self.verbose:
                print(f"岭核回归最佳参数: {best_params}")
                print(f"最佳均方误差: {best_score:.6f}")
                
                # # 可选：打印超参数重要性
                # if len(study.trials) > 5:
                #     try:
                #         importance = optuna.importance.get_param_importances(study)
                #         print("超参数重要性:")
                #         for param, imp in importance.items():
                #             print(f"  {param}: {imp:.3f}")
                #     except:
                #         pass
        
        # 使用最佳参数训练最终的模型
        # print("\n" + "=" * 60)
        # print("使用最佳参数训练最终的岭核回归模型...")
        # print("=" * 60)
        
        cond_mean = build_reg(alpha=alpha, gamma=gamma)
        cond_mean.fit(x_fit, y_fit)
        
        # 保存参数和模型
        self.params["eta"] = best_params
        self.eta_model = cond_mean
        
        # # 评估最终模型
        # train_pred = cond_mean.predict(x_fit)
        # val_pred = cond_mean.predict(x_val)
        
        # train_mse = np.mean((y_fit - train_pred) ** 2)
        # val_mse = np.mean((y_val - val_pred) ** 2)
        
        # print(f"训练集 MSE: {train_mse:.6f}")
        # print(f"验证集 MSE: {val_mse:.6f}")
        # print(f"模型系数形状: {cond_mean.named_steps['kernel_ridge'].dual_coef_.shape}")
        
        return cond_mean
   

    
def main():
    # Define the number of samples and type of setup
    n = 500
    # Create dataset using the given setup
    dataset = linear_dgp.LinearHidMedDGP(xdim=5, zdim=1, wdim=1, mdim=1, udim=1)
    data = dataset.sample_dataset(n=n, seed=2)
    data_test = dataset.sample_dataset(n=100, seed=1234)
    
    # Initialize the estimator
    estimator = ProximalEstimatorBase(setup=type, num_runs=100, num_jobs=1, verbose=True)
    
    # You may need to modify this part depending on your specific estimator's requirements
    # For example, tuning hyperparameters using `fit_bridge` method
    bridge, params, scores = estimator.fit_bridge(data, data_test, which="h")  # Adjust according to your needs
    
    print("Best hyperparameters found: ", params)
    print("Best score: ", scores['score'])

# if __name__ == "__main__":
#     main()