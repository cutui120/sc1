import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
import optuna
import warnings
import matplotlib.pyplot as plt
from scipy import stats
import shap
import os


warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class AdvancedErgothioneineOptimizer:
    """高级麦角硫因合成基因表达优化器 - 改进版"""
    
    # 麦角硫因合成关键基因分组
    PATHWAY_GENES = {
        'histidine_biosynthesis': [0, 1, 2, 3],  # 组氨酸合成途径
        'histamine_production': [4, 5],         # 组氨酸到组胺
        'ergothioneine_core': [6, 7, 8, 9, 10], # EgtB, EgtC, EgtD, EgtE等核心合成酶
        'sulfur_donors': [11, 12],              # 硫供体相关基因
        'cofactor_regeneration': [13, 14, 15],  # 辅因子再生（NADPH, ATP等）
        'central_metabolism': [16, 17, 18, 19], # 中心碳代谢
        'transporters': [20, 21],               # 转运蛋白
        'stress_response': [22, 23]             # 应激响应
    }
    
    def __init__(self, n_genes=24, random_state=42):
        """初始化优化器"""
        self.n_genes = n_genes
        self.random_state = random_state
        np.random.seed(random_state)
        self.scaler = RobustScaler()
        self.engineered_scaler = StandardScaler()  # 新增：工程特征缩放器
        self.best_combination = None
        self.best_predicted_yield = None
        self.original_n_features = n_genes
        self.enhanced_feature_indices = None
        self.engineered_feature_names = []
        
        # 麦角硫因合成关键比例约束
        self.key_ratios = {
            ('ergothioneine_core', 'histidine_biosynthesis'): (0.8, 1.5),
            ('cofactor_regeneration', 'ergothioneine_core'): (1.0, 2.0),
            ('sulfur_donors', 'ergothioneine_core'): (0.7, 1.3)
        }
        
        # SHAP解释器初始化
        self.explainer = None
        self.shap_values = None
        self.gene_importance = None
        self.optimization_history = []
    
    def load_data(self, filepath):
        """加载真实麦角硫因合成数据"""
        data = pd.read_csv(filepath)
        X = data.iloc[:, :self.n_genes].values
        y = data.iloc[:, self.n_genes].values
        print(f"加载数据: {X.shape[0]}个样本, {X.shape[1]}个基因")
        
        # 数据质量检查
        self._check_data_quality(X, y)
        
        return X, y
    
    def _check_data_quality(self, X, y):
        """检查数据质量"""
        print("\n=== 数据质量检查 ===")
        
        # 检查缺失值
        missing_X = np.isnan(X).sum()
        missing_y = np.isnan(y).sum()
        print(f"特征缺失值: {missing_X}, 目标变量缺失值: {missing_y}")
        
        # 检查离群值（使用稳健方法）
        try:
            outlier_detector = EllipticEnvelope(contamination=0.05, random_state=self.random_state)
            is_inlier = outlier_detector.fit_predict(np.column_stack([X, y.reshape(-1, 1)]))
            outlier_percentage = (is_inlier == -1).sum() / len(y) * 100
            print(f"离群值比例: {outlier_percentage:.2f}%")
        except Exception:
            print("离群值检测跳过 (协方差矩阵问题)")
        
        # 检查目标变量分布
        y_skewness = stats.skew(y)
        y_kurtosis = stats.kurtosis(y)
        print(f"产量分布 - 偏度: {y_skewness:.3f}, 峰度: {y_kurtosis:.3f}")
        
        # 检查特征相关性
        gene_correlations = np.corrcoef(X.T)
        high_corr_pairs = np.where(np.abs(gene_correlations) > 0.8)
        unique_pairs = set()
        for i, j in zip(*high_corr_pairs):
            if i < j:
                unique_pairs.add((i, j))
        
        if len(unique_pairs) > 0:
            print(f"高相关基因对 ({len(unique_pairs)}对, |r| > 0.8)")
            for i, j in list(unique_pairs)[:5]:
                print(f"  基因 {i+1} & 基因 {j+1}: r = {gene_correlations[i, j]:.3f}")
            if len(unique_pairs) > 5:
                print(f"  还有 {len(unique_pairs) - 5} 对未显示")
        else:
            print("无高度相关的基因对")
    
    def create_metabolic_pathway_features(self, X, fit_scaler=False):
        """
        基于麦角硫因合成知识创建高级特征
        改进：添加更多生物学相关特征，并对工程特征进行缩放
        """
        new_features = []
        feature_names = []
        
        # 1. 关键酶表达比例特征 (核心酶与辅因子供应的平衡)
        core_genes = self.PATHWAY_GENES['ergothioneine_core']
        cofactor_genes = self.PATHWAY_GENES['cofactor_regeneration']
        his_genes = self.PATHWAY_GENES['histidine_biosynthesis']
        sulfur_genes = self.PATHWAY_GENES['sulfur_donors']
        transporter_genes = self.PATHWAY_GENES['transporters']
        stress_genes = self.PATHWAY_GENES['stress_response']
        
        # 安全获取基因表达均值
        def safe_mean(X, indices):
            valid_indices = [i for i in indices if i < X.shape[1]]
            if valid_indices:
                return X[:, valid_indices].mean(axis=1)
            return np.zeros(X.shape[0])
        
        core_expr = safe_mean(X, core_genes)
        cofactor_expr = safe_mean(X, cofactor_genes)
        his_expr = safe_mean(X, his_genes)
        sulfur_expr = safe_mean(X, sulfur_genes)
        transporter_expr = safe_mean(X, transporter_genes)
        stress_expr = safe_mean(X, stress_genes)
        total_expr = X.mean(axis=1)
        
        # 1. 核心酶与辅因子比例
        ratio_feature = core_expr / (cofactor_expr + 1e-6)
        new_features.append(ratio_feature.reshape(-1, 1))
        feature_names.append('core_to_cofactor_ratio')
        
        # 2. 限速步骤识别特征
        bottleneck_feature = core_expr - his_expr
        new_features.append(bottleneck_feature.reshape(-1, 1))
        feature_names.append('core_minus_his_bottleneck')
        
        # 3. 代谢负担特征
        all_pathway_indices = list(set().union(*self.PATHWAY_GENES.values()))
        valid_pathway_indices = [i for i in all_pathway_indices if i < X.shape[1]]
        if valid_pathway_indices:
            pathway_expr = X[:, valid_pathway_indices].mean(axis=1)
        else:
            pathway_expr = total_expr
        burden_feature = pathway_expr / (total_expr + 1e-6)
        new_features.append(burden_feature.reshape(-1, 1))
        feature_names.append('pathway_burden_ratio')
        
        # 4. 合成途径平衡指数
        his_core_ratio = his_expr / (core_expr + 1e-6)
        cofactor_core_ratio = cofactor_expr / (core_expr + 1e-6)
        balance_index = np.sqrt(his_core_ratio**2 + cofactor_core_ratio**2)
        new_features.append(balance_index.reshape(-1, 1))
        feature_names.append('pathway_balance_index')
        
        # 5. 交互项：关键酶与辅因子的乘积
        interaction = core_expr * cofactor_expr
        new_features.append(interaction.reshape(-1, 1))
        feature_names.append('core_cofactor_interaction')
        
        # 6. 新增：硫供体与核心酶比例
        sulfur_core_ratio = sulfur_expr / (core_expr + 1e-6)
        new_features.append(sulfur_core_ratio.reshape(-1, 1))
        feature_names.append('sulfur_core_ratio')
        
        # 7. 新增：转运蛋白与核心酶比例（产物输出能力）
        transporter_core_ratio = transporter_expr / (core_expr + 1e-6)
        new_features.append(transporter_core_ratio.reshape(-1, 1))
        feature_names.append('transporter_core_ratio')
        
        # 8. 新增：应激响应与总表达比例
        stress_total_ratio = stress_expr / (total_expr + 1e-6)
        new_features.append(stress_total_ratio.reshape(-1, 1))
        feature_names.append('stress_total_ratio')
        
        # 9. 新增：核心酶表达方差（酶平衡度）
        core_valid = [i for i in core_genes if i < X.shape[1]]
        if len(core_valid) > 1:
            core_variance = X[:, core_valid].std(axis=1)
            new_features.append(core_variance.reshape(-1, 1))
            feature_names.append('core_enzyme_variance')
        
        # 10. 新增：组氨酸与辅因子交互
        his_cofactor_interaction = his_expr * cofactor_expr
        new_features.append(his_cofactor_interaction.reshape(-1, 1))
        feature_names.append('his_cofactor_interaction')
        
        # 组合所有新特征
        if new_features:
            engineered_features = np.hstack(new_features)
            
            # 对工程特征进行缩放
            if fit_scaler:
                engineered_features_scaled = self.engineered_scaler.fit_transform(engineered_features)
            else:
                engineered_features_scaled = self.engineered_scaler.transform(engineered_features)
            
            X_enhanced = np.hstack([X, engineered_features_scaled])
            
            if fit_scaler:
                print(f"特征工程完成：新增 {len(new_features)} 个特征: {feature_names}")
                print(f"特征维度: {X.shape[1]} -> {X_enhanced.shape[1]}")
            
            # 保存新特征的索引信息
            self.enhanced_feature_indices = {
                'original': list(range(self.original_n_features)),
                'engineered': list(range(self.original_n_features, X_enhanced.shape[1]))
            }
            self.engineered_feature_names = feature_names
            
            return X_enhanced
        else:
            return X

    
    def preprocess_data(self, X, y, test_size=0.2):
        """数据预处理，保留生物相关性"""
        # 先创建高级特征（fit_scaler=True表示训练模式）
        print("\n=== 创建高级代谢通路特征 ===")
        X_enhanced = self.create_metabolic_pathway_features(X, fit_scaler=True)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_enhanced, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\n训练集: {X_train.shape[0]}样本, 测试集: {X_test.shape[0]}样本")
        print(f"总特征数: {X_train.shape[1]}")
        
        # 对原始特征使用RobustScaler减少异常值影响
        if self.enhanced_feature_indices is not None:
            original_indices = self.enhanced_feature_indices['original']
            engineered_indices = self.enhanced_feature_indices.get('engineered', [])
            
            # 只对原始基因表达特征进行缩放
            X_train_original_scaled = self.scaler.fit_transform(X_train[:, original_indices])
            X_test_original_scaled = self.scaler.transform(X_test[:, original_indices])
            
            # 合并缩放后的原始特征和已缩放的工程特征
            if engineered_indices:
                X_train_scaled = np.hstack([X_train_original_scaled, X_train[:, engineered_indices]])
                X_test_scaled = np.hstack([X_test_original_scaled, X_test[:, engineered_indices]])
            else:
                X_train_scaled = X_train_original_scaled
                X_test_scaled = X_test_original_scaled
        else:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_ensemble_model(self, X_train, y_train):
        """构建改进的集成预测模型 - 使用Stacking提高预测精度"""
        print("\n=== 构建改进的集成预测模型 ===")
        
        # 使用Optuna优化基模型超参数
        print("优化梯度提升模型超参数...")
        gb_params = self._optimize_gb_params(X_train, y_train)
        
        # 定义优化后的基模型
        gb_model = GradientBoostingRegressor(
            n_estimators=gb_params['n_estimators'],
            learning_rate=gb_params['learning_rate'],
            max_depth=gb_params['max_depth'],
            min_samples_split=gb_params['min_samples_split'],
            subsample=gb_params['subsample'],
            random_state=self.random_state
        )
        
        rf_model = RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_split=4,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        ridge_model = RidgeCV(
            alphas=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            cv=5
        )
        
        elastic_model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=[0.01, 0.1, 1.0, 10.0],
            cv=5,
            random_state=self.random_state
        )
        
        svr_model = SVR(
            kernel='rbf',
            C=5.0,
            epsilon=0.05,
            gamma='scale'
        )
        
        # 使用Stacking集成（比VotingRegressor更强大）
        estimators = [
            ('gb', gb_model),
            ('rf', rf_model),
            ('ridge', ridge_model),
            ('elastic', elastic_model),
            ('svr', svr_model)
        ]
        
        # 元学习器使用Ridge回归
        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
            cv=5,
            n_jobs=-1
        )
        
        print("训练Stacking集成模型...")
        stacking_model.fit(X_train, y_train)
        
        # 单独训练梯度提升模型用于特征重要性分析
        gb_model.fit(X_train, y_train)
        self.gb_model = gb_model
        
        # 5折交叉验证评估集成模型
        cv_scores = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='r2')
        print(f"Stacking模型交叉验证R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 基因重要性分析
        self.gene_importance = self._analyze_gene_importance(gb_model, X_train)
        
        # 创建SHAP解释器
        try:
            print("\n创建SHAP解释器...")
            # 使用TreeExplainer更高效
            self.explainer = shap.TreeExplainer(gb_model)
            sample_size = min(100, X_train.shape[0])
            self.shap_values = self.explainer.shap_values(X_train[:sample_size])
            self.X_train_sample = X_train[:sample_size]
            print("SHAP解释器创建成功")
        except Exception as e:
            print(f"SHAP解释器创建失败: {e}")
        
        return stacking_model
    
    def _optimize_gb_params(self, X_train, y_train, n_trials=30):
        """使用Optuna优化梯度提升模型超参数"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0)
            }
            
            model = GradientBoostingRegressor(**params, random_state=self.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        print(f"最佳超参数: {study.best_params}")
        print(f"最佳交叉验证R²: {study.best_value:.4f}")
        
        return study.best_params
    
    def _analyze_gene_importance(self, model, X_train):
        """分析基因重要性，重点关注麦角硫因合成关键基因"""
        importance = model.feature_importances_
        
        if self.enhanced_feature_indices is not None:
            original_indices = self.enhanced_feature_indices['original']
            engineered_indices = self.enhanced_feature_indices.get('engineered', [])
            
            original_importance = importance[:len(original_indices)]
            engineered_importance = importance[len(original_indices):] if engineered_indices else np.array([])
            
            print("\n=== 基因重要性分析 ===")
            print(f"原始基因特征数: {len(original_importance)}, 工程特征数: {len(engineered_importance)}")
            
            print("\n关键基因重要性（按麦角硫因合成途径分组）:")
            pathway_importance = {}
            
            for pathway, gene_indices in self.PATHWAY_GENES.items():
                valid_indices = [i for i in gene_indices if i < len(original_importance)]
                if valid_indices:
                    pathway_imp = np.sum([original_importance[i] for i in valid_indices])
                    pathway_importance[pathway] = pathway_imp
            
            sorted_pathways = sorted(pathway_importance.items(), key=lambda x: x[1], reverse=True)
            
            for pathway, imp in sorted_pathways:
                gene_indices = self.PATHWAY_GENES[pathway]
                valid_indices = [i for i in gene_indices if i < len(original_importance)]
                if valid_indices:
                    print(f"{pathway.replace('_', ' ').title()}: {imp:.4f}")
                    pathway_genes = [(i, original_importance[i]) for i in valid_indices]
                    pathway_genes = sorted(pathway_genes, key=lambda x: x[1], reverse=True)[:2]
                    for idx, imp_val in pathway_genes:
                        print(f"  - 基因 {idx+1}: {imp_val:.4f}")
            
            if len(engineered_importance) > 0:
                print("\n工程特征重要性:")
                for i, imp_val in enumerate(engineered_importance):
                    if i < len(self.engineered_feature_names):
                        feature_name = self.engineered_feature_names[i]
                    else:
                        feature_name = f"工程特征{i+1}"
                    print(f"  - {feature_name}: {imp_val:.4f}")
        
        return importance

    
    def _constraint_check(self, gene_expression):
        """检查基因表达是否符合麦角硫因合成的生物约束"""
        for (pathway1, pathway2), (min_ratio, max_ratio) in self.key_ratios.items():
            genes1 = self.PATHWAY_GENES[pathway1]
            genes2 = self.PATHWAY_GENES[pathway2]
            
            genes1 = [g for g in genes1 if g < self.n_genes]
            genes2 = [g for g in genes2 if g < self.n_genes]
            
            if genes1 and genes2:
                expr1 = np.mean([gene_expression[i] for i in genes1])
                expr2 = np.mean([gene_expression[i] for i in genes2])
                
                ratio = expr1 / (expr2 + 1e-6)
                
                if not (min_ratio <= ratio <= max_ratio):
                    return False
        
        total_expression = np.sum(gene_expression)
        if total_expression > 30:
            return False
            
        return True
    
    def optimize_expression_advanced(self, model, X_train, n_trials=200):
        """高级优化：使用改进的贝叶斯优化策略"""
        print("\n=== 高级基因表达优化 ===")
        
        # 获取训练数据的统计信息
        if self.enhanced_feature_indices is not None:
            original_indices = self.enhanced_feature_indices['original']
            X_train_original = X_train[:, :len(original_indices)]
        else:
            X_train_original = X_train
        
        train_mean = X_train_original.mean(axis=0)
        train_std = X_train_original.std(axis=0)
        
        # 定义参数边界（扩大搜索范围）
        bounds = [(train_mean[i] - 2.5*train_std[i], train_mean[i] + 2.5*train_std[i]) 
                  for i in range(self.n_genes)]
        
        self.optimization_history = []
        
        # 获取基因重要性的中位数用于判断
        if self.gene_importance is not None:
            importance_median = np.median(self.gene_importance[:self.n_genes])
        else:
            importance_median = 0
        
        def objective(trial):
            gene_expr = []
            for i in range(self.n_genes):
                low, high = bounds[i]
                
                # 根据基因重要性调整采样策略
                if self.gene_importance is not None and i < len(self.gene_importance):
                    if self.gene_importance[i] > importance_median:
                        # 重要基因：在更宽范围内搜索
                        value = trial.suggest_float(f'gene_{i}', low, high)
                    else:
                        # 次要基因：在更窄范围内搜索
                        narrow_low = train_mean[i] - 1.5*train_std[i]
                        narrow_high = train_mean[i] + 1.5*train_std[i]
                        value = trial.suggest_float(f'gene_{i}', narrow_low, narrow_high)
                else:
                    value = trial.suggest_float(f'gene_{i}', low, high)
                
                gene_expr.append(value)
            
            gene_expr = np.array(gene_expr)
            
            # 检查硬约束
            if not self._constraint_check(gene_expr):
                return -1e6
            
            # 软约束惩罚
            penalty = 0
            
            for (pathway1, pathway2), (min_ratio, max_ratio) in self.key_ratios.items():
                genes1 = self.PATHWAY_GENES[pathway1]
                genes2 = self.PATHWAY_GENES[pathway2]
                
                genes1 = [g for g in genes1 if g < self.n_genes]
                genes2 = [g for g in genes2 if g < self.n_genes]
                
                if genes1 and genes2:
                    expr1 = np.mean([gene_expr[i] for i in genes1])
                    expr2 = np.mean([gene_expr[i] for i in genes2])
                    
                    ratio = expr1 / (expr2 + 1e-6)
                    
                    if ratio < min_ratio:
                        penalty += (min_ratio - ratio) * 50
                    elif ratio > max_ratio:
                        penalty += (ratio - max_ratio) * 50
            
            total_expr = np.sum(gene_expr)
            if total_expr > 25:
                penalty += (total_expr - 25) * 10
            
            # 创建增强特征用于预测
            try:
                gene_expr_reshaped = gene_expr.reshape(1, -1)
                gene_expr_enhanced = self.create_metabolic_pathway_features(gene_expr_reshaped, fit_scaler=False)
                
                if self.enhanced_feature_indices is not None:
                    original_indices = self.enhanced_feature_indices['original']
                    engineered_indices = self.enhanced_feature_indices.get('engineered', [])
                    
                    gene_expr_original_scaled = self.scaler.transform(gene_expr_enhanced[:, :len(original_indices)])
                    
                    if engineered_indices:
                        gene_expr_engineered = gene_expr_enhanced[:, len(original_indices):]
                        gene_expr_final = np.hstack([gene_expr_original_scaled, gene_expr_engineered])
                    else:
                        gene_expr_final = gene_expr_original_scaled
                else:
                    gene_expr_final = self.scaler.transform(gene_expr_enhanced)
                
                prediction = model.predict(gene_expr_final)[0]
            except Exception as e:
                prediction = np.mean(X_train_original) * 0.5
            
            self.optimization_history.append({
                'trial': len(self.optimization_history),
                'gene_expr': gene_expr.copy(),
                'prediction': prediction,
                'penalty': penalty,
                'score': prediction - penalty
            })
            
            return prediction - penalty
        
        # 使用改进的优化器
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                seed=self.random_state,
                consider_prior=True,
                prior_weight=1.0,
                consider_magic_clip=True,
                n_startup_trials=20
            )
        )
        
        print(f"开始贝叶斯优化 ({n_trials} 次试验)...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # 获取最佳表达组合
        best_params = study.best_params
        
        best_gene_expr = []
        for i in range(self.n_genes):
            if f'gene_{i}' in best_params:
                best_gene_expr.append(best_params[f'gene_{i}'])
            else:
                best_gene_expr.append(train_mean[i])
        
        self.best_combination = np.array(best_gene_expr)
        self.best_predicted_yield = study.best_value
        
        print(f"\n优化完成！预测最佳产量: {self.best_predicted_yield:.4f}")
        
        self._analyze_optimization_results(study, X_train_original)
        
        return self.best_combination, self.best_predicted_yield
    
    def _analyze_optimization_results(self, study, X_train):
        """分析优化结果"""
        print("\n=== 优化结果分析 ===")
        
        train_mean = X_train.mean(axis=0)
        train_std = X_train.std(axis=0) + 1e-6
        z_scores = (self.best_combination - train_mean) / train_std
        
        print("\n代谢途径表达变化 (Z-score):")
        for pathway, gene_indices in self.PATHWAY_GENES.items():
            valid_genes = [i for i in gene_indices if i < len(z_scores)]
            if valid_genes:
                pathway_z = np.mean([z_scores[i] for i in valid_genes])
                pathway_std = np.std([z_scores[i] for i in valid_genes]) if len(valid_genes) > 1 else 0
                
                if abs(pathway_z) > 0.3 or pathway_std > 0.5:
                    direction = "↑↑↑" if pathway_z > 1.5 else ("↑↑" if pathway_z > 0.7 else 
                            ("↑" if pathway_z > 0.3 else 
                            ("↓" if pathway_z < -0.3 else 
                            ("↓↓" if pathway_z < -0.7 else "↓↓↓"))))
                    print(f"{pathway.replace('_', ' ').title():25} {direction} {pathway_z:+.2f}σ (±{pathway_std:.2f})")
        
        print("\n关键基因表达变化 (绝对值最大的5个):")
        top_change_indices = np.argsort(np.abs(z_scores))[::-1][:5]
        
        for idx in top_change_indices:
            gene_name = self._get_gene_name(idx)
            change = z_scores[idx]
            direction = "增强" if change > 0 else "降低"
            magnitude = "极显著" if abs(change) > 2.0 else ("显著" if abs(change) > 1.0 else "适度")
            
            print(f"基因 {idx+1:2d} ({gene_name:15}): {direction}表达 ({magnitude}, {change:+.2f}σ)")
            print(f"     优化值: {self.best_combination[idx]:.3f}, 训练平均: {train_mean[idx]:.3f}")
        
        if self.optimization_history:
            scores = [trial['score'] for trial in self.optimization_history]
            best_score_idx = np.argmax(scores)
            
            print(f"\n优化过程: 在 {best_score_idx+1}/{len(scores)} 次试验中找到最佳解")
            print(f"最终得分: {scores[best_score_idx]:.4f} (预测产量: {self.optimization_history[best_score_idx]['prediction']:.4f})")
    
    def _get_gene_name(self, idx):
        """获取基因的代谢功能名称"""
        for pathway, indices in self.PATHWAY_GENES.items():
            if idx in indices:
                if pathway == 'ergothioneine_core':
                    core_genes = ['EgtB', 'EgtC', 'EgtD', 'EgtE', 'Egt1']
                    local_idx = indices.index(idx)
                    if local_idx < len(core_genes):
                        return core_genes[local_idx]
                elif pathway == 'histidine_biosynthesis':
                    his_genes = ['HisA', 'HisB', 'HisC', 'HisD']
                    local_idx = indices.index(idx)
                    if local_idx < len(his_genes):
                        return his_genes[local_idx]
                return pathway.split('_')[0] + f"_{idx}"
        return f"Gene_{idx+1}"

    
    def advanced_validation(self, model, X_test, y_test, X_train=None, y_train=None):
        """高级模型诊断：不仅看整体指标，还要分析预测误差的来源"""
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        residuals = y_test - y_pred
        
        print("\n" + "="*60)
        print("高级模型诊断")
        print("="*60)
        
        print(f"\n1. 基础性能指标:")
        print(f"   R²:     {r2:.4f}")
        print(f"   RMSE:   {rmse:.4f}")
        print(f"   MAE:    {mae:.4f}")
        
        print(f"\n2. 误差分布分析:")
        print(f"   残差均值:      {residuals.mean():.4f}")
        print(f"   残差标准差:    {residuals.std():.4f}")
        print(f"   残差范围:      [{residuals.min():.4f}, {residuals.max():.4f}]")
        print(f"   绝对误差中位数: {np.median(np.abs(residuals)):.4f}")
        
        median_yield = np.median(y_test)
        high_yield_mask = y_test > median_yield
        low_yield_mask = y_test <= median_yield
        
        if high_yield_mask.sum() > 1:
            rmse_high = np.sqrt(mean_squared_error(y_test[high_yield_mask], y_pred[high_yield_mask]))
            try:
                r2_high = r2_score(y_test[high_yield_mask], y_pred[high_yield_mask])
            except:
                r2_high = float('nan')
            print(f"\n3. 高产量样本 (>{median_yield:.2f}):")
            print(f"   样本数: {high_yield_mask.sum()}")
            print(f"   R²:    {r2_high:.4f}")
            print(f"   RMSE:  {rmse_high:.4f}")
        
        if low_yield_mask.sum() > 1:
            rmse_low = np.sqrt(mean_squared_error(y_test[low_yield_mask], y_pred[low_yield_mask]))
            try:
                r2_low = r2_score(y_test[low_yield_mask], y_pred[low_yield_mask])
            except:
                r2_low = float('nan')
            print(f"\n4. 低产量样本 (≤{median_yield:.2f}):")
            print(f"   样本数: {low_yield_mask.sum()}")
            print(f"   R²:    {r2_low:.4f}")
            print(f"   RMSE:  {rmse_low:.4f}")
        
        if X_train is not None and y_train is not None:
            y_pred_train = model.predict(X_train)
            r2_train = r2_score(y_train, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            
            print(f"\n5. 过拟合检查:")
            print(f"   训练集 R²:   {r2_train:.4f}")
            print(f"   测试集 R²:   {r2:.4f}")
            print(f"   泛化差距:    {r2_train - r2:.4f}")
            
            if r2_train - r2 > 0.15:
                print("   ⚠ 警告：模型可能存在过拟合，建议增强正则化。")
            elif r2_train - r2 < 0:
                print("   ⚠ 警告：模型可能存在欠拟合，建议增加模型复杂度。")
            else:
                print("   ✓ 模型泛化能力良好。")
        
        print(f"\n6. 残差正态性检验:")
        sample_size = min(500, len(residuals))
        try:
            _, p_value = stats.shapiro(residuals[:sample_size])
            if p_value > 0.05:
                print(f"   残差近似正态分布 (p={p_value:.4f}) ✓")
            else:
                print(f"   残差不符合正态分布 (p={p_value:.4f})")
                print("   建议：检查模型是否捕获了所有重要关系，或考虑数据转换。")
        except Exception as e:
            print(f"   残差正态性检验失败: {e}")
        
        print(f"\n7. 预测不确定性:")
        
        if X_train is not None and y_train is not None:
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_predictions = []
            cv_actuals = []
            
            for train_idx, val_idx in kf.split(X_train):
                X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                
                cv_model = GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=self.random_state
                )
                cv_model.fit(X_cv_train, y_cv_train)
                cv_pred = cv_model.predict(X_cv_val)
                cv_predictions.extend(cv_pred)
                cv_actuals.extend(y_cv_val)
            
            cv_residuals = np.array(cv_actuals) - np.array(cv_predictions)
            prediction_std = np.std(cv_residuals)
            print(f"   基于交叉验证的预测标准差: {prediction_std:.4f}")
            print(f"   95%预测区间宽度: ±{1.96 * prediction_std:.4f}")
        
        return r2, rmse, mae
    
    def visualize_shap_analysis(self, X_train, feature_names=None):
        """可视化SHAP分析结果"""
        if self.explainer is None or self.shap_values is None:
            print("SHAP解释器未初始化")
            return
        
        try:
            print("\n=== SHAP特征重要性分析 ===")
            
            if feature_names is None:
                feature_names = []
                for i in range(self.original_n_features):
                    feature_names.append(f"Gene_{i+1}")
                
                if self.enhanced_feature_indices is not None:
                    engineered_indices = self.enhanced_feature_indices.get('engineered', [])
                    for i in range(len(engineered_indices)):
                        if i < len(self.engineered_feature_names):
                            feature_names.append(self.engineered_feature_names[i])
                        else:
                            feature_names.append(f"Engineered_{i+1}")
            
            # 确保特征名称数量匹配
            n_features = self.shap_values.shape[1] if len(self.shap_values.shape) > 1 else len(self.shap_values)
            if len(feature_names) > n_features:
                feature_names = feature_names[:n_features]
            elif len(feature_names) < n_features:
                feature_names.extend([f"Feature_{i}" for i in range(len(feature_names), n_features)])
            
            # 1. 特征重要性条形图
            plt.figure(figsize=(12, 8))
            shap.summary_plot(self.shap_values, self.X_train_sample, 
                             feature_names=feature_names, plot_type="bar", show=False)
            plt.title("SHAP特征重要性 (基于梯度提升模型)")
            plt.tight_layout()
            plt.savefig("shap_feature_importance.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2. 特征影响散点图
            plt.figure(figsize=(14, 10))
            shap.summary_plot(self.shap_values, self.X_train_sample, 
                             feature_names=feature_names, show=False)
            plt.title("SHAP特征影响分析")
            plt.tight_layout()
            plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print("SHAP分析图已保存为 'shap_feature_importance.png' 和 'shap_summary_plot.png'")
            
        except Exception as e:
            print(f"SHAP可视化失败: {e}")

    
    def generate_recommendations(self, X_train):
        """生成实验验证建议，专注于麦角硫因合成"""
        if self.best_combination is None:
            print("错误：尚未进行优化")
            return None
        
        if self.enhanced_feature_indices is not None:
            original_indices = self.enhanced_feature_indices['original']
            X_train_original = X_train[:, :len(original_indices)]
        else:
            X_train_original = X_train
        
        train_mean = X_train_original.mean(axis=0)
        train_std = X_train_original.std(axis=0) + 1e-6
        z_scores = (self.best_combination - train_mean) / train_std
        
        print("\n" + "="*60)
        print("麦角硫因合成实验验证建议")
        print("="*60)
        
        print("\n1. 关键调控策略:")
        
        core_genes = self.PATHWAY_GENES['ergothioneine_core']
        core_z = np.mean([z_scores[i] for i in core_genes if i < len(z_scores)])
        if core_z > 0.5:
            print("   - 核心合成酶(EgtB/C/D/E): 建议使用强启动子(如J23100系列)增强表达")
            print("     调控策略: 考虑使用诱导型启动子，在菌体生长后期激活表达")
        else:
            print("   - 核心合成酶(EgtB/C/D/E): 保持中等表达水平，避免中间产物积累")
            print("     调控策略: 使用中等强度组成型启动子(如J23106)")
        
        his_genes = self.PATHWAY_GENES['histidine_biosynthesis']
        his_z = np.mean([z_scores[i] for i in his_genes if i < len(z_scores)])
        if his_z > 0.3:
            print("   - 组氨酸合成途径: 适度增强，确保前体供应")
            print("     调控策略: 增强his操纵子表达，避免反馈抑制")
        else:
            print("   - 组氨酸合成途径: 保持基础表达水平")
        
        cofactor_genes = self.PATHWAY_GENES['cofactor_regeneration']
        cofactor_z = np.mean([z_scores[i] for i in cofactor_genes if i < len(z_scores)])
        if cofactor_z > 0.5:
            print("   - 辅因子再生系统(NADPH/ATP): 优先增强表达，支持高通量合成")
            print("     调控策略: 过表达pntAB(NADPH再生)和atp操纵子")
        
        sulfur_genes = self.PATHWAY_GENES['sulfur_donors']
        sulfur_z = np.mean([z_scores[i] for i in sulfur_genes if i < len(z_scores)])
        if sulfur_z > 0.4:
            print("   - 硫供体途径: 适度增强，避免含硫中间产物毒性")
            print("     调控策略: 增强cys基因簇表达，配合硫源优化")
        
        print("\n2. 分阶段验证策略:")
        print("   阶段1: 仅验证核心合成酶(EgtB/C/D/E)表达比例优化")
        print("         目标: 确定最佳酶比例，避免瓶颈")
        print("         方法: 构建不同启动子强度的组合库")
        
        print("\n   阶段2: 添加组氨酸合成途径和辅因子再生系统优化")
        print("         目标: 平衡前体供应与核心酶需求")
        print("         方法: 启动子工程+RBS优化")
        
        print("\n   阶段3: 完整代谢途径平衡与转运系统优化")
        print("         目标: 最大化产物输出，最小化代谢负担")
        print("         方法: 动态调控+发酵条件优化")
        
        print("\n3. 关键基因调控优先级 (按影响程度排序):")
        sorted_genes = np.argsort(np.abs(z_scores))[::-1]
        
        priority_map = {
            1: "最高优先级 - 立即验证",
            2: "高优先级 - 第一阶段验证",
            3: "中等优先级 - 第二阶段验证",
            4: "低优先级 - 第三阶段验证",
            5: "参考调整 - 微调阶段"
        }
        
        for rank, idx in enumerate(sorted_genes[:10], 1):
            if idx >= len(z_scores):
                continue
                
            gene_name = self._get_gene_name(idx)
            direction = "增强" if z_scores[idx] > 0 else "降低"
            magnitude = "极显著" if abs(z_scores[idx]) > 2.0 else ("显著" if abs(z_scores[idx]) > 1.0 else "适度")
            
            if rank <= 5:
                priority = priority_map[rank]
            else:
                priority = "辅助调整"
            
            change_percent = (self.best_combination[idx] / (train_mean[idx] + 1e-6) - 1) * 100
            
            print(f"   #{rank}: {gene_name:10}")
            print(f"       调控方向: {direction}表达 ({magnitude}, {z_scores[idx]:+.2f}σ)")
            print(f"       变化幅度: {change_percent:+.1f}% ({train_mean[idx]:.3f} → {self.best_combination[idx]:.3f})")
            print(f"       验证优先级: {priority}")
            
            if rank % 3 == 0 and rank < 10:
                print()
        
        print("\n4. 实验设计建议:")
        print("   a. 启动子选择:")
        print("      - 强表达: J23100, T7")
        print("      - 中表达: J23106, J23107")
        print("      - 弱表达: J23114, J23102")
        
        print("\n   b. RBS优化:")
        print("      - 使用RBS计算器设计不同强度的RBS")
        print("      - 考虑使用RBS库筛选最佳序列")
        
        print("\n   c. 发酵条件:")
        print("      - 温度: 30-37°C (根据酶最适温度调整)")
        print("      - pH: 7.0-7.5")
        print("      - 溶氧: >30%饱和度")
        
        total_expr = np.sum(self.best_combination)
        train_total = np.sum(train_mean)
        
        print("\n5. 风险提示与缓解策略:")
        if total_expr > train_total * 1.5:
            print("   ⚠ 重要提示: 预测总表达量较高 (增加{:.1f}%)".format((total_expr/train_total-1)*100))
            print("      风险: 可能导致代谢负担，影响菌体生长")
            print("      缓解策略:")
            print("        - 采用诱导表达策略，生长阶段与生产阶段分离")
            print("        - 使用动态调控系统(如群体感应系统)")
            print("        - 优化培养基成分，支持高表达需求")
        else:
            print("   ✓ 总表达量在合理范围内，代谢负担风险较低")
        
        print("\n6. 关键代谢比例检查:")
        for (pathway1, pathway2), (min_ratio, max_ratio) in self.key_ratios.items():
            genes1 = self.PATHWAY_GENES[pathway1]
            genes2 = self.PATHWAY_GENES[pathway2]
            
            genes1 = [g for g in genes1 if g < len(self.best_combination)]
            genes2 = [g for g in genes2 if g < len(self.best_combination)]
            
            if genes1 and genes2:
                expr1 = np.mean([self.best_combination[i] for i in genes1])
                expr2 = np.mean([self.best_combination[i] for i in genes2])
                ratio = expr1 / (expr2 + 1e-6)
                
                if min_ratio <= ratio <= max_ratio:
                    print(f"   ✓ {pathway1}/{pathway2}: {ratio:.2f} (目标范围: {min_ratio:.1f}-{max_ratio:.1f})")
                else:
                    print(f"   ⚠ {pathway1}/{pathway2}: {ratio:.2f} (超出目标范围: {min_ratio:.1f}-{max_ratio:.1f})")
        
        return {
            'core_enzymes': [self.best_combination[i] for i in core_genes if i < self.n_genes],
            'recommended_changes': {i: z_scores[i] for i in sorted_genes[:10] if i < len(z_scores)},
            'optimization_summary': {
                'predicted_yield': self.best_predicted_yield,
                'total_expression': total_expr,
                'expression_increase': (total_expr/train_total-1)*100
            }
        }

    
    def save_results(self, filepath='ergothioneine_optimization_results.csv'):
        """保存优化结果"""
        if self.best_combination is None:
            print("错误：没有可保存的优化结果")
            return
        
        results_df = pd.DataFrame({
            'gene_index': range(1, self.n_genes + 1),
            'gene_function': [self._get_gene_name(i) for i in range(self.n_genes)],
            'optimal_expression': self.best_combination,
            'expression_level': ['高' if x > 0.7 else ('中' if x > 0.3 else '低') 
                               for x in self.best_combination]
        })
        
        results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n优化结果已保存至: {filepath}")
        
        with open('ergothioneine_experiment_recommendations.txt', 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("麦角硫因合成菌株优化实验建议\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"预测最佳产量: {self.best_predicted_yield:.4f}\n")
            f.write(f"优化完成时间: {pd.Timestamp.now()}\n")
            f.write(f"随机种子: {self.random_state}\n\n")
            
            f.write("关键调控基因:\n")
            f.write("-"*40 + "\n")
            
            core_genes = self.PATHWAY_GENES['ergothioneine_core']
            for i, idx in enumerate(core_genes):
                if idx < self.n_genes:
                    gene_name = self._get_gene_name(idx)
                    expr_value = self.best_combination[idx]
                    
                    f.write(f"\n{gene_name}:\n")
                    f.write(f"   最佳表达水平: {expr_value:.4f}\n")
                    
                    if expr_value > 0.7:
                        f.write("   推荐启动子强度: 强 (如J23100, T7)\n")
                        f.write("   建议RBS: 强RBS序列\n")
                    elif expr_value > 0.3:
                        f.write("   推荐启动子强度: 中 (如J23106, J23107)\n")
                        f.write("   建议RBS: 中等强度RBS\n")
                    else:
                        f.write("   推荐启动子强度: 弱 (如J23114, J23102)\n")
                        f.write("   建议RBS: 弱RBS或天然RBS\n")
                    
                    if gene_name == 'EgtB':
                        f.write("   功能: 组氨酸三甲基化酶\n")
                    elif gene_name == 'EgtC':
                        f.write("   功能: 谷胱甘肽裂解酶\n")
                    elif gene_name == 'EgtD':
                        f.write("   功能: 硫转移酶\n")
                    elif gene_name == 'EgtE':
                        f.write("   功能: 麦角硫因合成酶\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("实验验证计划\n")
            f.write("="*70 + "\n\n")
            
            f.write("阶段1: 核心酶比例验证\n")
            f.write("   - 目标: 确定EgtB:EgtC:EgtD:EgtE最佳比例\n")
            f.write("   - 方法: 构建启动子强度组合库\n")
            f.write("   - 预计时间: 2-3周\n\n")
            
            f.write("阶段2: 前体供应优化\n")
            f.write("   - 目标: 优化组氨酸和辅因子供应\n")
            f.write("   - 方法: his操纵子和辅因子再生系统过表达\n")
            f.write("   - 预计时间: 3-4周\n\n")
            
            f.write("阶段3: 系统整合与发酵优化\n")
            f.write("   - 目标: 最大化产量，最小化代谢负担\n")
            f.write("   - 方法: 动态调控+发酵条件优化\n")
            f.write("   - 预计时间: 4-6周\n")
        
        print("详细实验建议已保存至: ergothioneine_experiment_recommendations.txt")
        
        if self.gene_importance is not None:
            importance_df = pd.DataFrame({
                'feature_index': range(len(self.gene_importance)),
                'importance': self.gene_importance
            })
            
            if self.enhanced_feature_indices is not None:
                feature_types = []
                for i in range(len(self.gene_importance)):
                    if i < len(self.enhanced_feature_indices['original']):
                        feature_types.append('original_gene')
                    else:
                        eng_idx = i - len(self.enhanced_feature_indices['original'])
                        if eng_idx < len(self.engineered_feature_names):
                            feature_types.append(f"engineered_{self.engineered_feature_names[eng_idx]}")
                        else:
                            feature_types.append('engineered')
                
                importance_df['feature_type'] = feature_types
            
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_df.to_csv('feature_importance_analysis.csv', index=False, encoding='utf-8-sig')
            print("特征重要性分析已保存至: feature_importance_analysis.csv")


def generate_synthetic_data(n_samples=500, n_genes=24, random_state=42):
    """生成合成麦角硫因合成数据用于演示"""
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_genes)
    
    for i in range(0, 4, 2):
        X[:, i:i+2] = 0.8 * X[:, i:i+2] + 0.2 * np.random.randn(n_samples, 2)
    
    X[:, 6:11] = 0.7 * X[:, 6:11] + 0.3 * np.random.randn(n_samples, 5)
    X[:, 13:16] = 0.6 * X[:, 13:16] + 0.4 * np.random.randn(n_samples, 3)
    
    core_genes = [6, 7, 8, 9, 10]
    core_effect = 2.5 * np.tanh(0.4 * np.sum(X[:, core_genes], axis=1))
    
    his_genes = [0, 1, 2, 3]
    his_effect = 1.2 * np.tanh(0.3 * np.sum(X[:, his_genes], axis=1))
    
    cofactor_genes = [13, 14, 15]
    cofactor_effect = 1.8 * np.tanh(0.35 * np.sum(X[:, cofactor_genes], axis=1))
    
    transporter_genes = [20, 21]
    transporter_effect = 1.5 * np.tanh(0.3 * np.sum(X[:, transporter_genes], axis=1))
    
    stress_genes = [22, 23]
    stress_effect = -0.6 * np.tanh(0.2 * np.sum(X[:, stress_genes], axis=1))
    
    y = core_effect + his_effect + cofactor_effect + transporter_effect + stress_effect
    
    y += 0.9 * np.tanh(0.25 * X[:, 6] * X[:, 13])
    y += 0.7 * np.tanh(0.2 * X[:, 7] * X[:, 0])
    y += 0.5 * np.tanh(0.15 * X[:, 8] * X[:, 20])
    
    for i in core_genes[:3]:
        y += -0.3 * (X[:, i] - 0.5) ** 2
    
    y += 0.4 * np.random.randn(n_samples)
    
    y = (y - y.min()) / (y.max() - y.min()) * 80 + 20
    
    column_names = [f'Gene_{i+1}' for i in range(n_genes)] + ['Ergothioneine_Yield']
    data = pd.DataFrame(np.column_stack([X, y]), columns=column_names)
    
    print(f"生成合成数据: {n_samples}样本, {n_genes}基因")
    print(f"产量分布 - 最小值: {y.min():.2f}, 最大值: {y.max():.2f}, 平均值: {y.mean():.2f}")
    
    return data


def main():
    """主执行函数"""
    print("="*70)
    print("高级麦角硫因合成优化系统 - 改进版")
    print("="*70)
    
    data_file = 'ergothioneine_data.csv'
    
    if not os.path.exists(data_file):
        print("\n生成合成麦角硫因合成数据用于演示...")
        synthetic_data = generate_synthetic_data(n_samples=500, n_genes=24)
        synthetic_data.to_csv(data_file, index=False)
        print(f"合成数据已保存至: {data_file}")
    
    print("\n初始化高级麦角硫因优化器...")
    optimizer = AdvancedErgothioneineOptimizer(n_genes=24, random_state=42)
    
    print("\n加载麦角硫因合成数据...")
    X, y = optimizer.load_data(data_file)
    
    print("\n预处理数据并创建高级特征...")
    X_train, X_test, y_train, y_test = optimizer.preprocess_data(X, y)
    
    print("\n构建集成预测模型...")
    model = optimizer.build_ensemble_model(X_train, y_train)
    
    print("\n执行高级模型验证...")
    r2, rmse, mae = optimizer.advanced_validation(model, X_test, y_test, X_train, y_train)
    
    print("\n执行SHAP分析...")
    optimizer.visualize_shap_analysis(X_train)
    
    print("\n执行高级基因表达优化...")
    best_combination, best_yield = optimizer.optimize_expression_advanced(model, X_train, n_trials=150)
    
    print("\n生成详细实验验证建议...")
    recommendations = optimizer.generate_recommendations(X_train)
    
    optimizer.save_results()
    
    print("\n" + "="*70)
    print("优化流程完成！")
    print("="*70)
    print("\n下一步:")
    print("1. 查看 'ergothioneine_experiment_recommendations.txt' 获取详细实验计划")
    print("2. 根据 'ergothioneine_optimization_results.csv' 中的基因表达值设计菌株")
    print("3. 使用 'feature_importance_analysis.csv' 了解关键调控因子")
    print("4. 参考SHAP分析图理解模型决策过程")
    print("\n建议首先验证优先级最高的3-5个基因，然后逐步扩大验证范围。")


if __name__ == "__main__":
    main()

