#!/usr/bin/env python3
"""
优化的语义交互分析系统
解决过拟合、多重共线性、数值范围等问题
采用特征选择、正则化、适当预处理等策略
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedInteractionAnalyzer:
    def __init__(self):
        """初始化优化交互分析器"""
        self.merged_data = None
        self.semantic_classes = []
        
        # 特征组织
        self.main_features = []
        self.selected_interaction_features = []
        self.final_features = []
        
        # 模型结果
        self.baseline_results = {}
        self.optimized_results = {}
        self.feature_selection_results = {}
        
    def load_data(self, pixel_file, brightness_file, depth_file, perceptions_file):
        """加载数据，使用已验证的方法"""
        print("📁 加载数据...")
        
        from semantic_triple_interaction_analyzer import SemanticTripleInteractionAnalyzer
        temp_analyzer = SemanticTripleInteractionAnalyzer()
        
        pixel_data, brightness_data, depth_data = temp_analyzer.load_data(pixel_file, brightness_file, depth_file)
        merged_data = temp_analyzer.merge_datasets(pixel_data, brightness_data, depth_data, perceptions_file)
        
        self.semantic_classes = temp_analyzer.semantic_classes
        self.merged_data = merged_data
        
        print(f"✅ 数据加载完成: {merged_data.shape}")
        print(f"🎯 语义类别: {self.semantic_classes}")
        
        return merged_data
    
    def create_smart_interaction_features(self):
        """智能创建交互特征 - 避免过拟合"""
        print("🧠 智能特征工程...")
        
        # 1. 主效应特征 - 只包含实际存在的数值列
        main_features = []
        exclude_cols = ['Image', 'image_id', 'row_index', 'safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
        
        for sem in self.semantic_classes:
            # 检查列是否存在且不在排除列表中
            if sem in self.merged_data.columns and sem not in exclude_cols:
                # 检查是否为数值列
                if pd.api.types.is_numeric_dtype(self.merged_data[sem]):
                    main_features.append(sem)
            if f'{sem}_brightness' in self.merged_data.columns:
                if pd.api.types.is_numeric_dtype(self.merged_data[f'{sem}_brightness']):
                    main_features.append(f'{sem}_brightness')
            if f'{sem}_depth' in self.merged_data.columns:
                if pd.api.types.is_numeric_dtype(self.merged_data[f'{sem}_depth']):
                    main_features.append(f'{sem}_depth')
        self.main_features = main_features
        print(f"  📋 主效应特征: {len(self.main_features)} 个")
        
        # 2. 精选交互特征 - 基于理论意义
        print("  💡 创建有意义的交互特征...")
        
        for sem in self.semantic_classes:
            try:
                P = pd.to_numeric(self.merged_data[sem], errors='coerce').fillna(0)
                B = pd.to_numeric(self.merged_data[f'{sem}_brightness'], errors='coerce').fillna(0)
                D = pd.to_numeric(self.merged_data[f'{sem}_depth'], errors='coerce').fillna(0)
                
                # 标准化到相同范围
                P_norm = P  # 已经是0-1
                B_norm = B / 255.0  # 标准化到0-1
                D_norm = D  # 已经是0-1
                
                # 核心三元交互 (标准化后)
                self.merged_data[f'{sem}_core_interaction'] = P_norm * B_norm * D_norm
            except Exception as e:
                print(f"    ⚠️ 跳过 {sem} 的特征创建: {e}")
                continue
            
            # 占比调节的亮度效应
            self.merged_data[f'{sem}_weighted_brightness'] = P_norm * B_norm
            
            # 占比调节的深度效应  
            self.merged_data[f'{sem}_weighted_depth'] = P_norm * D_norm
            
            # 亮度-深度对比度 (占比作为权重)
            self.merged_data[f'{sem}_brightness_depth_contrast'] = P_norm * abs(B_norm - D_norm)
            
        # 3. 跨语义的关键交互 (仅主要语义)
        print("  🔗 创建跨语义交互...")
        key_semantics = self.semantic_classes[:4]  # 仅前4个，避免组合爆炸
        
        for sem1, sem2 in combinations(key_semantics, 2):
            try:
                # 占比竞争关系
                p1 = pd.to_numeric(self.merged_data[sem1], errors='coerce').fillna(0)
                p2 = pd.to_numeric(self.merged_data[sem2], errors='coerce').fillna(0)
                self.merged_data[f'{sem1}_{sem2}_dominance'] = p1 / (p1 + p2 + 0.001)
                
                # 亮度对比效应
                b1 = pd.to_numeric(self.merged_data[f'{sem1}_brightness'], errors='coerce').fillna(0) / 255.0
                b2 = pd.to_numeric(self.merged_data[f'{sem2}_brightness'], errors='coerce').fillna(0) / 255.0
                self.merged_data[f'{sem1}_{sem2}_brightness_contrast'] = abs(b1 - b2)
            except Exception as e:
                print(f"    ⚠️ 跳过 {sem1}-{sem2} 的交互创建: {e}")
                continue
        
        # 4. 全局交互特征
        print("  🌍 创建全局特征...")
        
        try:
            # 整体亮度多样性
            brightness_cols = [f'{sem}_brightness' for sem in self.semantic_classes]
            brightness_values = self.merged_data[brightness_cols].apply(pd.to_numeric, errors='coerce').fillna(0) / 255.0
            self.merged_data['brightness_diversity'] = brightness_values.std(axis=1)
            
            # 整体深度层次感
            depth_cols = [f'{sem}_depth' for sem in self.semantic_classes]
            depth_values = self.merged_data[depth_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            self.merged_data['depth_layering'] = depth_values.max(axis=1) - depth_values.min(axis=1)
            
            # 主导语义的强度
            pixel_cols = self.semantic_classes
            pixel_values = self.merged_data[pixel_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            max_pixel = pixel_values.max(axis=1)
            self.merged_data['dominance_strength'] = max_pixel
        except Exception as e:
            print(f"    ⚠️ 全局特征创建失败: {e}")
        
        # 获取所有新创建的数值特征
        exclude_cols = self.main_features + ['Image', 'image_id', 'row_index', 'safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
        new_features = [col for col in self.merged_data.columns 
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.merged_data[col])]
        
        print(f"  ✅ 创建了 {len(new_features)} 个精选交互特征")
        print(f"     主效应: {len(self.main_features)}")
        print(f"     交互项: {len(new_features)}")
        
        return new_features
    
    def select_best_features(self, target, max_features=20):
        """特征选择 - 避免维度灾难"""
        print(f"🎯 为 {target} 选择最优特征...")
        
        # 准备数据
        new_features = self.create_smart_interaction_features()
        all_features = self.main_features + new_features
        
        # 过滤实际存在的列
        available_features = [col for col in all_features if col in self.merged_data.columns]
        print(f"  📊 可用特征: {len(available_features)}/{len(all_features)}")
        
        X = self.merged_data[available_features]
        y = self.merged_data[target]
        
        # 处理缺失值和异常值
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 确保特征数量一致
        print(f"  🔍 X形状: {X.shape}, available_features长度: {len(available_features)}")
        
        # 鲁棒标准化
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 确保列数匹配
        if X_scaled.shape[1] != len(available_features):
            print(f"  ⚠️ 形状不匹配: {X_scaled.shape[1]} vs {len(available_features)}")
            if X_scaled.shape[1] < len(available_features):
                available_features = available_features[:X_scaled.shape[1]]
            else:
                # 如果标准化后的特征数更多，截断到available_features的长度
                X_scaled = X_scaled[:, :len(available_features)]
        
        X_scaled = pd.DataFrame(X_scaled, columns=available_features, index=X.index)
        
        # 多种特征选择方法
        selection_methods = {}
        
        # 1. 统计检验选择
        selector_f = SelectKBest(score_func=f_regression, k=min(max_features, len(available_features)))
        X_f = selector_f.fit_transform(X_scaled, y)
        selected_f = selector_f.get_support(indices=True)
        selection_methods['f_regression'] = [available_features[i] for i in selected_f]
        
        # 2. Lasso正则化选择
        lasso = Lasso(alpha=0.01, random_state=42)
        selector_lasso = SelectFromModel(lasso, max_features=max_features)
        X_lasso = selector_lasso.fit_transform(X_scaled, y)
        selected_lasso = selector_lasso.get_support(indices=True)
        selection_methods['lasso'] = [available_features[i] for i in selected_lasso]
        
        # 3. 随机森林重要性选择
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
        selector_rf = SelectFromModel(rf, max_features=max_features)
        X_rf = selector_rf.fit_transform(X_scaled, y)
        selected_rf = selector_rf.get_support(indices=True)
        selection_methods['random_forest'] = [available_features[i] for i in selected_rf]
        
        # 4. 递归特征消除
        estimator = Ridge(alpha=1.0)
        selector_rfe = RFE(estimator, n_features_to_select=max_features)
        X_rfe = selector_rfe.fit_transform(X_scaled, y)
        selected_rfe = selector_rfe.get_support(indices=True)
        selection_methods['rfe'] = [available_features[i] for i in selected_rfe]
        
        # 投票选择最终特征
        feature_votes = {}
        for method_name, features in selection_methods.items():
            for feature in features:
                if feature not in feature_votes:
                    feature_votes[feature] = 0
                feature_votes[feature] += 1
        
        # 选择获得多数票的特征
        min_votes = 2  # 至少2个方法都选中
        final_features = [feat for feat, votes in feature_votes.items() if votes >= min_votes]
        
        # 如果特征太少，降低投票门槛
        if len(final_features) < 10:
            min_votes = 1
            final_features = [feat for feat, votes in feature_votes.items() if votes >= min_votes]
        
        # 限制最大特征数
        if len(final_features) > max_features:
            sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            final_features = [feat for feat, votes in sorted_features[:max_features]]
        
        print(f"  📊 特征选择结果:")
        for method, features in selection_methods.items():
            print(f"    {method}: {len(features)} 个特征")
        print(f"  🎯 最终选择: {len(final_features)} 个特征")
        
        return final_features, scaler
    
    def compare_optimization_strategies(self, target):
        """对比不同优化策略（包含对数变换）"""
        print(f"\n🔬 优化策略对比 - {target.upper()}")
        
        # 原始目标变量
        y_original = self.merged_data[target]
        
        # 对数变换目标变量 
        y_log = np.log(y_original + 0.001)
        print(f"  📊 目标变量分布:")
        print(f"    原始: 范围 [{y_original.min():.3f}, {y_original.max():.3f}], 标准差 {y_original.std():.3f}")
        print(f"    对数: 范围 [{y_log.min():.3f}, {y_log.max():.3f}], 标准差 {y_log.std():.3f}")
        
        # 对比两种目标变量
        targets_to_test = {
            'original': y_original,
            'log_transformed': y_log
        }
        
        all_results = {}
        
        for target_type, y in targets_to_test.items():
            print(f"\n  🧪 测试 {target_type} 目标变量...")
            
            # 确保主特征已初始化
            if not hasattr(self, 'main_features') or not self.main_features:
                _ = self.create_smart_interaction_features()
            
            # 准备特征数据 (所有策略共用)
            X_main = self.merged_data[self.main_features]
            X_main = X_main.fillna(0).replace([np.inf, -np.inf], 0)
            scaler_main = StandardScaler()
            X_main_scaled = scaler_main.fit_transform(X_main)
            
            # 特征选择 (为当前目标变量重新选择)
            selected_features, _ = self.select_best_features(target, max_features=15)
            X_selected = self.merged_data[selected_features]
            X_selected = X_selected.fillna(0).replace([np.inf, -np.inf], 0)
            scaler_selected = StandardScaler()
            X_selected_scaled = scaler_selected.fit_transform(X_selected)
            
            # PCA降维
            interaction_features = [col for col in self.merged_data.columns 
                                  if col not in self.main_features + ['image_id', 'safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']]
            all_features = self.main_features + interaction_features
            X_all = self.merged_data[all_features]
            X_all = X_all.fillna(0).replace([np.inf, -np.inf], 0)
            scaler_all = StandardScaler()
            X_all_scaled = scaler_all.fit_transform(X_all)
            
            pca = PCA(n_components=0.9, random_state=42)
            X_pca = pca.fit_transform(X_all_scaled)
            
            # 模型配置
            models = {
                'Ridge': Ridge(alpha=10.0),
                'Lasso': Lasso(alpha=0.1),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
                'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
            }
            
            # 交叉验证配置
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            results = {
                'baseline_main': {},
                'feature_selected': {},
                'pca_reduced': {},
                'regularized_all': {}
            }
            
            # 评估所有组合
            for model_name, model in models.items():
                try:
                    # 基线模型
                    scores_main = cross_val_score(model, X_main_scaled, y, cv=cv, scoring='r2')
                    results['baseline_main'][model_name] = {
                        'mean': scores_main.mean(),
                        'std': scores_main.std(),
                        'features': len(self.main_features)
                    }
                    
                    # 特征选择模型
                    scores_selected = cross_val_score(model, X_selected_scaled, y, cv=cv, scoring='r2')
                    results['feature_selected'][model_name] = {
                        'mean': scores_selected.mean(),
                        'std': scores_selected.std(),
                        'features': len(selected_features)
                    }
                    
                    # PCA模型
                    scores_pca = cross_val_score(model, X_pca, y, cv=cv, scoring='r2')
                    results['pca_reduced'][model_name] = {
                        'mean': scores_pca.mean(),
                        'std': scores_pca.std(),
                        'features': X_pca.shape[1]
                    }
                    
                    # 正则化全特征模型
                    if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
                        scores_reg = cross_val_score(model, X_all_scaled, y, cv=cv, scoring='r2')
                        results['regularized_all'][model_name] = {
                            'mean': scores_reg.mean(),
                            'std': scores_reg.std(),
                            'features': X_all_scaled.shape[1]
                        }
                    
                except Exception as e:
                    print(f"      {model_name} 失败: {str(e)}")
            
            all_results[target_type] = results
            
            # 显示该目标变量的结果
            print(f"    📊 {target_type.upper()} 结果:")
            for strategy_name, strategy_results in results.items():
                if strategy_results:
                    best_score = max([metrics['mean'] for metrics in strategy_results.values()])
                    print(f"      {strategy_name}: 最佳 R² = {best_score:.4f}")
        
        # 对比两种目标变量的性能
        print(f"\n  🏆 {target.upper()} 最佳性能对比:")
        for target_type in ['original', 'log_transformed']:
            if target_type in all_results:
                best_scores = []
                for strategy_results in all_results[target_type].values():
                    if strategy_results:
                        strategy_best = max([metrics['mean'] for metrics in strategy_results.values()])
                        best_scores.append(strategy_best)
                
                if best_scores:
                    overall_best = max(best_scores)
                    improvement = overall_best - max([max([metrics['mean'] for metrics in all_results['original'][strategy].values()]) for strategy in all_results['original'] if all_results['original'][strategy]]) if target_type == 'log_transformed' else 0
                    
                    status = "🟢" if overall_best > 0.3 else "🟡" if overall_best > 0.1 else "🔴"
                    print(f"    {status} {target_type}: 最佳 R² = {overall_best:.4f}", end="")
                    if target_type == 'log_transformed' and improvement != 0:
                        print(f" (提升: {improvement:+.4f})")
                    else:
                        print()
        
        return all_results
    
    def run_optimization_analysis(self, pixel_file, brightness_file, depth_file, perceptions_file):
        """运行完整的优化分析"""
        print("="*80)
        print("🔧 优化语义交互分析系统")
        print("   解决过拟合、多重共线性、维度灾难等问题")
        print("="*80)
        
        # 加载数据
        self.load_data(pixel_file, brightness_file, depth_file, perceptions_file)
        
        # 初始化特征
        _ = self.create_smart_interaction_features()
        
        # 分析每个感知维度
        perception_targets = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
        
        all_results = {}
        
        for target in perception_targets:
            if target in self.merged_data.columns:
                results = self.compare_optimization_strategies(target)
                all_results[target] = results
        
        return all_results 