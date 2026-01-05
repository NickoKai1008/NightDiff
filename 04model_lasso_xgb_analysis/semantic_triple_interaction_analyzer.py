#!/usr/bin/env python3
"""
语义占比 × 语义亮度 × 语义深度 三元交互分析系统
分析三者如何共同影响感知的非线性关系和阈值效应
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from itertools import combinations
import warnings
import re
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

class SemanticTripleInteractionAnalyzer:
    def __init__(self):
        """初始化三元交互分析器"""
        self.pixel_data = None
        self.brightness_data = None
        self.depth_data = None
        self.perceptions_data = None
        self.merged_data = None
        self.semantic_classes = []
        self.results = {}
        
    def fuzzy_match_columns(self, target_cols, source_cols):
        """
        严格精确匹配 - 用户已确保CSV中有正确的列名
        """
        mapping = {}
        
        for target_col in target_cols:
            target_lower = target_col.lower()
            found = False
            
            for source_col in source_cols:
                # Split by semicolon and check each part, 同时也检查下划线前的部分 
                source_parts = [part.strip().lower() for part in source_col.split(';')]
                
                # 对于亮度数据，还要检查下划线前的部分 (如 tree_mean_brightness -> tree)
                underscore_part = source_col.split('_')[0].lower()
                source_parts.append(underscore_part)
                
                # 严格精确匹配
                if target_lower in source_parts:
                    mapping[target_col] = source_col
                    print(f"  ✅ 精确匹配: {target_col} -> {source_col}")
                    found = True
                    break
            
            if not found:
                print(f"  ❌ 未找到匹配: {target_col}")
                
        return mapping
    
    def standardize_column_names(self, df, mapping):
        """
        Standardize column names based on the mapping.
        """
        # Create reverse mapping
        rename_dict = {v: k for k, v in mapping.items()}
        return df.rename(columns=rename_dict)
        
    def standardize_image_id(self, df):
        """
        Standardize image ID format across datasets with auto-detection.
        """
        df = df.copy()
        
        # Auto-detect ID column
        possible_id_cols = ['image_id', 'Image', 'id', 'ID', 'img_id', 'ImageID', 'ImageName']
        id_column = None
        for col in possible_id_cols:
            if col in df.columns:
                id_column = col
                break
        
        if id_column is None:
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"No ID column found from {possible_id_cols}")
        
        print(f"  Using ID column: '{id_column}'")
        
        # Convert to string and remove common suffixes
        df[id_column] = df[id_column].astype(str)
        df[id_column] = df[id_column].str.replace('.png', '', regex=False)
        df[id_column] = df[id_column].str.replace('.jpg', '', regex=False)
        df[id_column] = df[id_column].str.replace('.jpeg', '', regex=False)
        
        # Rename to standard column name
        if id_column != 'image_id':
            df.rename(columns={id_column: 'image_id'}, inplace=True)
        
        return df
    
    def load_data(self, pixel_file, brightness_file, depth_file):
        """
        Load and preprocess the three data sources with fuzzy matching.
        """
        print("Loading data files...")
        
        # Load pixel ratio data
        self.pixel_data = pd.read_csv(pixel_file)
        print(f"Pixel data shape: {self.pixel_data.shape}")
        
        # Load brightness data
        self.brightness_data = pd.read_csv(brightness_file)
        print(f"Brightness data shape: {self.brightness_data.shape}")
        
        # Load depth data  
        self.depth_data = pd.read_csv(depth_file)
        print(f"Depth data shape: {self.depth_data.shape}")
        
        # Add simple row-based IDs (no image name matching needed)
        self.pixel_data['image_id'] = range(len(self.pixel_data))
        self.brightness_data['image_id'] = range(len(self.brightness_data))
        self.depth_data['image_id'] = range(len(self.depth_data))
        
        # Get semantic columns (exclude ID columns and text columns)
        pixel_semantic_cols = [col for col in self.pixel_data.columns 
                              if col not in ['ImageName', 'ImageID', 'image_id', 'Image']]
        brightness_semantic_cols = [col for col in self.brightness_data.columns 
                                   if col not in ['image_id', 'Image']]
        depth_semantic_cols = [col for col in self.depth_data.columns 
                              if col not in ['image_id', 'Image']]
        
        print(f"Found {len(pixel_semantic_cols)} pixel semantic columns")
        print(f"Found {len(brightness_semantic_cols)} brightness semantic columns")
        print(f"Found {len(depth_semantic_cols)} depth semantic columns")
        
        # 🔍 专门查找streetlight相关列
        print(f"\n🔍 查找包含'streetlight'的列:")
        pixel_streetlight = [col for col in pixel_semantic_cols if 'streetlight' in col.lower()]
        brightness_streetlight = [col for col in brightness_semantic_cols if 'streetlight' in col.lower()]
        depth_streetlight = [col for col in depth_semantic_cols if 'streetlight' in col.lower()]
        
        print(f"  Pixel数据中的streetlight列: {pixel_streetlight}")
        print(f"  Brightness数据中的streetlight列: {brightness_streetlight}")  
        print(f"  Depth数据中的streetlight列: {depth_streetlight}")
        
        # 🔍 查找所有可能相关的列（包含light、lamp等）
        print(f"\n🔍 查找包含'light'/'lamp'的列:")
        pixel_light = [col for col in pixel_semantic_cols if any(word in col.lower() for word in ['light', 'lamp'])]
        brightness_light = [col for col in brightness_semantic_cols if any(word in col.lower() for word in ['light', 'lamp'])]
        depth_light = [col for col in depth_semantic_cols if any(word in col.lower() for word in ['light', 'lamp'])]
        
        print(f"  Pixel数据中的light/lamp列: {pixel_light}")
        print(f"  Brightness数据中的light/lamp列: {brightness_light}")
        print(f"  Depth数据中的light/lamp列: {depth_light}")
        
        # 显示前20个列名供参考
        print(f"\n📋 所有列名预览:")
        print(f"  Pixel前20列: {pixel_semantic_cols[:20]}")
        print(f"  Brightness前20列: {brightness_semantic_cols[:20]}")
        print(f"  Depth前20列: {depth_semantic_cols[:20]}")
        
        # Create fuzzy mappings with detailed output
        print("\n🔍 开始模糊匹配 - Brightness 数据:")
        print(f"  Pixel列 (目标): {pixel_semantic_cols[:10]}...")
        print(f"  Brightness列 (源): {brightness_semantic_cols[:10]}...")
        brightness_mapping = self.fuzzy_match_columns(pixel_semantic_cols, brightness_semantic_cols)
        
        print(f"\n🔍 开始模糊匹配 - Depth 数据:")
        print(f"  Pixel列 (目标): {pixel_semantic_cols[:10]}...")
        print(f"  Depth列 (源): {depth_semantic_cols[:10]}...")
        depth_mapping = self.fuzzy_match_columns(pixel_semantic_cols, depth_semantic_cols)
        
        print(f"\n📊 匹配结果:")
        print(f"  Brightness映射: {len(brightness_mapping)}/{len(pixel_semantic_cols)} 个")
        print(f"  Depth映射: {len(depth_mapping)}/{len(pixel_semantic_cols)} 个")
        
        # 检查关键变量是否被匹配
        key_variables = ['person', 'streetlight', 'tree', 'sky']
        print(f"\n🎯 关键变量匹配检查:")
        for var in key_variables:
            in_brightness = var in brightness_mapping
            in_depth = var in depth_mapping
            print(f"  {var}: Brightness✅" if in_brightness else f"  {var}: Brightness❌", end="")
            print(f" Depth✅" if in_depth else f" Depth❌")
            if in_brightness:
                print(f"    -> Brightness: {brightness_mapping[var]}")
            if in_depth:
                print(f"    -> Depth: {depth_mapping[var]}")
        
        # Find common semantics across all three datasets (exclude 'Image' text column)
        common_pixel_cols = set(pixel_semantic_cols)
        common_brightness_cols = set(brightness_mapping.keys())
        common_depth_cols = set(depth_mapping.keys())
        
        all_common = common_pixel_cols & common_brightness_cols & common_depth_cols
        self.semantic_classes = [col for col in all_common if col != 'Image']
        
        print(f"\n📋 最终语义类别:")
        print(f"  总共找到: {len(self.semantic_classes)} 个共同语义类别")
        print(f"  完整列表: {self.semantic_classes}")
        
        # 检查缺失的关键变量
        missing_vars = [var for var in key_variables if var not in self.semantic_classes]
        if missing_vars:
            print(f"\n⚠️  缺失的关键变量: {missing_vars}")
            print("   这些变量将无法用于分析，需要检查数据源命名")
        else:
            print(f"\n✅ 所有关键变量都成功匹配!")
        
        # Standardize column names
        brightness_subset = self.brightness_data[['image_id'] + list(brightness_mapping.values())].copy()
        brightness_subset = self.standardize_column_names(brightness_subset, brightness_mapping)
        
        depth_subset = self.depth_data[['image_id'] + list(depth_mapping.values())].copy()
        depth_subset = self.standardize_column_names(depth_subset, depth_mapping)
        
        # 检查重命名后的列名
        print(f"重命名后的Brightness列: {brightness_subset.columns.tolist()}")
        print(f"重命名后的Depth列: {depth_subset.columns.tolist()}")
        print(f"目标语义类别: {self.semantic_classes}")
        
        # 使用已经过滤后的语义类别
        valid_semantic_classes = self.semantic_classes  # 已经排除了'Image'
        
        # 找到实际存在的列名
        available_brightness_cols = [col for col in valid_semantic_classes if col in brightness_subset.columns]
        available_depth_cols = [col for col in valid_semantic_classes if col in depth_subset.columns]
        
        print(f"Brightness中可用的列: {available_brightness_cols}")
        print(f"Depth中可用的列: {available_depth_cols}")
        
        # 使用可用的列创建子集
        pixel_subset = self.pixel_data[['image_id'] + valid_semantic_classes].copy()
        brightness_subset = brightness_subset[['image_id'] + available_brightness_cols].copy()
        depth_subset = depth_subset[['image_id'] + available_depth_cols].copy()
        
        print(f"Pixel subset shape: {pixel_subset.shape}")
        print(f"Brightness subset shape: {brightness_subset.shape}")
        print(f"Depth subset shape: {depth_subset.shape}")
        
        return pixel_subset, brightness_subset, depth_subset
        
    def merge_datasets(self, pixel_data, brightness_data, depth_data, perceptions_file='perceptions1.csv'):
        """
        Merge the three data sources with perception data by row index order.
        """
        print("Merging datasets by row index order...")
        
        # Load perception data
        perceptions_data = pd.read_csv(perceptions_file)
        print(f"Perceptions data shape: {perceptions_data.shape}")
        
        # Check if all datasets have the same number of rows
        datasets = [pixel_data, brightness_data, depth_data, perceptions_data]
        dataset_names = ['pixel', 'brightness', 'depth', 'perceptions']
        row_counts = [len(df) for df in datasets]
        
        print(f"Dataset row counts:")
        for name, count in zip(dataset_names, row_counts):
            print(f"  {name}: {count} rows")
        
        if len(set(row_counts)) > 1:
            min_rows = min(row_counts)
            print(f"⚠️  Row counts don't match! Using first {min_rows} rows from each dataset.")
            # Truncate all datasets to the minimum row count
            pixel_data = pixel_data.iloc[:min_rows].copy()
            brightness_data = brightness_data.iloc[:min_rows].copy()
            depth_data = depth_data.iloc[:min_rows].copy()
            perceptions_data = perceptions_data.iloc[:min_rows].copy()
        else:
            print("✅ All datasets have matching row counts")
        
        # Reset indices to ensure proper alignment
        pixel_data = pixel_data.reset_index(drop=True)
        brightness_data = brightness_data.reset_index(drop=True)
        depth_data = depth_data.reset_index(drop=True)
        perceptions_data = perceptions_data.reset_index(drop=True)
        
        # Start with pixel data as base (remove image_id if exists)
        merged = pixel_data.copy()
        if 'image_id' in merged.columns:
            merged = merged.drop('image_id', axis=1)
        print(f"Starting with pixel data: {merged.shape}")
        
        # Add brightness data (rename columns and remove image_id)
        brightness_cols = [col for col in brightness_data.columns if col != 'image_id']
        brightness_subset = brightness_data[brightness_cols].copy()
        brightness_subset = brightness_subset.rename(columns={
            col: f"{col}_brightness" for col in brightness_cols
        })
        
        # Concatenate by column (since rows are aligned by index)
        merged = pd.concat([merged, brightness_subset], axis=1)
        print(f"After adding brightness: {merged.shape}")
        
        # Add depth data (rename columns and remove image_id)
        depth_cols = [col for col in depth_data.columns if col != 'image_id']
        depth_subset = depth_data[depth_cols].copy()
        depth_subset = depth_subset.rename(columns={
            col: f"{col}_depth" for col in depth_cols
        })
        
        # Concatenate by column
        merged = pd.concat([merged, depth_subset], axis=1)
        print(f"After adding depth: {merged.shape}")
        
        # Add perception data
        perception_cols = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
        available_perception_cols = [col for col in perception_cols if col in perceptions_data.columns]
        
        if available_perception_cols:
            perception_subset = perceptions_data[available_perception_cols].copy()
            merged = pd.concat([merged, perception_subset], axis=1)
            print(f"After adding perceptions ({len(available_perception_cols)} cols): {merged.shape}")
        else:
            print("Warning: No perception columns found, skipping perception merge")
            
        # Add a simple index-based ID for reference
        merged['row_index'] = range(len(merged))
        
        self.merged_data = merged
        print(f"Final merged dataset shape: {self.merged_data.shape}")
        
        return self.merged_data
        
    def engineer_features(self):
        """创建三元交互特征"""
        print("🔄 构建三元交互特征...")
        
        feature_count = 0
        
        # 1. 基础三元交互: 占比 × 亮度 × 深度
        print("  💫 基础三元交互...")
        for cls in self.semantic_classes:
            pixel_col = cls
            brightness_col = f'{cls}_brightness'
            depth_col = f'{cls}_depth'
            
            # 三元乘积
            self.merged_data[f'{cls}_triple_product'] = (
                self.merged_data[pixel_col] * 
                self.merged_data[brightness_col] * 
                self.merged_data[depth_col]
            )
            
            # 加权亮度 (占比加权)
            self.merged_data[f'{cls}_weighted_brightness'] = (
                self.merged_data[pixel_col] * self.merged_data[brightness_col]
            )
            
            # 加权深度 (占比加权)
            self.merged_data[f'{cls}_weighted_depth'] = (
                self.merged_data[pixel_col] * self.merged_data[depth_col]
            )
            
            # 亮度-深度交互 (占比调节)
            self.merged_data[f'{cls}_brightness_depth_interaction'] = (
                self.merged_data[brightness_col] * self.merged_data[depth_col] * 
                (1 + self.merged_data[pixel_col])
            )
            
            feature_count += 4
            
        # 2. 跨语义类别交互
        print("  🌐 跨语义交互...")
        for cls1, cls2 in combinations(self.semantic_classes[:5], 2):  # 限制组合数
            # 占比对比
            self.merged_data[f'{cls1}_{cls2}_pixel_ratio'] = (
                self.merged_data[cls1] / (self.merged_data[cls2] + 0.001)
            )
            
            # 亮度对比
            self.merged_data[f'{cls1}_{cls2}_brightness_diff'] = (
                self.merged_data[f'{cls1}_brightness'] - self.merged_data[f'{cls2}_brightness']
            )
            
            # 深度对比
            self.merged_data[f'{cls1}_{cls2}_depth_diff'] = (
                self.merged_data[f'{cls1}_depth'] - self.merged_data[f'{cls2}_depth']
            )
            
            feature_count += 3
            
        # 3. 全局特征
        print("  🌍 全局交互特征...")
        
        # 主导语义 (占比最大)
        pixel_cols = self.semantic_classes
        self.merged_data['dominant_semantic_pixel'] = self.merged_data[pixel_cols].idxmax(axis=1)
        
        # 整体亮度-深度相关性
        brightness_cols = [f'{cls}_brightness' for cls in self.semantic_classes]
        depth_cols = [f'{cls}_depth' for cls in self.semantic_classes]
        
        self.merged_data['total_weighted_brightness'] = sum(
            self.merged_data[cls] * self.merged_data[f'{cls}_brightness'] 
            for cls in self.semantic_classes
        )
        
        self.merged_data['total_weighted_depth'] = sum(
            self.merged_data[cls] * self.merged_data[f'{cls}_depth'] 
            for cls in self.semantic_classes
        )
        
        # 语义多样性指数
        self.merged_data['semantic_diversity'] = sum(
            self.merged_data[cls] * np.log(self.merged_data[cls] + 0.001) 
            for cls in self.semantic_classes
        ) * -1
        
        # 亮度-深度复合指数
        self.merged_data['brightness_depth_composite'] = (
            self.merged_data['total_weighted_brightness'] * 
            self.merged_data['total_weighted_depth']
        )
        
        feature_count += 5
        
        print(f"  ✅ 共创建 {feature_count} 个交互特征")
        
    def compare_models(self, feature_cols, target):
        """分析单个感知目标的多模型性能"""
        
        # 准备特征
        X = self.merged_data[feature_cols]
        y = self.merged_data[target]
        
        # 处理无限值和缺失值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # 模型配置
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
        }
        
        target_results = {}
        
        for model_name, model in models.items():
            try:
                # 交叉验证
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                mean_cv_score = cv_scores.mean()
                std_cv_score = cv_scores.std()
                
                # 训练完整模型
                model.fit(X_scaled, y)
                
                # 特征重要性
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                elif hasattr(model, 'coef_'):
                    feature_importance = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': np.abs(model.coef_)
                    }).sort_values('importance', ascending=False)
                else:
                    feature_importance = None
                
                target_results[model_name] = {
                    'cv_score_mean': mean_cv_score,
                    'cv_score_std': std_cv_score,
                    'cv_scores': cv_scores,
                    'feature_importance': feature_importance,
                    'model': model
                }
                
                print(f"  {model_name}: R² = {mean_cv_score:.4f} ± {std_cv_score:.4f}")
                
            except Exception as e:
                print(f"  {model_name} failed: {str(e)}")
                target_results[model_name] = None
        
        self.results[target] = target_results
    
    def analyze_thresholds(self):
        """阈值效应分析"""
        print("🔄 分析阈值效应...")
        
        perception_targets = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
        
        # 选择关键特征进行阈值分析
        key_features = []
        for cls in self.semantic_classes[:5]:  # 前5个类别
            key_features.extend([
                f'{cls}_triple_product',
                f'{cls}_weighted_brightness', 
                f'{cls}_weighted_depth'
            ])
        
        threshold_results = {}
        
        for target in perception_targets:
            target_thresholds = {}
            y = self.merged_data[target]
            
            for feature in key_features:
                if feature in self.merged_data.columns:
                    x = self.merged_data[feature]
                    
                    # 分段回归检测阈值
                    try:
                        # 尝试不同的分割点
                        best_r2 = -np.inf
                        best_threshold = None
                        
                        quantiles = np.linspace(0.2, 0.8, 20)
                        thresholds = x.quantile(quantiles)
                        
                        for threshold in thresholds:
                            # 分成两段
                            mask_low = x <= threshold
                            mask_high = x > threshold
                            
                            if mask_low.sum() > 10 and mask_high.sum() > 10:
                                # 分别拟合
                                r2_low = 0
                                r2_high = 0
                                
                                if mask_low.sum() > 3:
                                    model_low = LinearRegression()
                                    try:
                                        model_low.fit(x[mask_low].values.reshape(-1, 1), y[mask_low])
                                        pred_low = model_low.predict(x[mask_low].values.reshape(-1, 1))
                                        r2_low = r2_score(y[mask_low], pred_low)
                                    except:
                                        r2_low = 0
                                
                                if mask_high.sum() > 3:
                                    model_high = LinearRegression()
                                    try:
                                        model_high.fit(x[mask_high].values.reshape(-1, 1), y[mask_high])
                                        pred_high = model_high.predict(x[mask_high].values.reshape(-1, 1))
                                        r2_high = r2_score(y[mask_high], pred_high)
                                    except:
                                        r2_high = 0
                                
                                # 加权R²
                                total_r2 = (r2_low * mask_low.sum() + r2_high * mask_high.sum()) / len(x)
                                
                                if total_r2 > best_r2:
                                    best_r2 = total_r2
                                    best_threshold = threshold
                        
                        target_thresholds[feature] = {
                            'threshold': best_threshold,
                            'r2_improvement': best_r2
                        }
                        
                    except Exception as e:
                        target_thresholds[feature] = None
            
            threshold_results[target] = target_thresholds
        
        self.threshold_results = threshold_results
        
    def run_analysis(self, pixel_file, brightness_file, depth_file, perceptions_file='perceptions1.csv'):
        """
        Run the complete analysis pipeline.
        """
        try:
            print("=== Starting Semantic Triple Interaction Analysis ===")
            
            # Load and validate data
            pixel_data, brightness_data, depth_data = self.load_data(pixel_file, brightness_file, depth_file)
            
            # Merge all datasets
            merged_data = self.merge_datasets(pixel_data, brightness_data, depth_data, perceptions_file)
            
            # Engineer interaction features
            print("\nCreating interaction features...")
            self.engineer_features()
            
            # Get feature names - 只选择数值特征
            perception_targets = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
            exclude_cols = ['image_id'] + perception_targets + ['dominant_semantic_pixel']
            
            feature_cols = []
            for col in self.merged_data.columns:
                if col not in exclude_cols:
                    # 检查是否为数值列
                    if self.merged_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        feature_cols.append(col)
                    else:
                        print(f"Skipping non-numeric column: {col} (type: {self.merged_data[col].dtype})")
            
            print(f"Using {len(feature_cols)} numeric features for training")
            
            # Run models for each perception
            
            for perception in perception_targets:
                if perception in self.merged_data.columns:
                    print(f"\nAnalyzing {perception}...")
                    self.compare_models(feature_cols, perception)
                    
            print("\n=== Analysis Complete ===")
            print(f"Results saved for {len(perception_targets)} perceptions")
            
            return self.results
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(traceback.format_exc())
            return None 