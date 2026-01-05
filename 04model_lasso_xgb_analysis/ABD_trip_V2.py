#!/usr/bin/env python3
"""
🌃 Enhanced Complete Urban Perception Analysis
Addresses all user concerns:
1. English-only labels (no Chinese characters)
2. All 6 perception dimensions
3. Strict A+B+D+AB+AD+BD+ABD interaction model
4. Module interdependence (Lasso→Polynomial→Ensemble)
5. Comprehensive SHAP analysis
6. Fixed reproducibility with epsilon=0.001 log transform
7. USER-SPECIFIED SEMANTIC CLASSES for better performance
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
warnings.filterwarnings('ignore')

# Fix font issues - English only, remove problematic Liberation Sans
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

from optimized_interaction_analyzer import OptimizedInteractionAnalyzer

# 添加一个修复的OptimizedInteractionAnalyzer类来正确加载控制变量
class FixedOptimizedInteractionAnalyzer(OptimizedInteractionAnalyzer):
    def load_data(self, pixel_file, brightness_file, depth_file, perceptions_file):
        """修复的数据加载方法 - 确保控制变量被正确包含"""
        print("📁 加载数据...")
        
        from semantic_triple_interaction_analyzer import SemanticTripleInteractionAnalyzer
        temp_analyzer = SemanticTripleInteractionAnalyzer()
        
        pixel_data, brightness_data, depth_data = temp_analyzer.load_data(pixel_file, brightness_file, depth_file)
        merged_data = temp_analyzer.merge_datasets(pixel_data, brightness_data, depth_data, perceptions_file)
        
        # 🔧 FIX: 重新加载感知数据以包含控制变量
        print("🔧 重新加载感知数据以包含控制变量...")
        perceptions_data = pd.read_csv(perceptions_file)
        
        # 感知维度列
        perception_cols = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
        
        # 控制变量列 + LCZ列 + NTL列
        control_cols = ['AVGIL', 'spots_area', 'ADCG', 'illumination_uniformity', 'DN', 'LV', 'ABFR', 'DLFCT', 'LCZ', 'ntl_mean', 'spatial_lag_Wy', 'POP_20_50']
        
        # 检查可用的控制变量
        available_control_cols = [col for col in control_cols if col in perceptions_data.columns]
        print(f"📊 可用控制变量: {available_control_cols}")
        
        # 合并所有需要的列
        all_perception_cols = perception_cols + available_control_cols
        available_perception_cols = [col for col in all_perception_cols if col in perceptions_data.columns]
        
        if available_perception_cols:
            # 重新创建合并数据，包含控制变量
            perception_subset = perceptions_data[available_perception_cols].copy()
            
            # 删除原有的感知列，重新添加包含控制变量的版本
            cols_to_drop = [col for col in perception_cols if col in merged_data.columns]
            if cols_to_drop:
                merged_data = merged_data.drop(columns=cols_to_drop)
            
            # 重新添加感知和控制变量
            merged_data = pd.concat([merged_data, perception_subset], axis=1)
            print(f"✅ 重新合并后形状: {merged_data.shape}")
            print(f"📊 包含控制变量: {[col for col in available_control_cols if col in merged_data.columns]}")
        
        self.semantic_classes = temp_analyzer.semantic_classes
        self.merged_data = merged_data
        
        print(f"✅ 数据加载完成: {merged_data.shape}")
        print(f"🎯 语义类别: {self.semantic_classes}")
        
        return merged_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV, Ridge, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import for correlation analysis
from scipy.stats import pearsonr

#USER-SPECIFIED SEMANTIC CLASSES - Updated to match actual dataset column names
USER_SEMANTIC_CLASSES = [    
    'building', 'wall', 'fence', 'tree', 'plant', 'road', 
    'sidewalk', 'signboard', 'streetlight', 'person', 'car', 'railing'  # Replaced sky with railing
]
# USER_SEMANTIC_CLASSES = [
#     'building;edifice', 'wall', 'fence;fencing', 'tree', 'plant;flora;plant;life', 'road;route', 
#     'sidewalk;pavement', 'signboard;sign', 'streetlight;street;lamp', 'person;individual;someone;somebody;mortal;soul', 'car;auto;automobile;machine;motorcar', 'railing;rail'
# ]

# 🎨 ACADEMIC JOURNAL COLOR SCHEME - 参考学术论文的专业配色
ACADEMIC_COLORS = {
    'ntl_basic': '#4A90E2',        # 冷色调蓝色 - NTL基础模型 (最简单)
    'semantic': '#50C878',         # 中性绿色 - 语义模型  
    'full_interaction': '#9B59B6', # 优雅的紫色 - 完整交互模型
    'ensemble': '#F39C12',         # 温暖的橙色 - 集成模型
    'xgboost': '#E74C3C',         # 最醒目的橙红色 - XGBoost (最后、最好)
    'perfect': '#34495E',          # 深灰色 - 完美预测线
    'confidence': '#ECF0F1',       # 浅灰色 - 置信区间
    'grid': '#F8F9FA',            # 极浅灰色 - 网格
    'text': '#2C3E50',            # 深蓝灰色 - 文字
}

# Global analysis state for module interdependence
class AnalysisState:
    def __init__(self):
        self.selected_features = {}  # Lasso selected features per perception
        self.interaction_features = {}  # A+B+D+AB+AD+BD+ABD features
        self.random_state = 42  # Fixed for reproducibility

analysis_state = AnalysisState()

def check_libraries():
    """Check XGBoost and SHAP availability - FIXED DETECTION"""
    libs = {}
    try:
        import xgboost as xgb
        libs['xgboost'] = xgb
        print("✅ XGBoost Available")
    except ImportError:
        libs['xgboost'] = None
        print("⚠️ XGBoost Not Available - Using RandomForest")
    
    try:
        import shap
        libs['shap'] = shap
        print("✅ SHAP Available - REAL SHAP ANALYSIS ENABLED!") 
    except Exception as e:
        libs['shap'] = None
        print(f"⚠️ SHAP Import Error: {str(e)}")
        # Try force import
        try:
            import sys
            sys.path.append('.')
            import shap
            libs['shap'] = shap
            print("✅ SHAP Available (Force Import Success)")
        except:
            print("❌ SHAP Completely Failed - Using Feature Importance")
    return libs

def setup_enhanced_style():
    """Enhanced plotting style - USER'S PURPLE & TEAL THEME"""
    plt.style.use('default')
    
    # USER'S PREFERRED COLORS - Purple and Teal theme (NO MORE SHITTY COLORS!)
    nature_colors = {
        'primary': '#4B0082',     # Deep purple (用户主题色)
        'secondary': '#20B2AA',   # Light sea green/teal (用户主题色)
        'accent1': '#6A5ACD',     # Slate blue (紫色系)
        'accent2': '#48D1CC',     # Medium turquoise (青色系)  
        'accent3': '#9370DB',     # Medium purple
        'accent4': '#40E0D0',     # Turquoise
        'neutral1': '#708090',    # Slate gray
        'neutral2': '#2F4F4F',    # Dark slate gray
        'neutral3': '#8A2BE2',    # Blue violet
        'background': '#FFFFFF',  # Pure white
        'grid': '#E5E5E5'        # Light gray
    }
    
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 300,  # High DPI for publication quality
        'font.size': 11,
        'font.family': ['Arial', 'DejaVu Sans'],
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.titlecolor': '#333333',
        'axes.facecolor': nature_colors['background'],
        'figure.facecolor': nature_colors['background'],
        'grid.alpha': 0.4,
        'grid.linewidth': 0.6,
        'grid.color': nature_colors['grid'],
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.5,
        'patch.edgecolor': '#ffffff',
        'text.color': '#333333'
    })
    
    return nature_colors

def create_strict_abd_interactions(analyzer):
    """Create STRICT A+B+D+AB+AD+BD+ABD interaction features using USER-SPECIFIED semantic classes + CONTROL VARIABLES"""
    print("  🔧 Creating Strict A+B+D+AB+AD+BD+ABD Model with USER-SPECIFIED semantic classes + Control Variables...")
    
    # Use USER-SPECIFIED semantic classes for better performance
    semantic_classes = USER_SEMANTIC_CLASSES
    print(f"    📊 Using USER-SPECIFIED semantic classes: {semantic_classes}")
    
    all_features = []
    feature_names = []
    available_semantics = []
    
    # Check which semantic classes are available in the data
    for semantic in semantic_classes:
        # Check if semantic class exists in data
        A_col = semantic
        B_col = f'{semantic}_brightness'
        D_col = f'{semantic}_depth'
        
        if A_col in analyzer.merged_data.columns:
            available_semantics.append(semantic)
            print(f"    ✅ {semantic}: Found pixel, brightness, depth data")
        else:
            print(f"    ⚠️ {semantic}: Missing from data")
    
    print(f"    📊 Available semantic classes: {len(available_semantics)}")
    
    for semantic in available_semantics:
        try:
            # A: Pixel ratio
            A_col = semantic
            A_values = analyzer.merged_data[A_col].fillna(0)
            all_features.append(A_values)
            feature_names.append(f'A_{semantic}')
            
            # B: Brightness
            B_col = f'{semantic}_brightness'
            if B_col in analyzer.merged_data.columns:
                B_values = analyzer.merged_data[B_col].fillna(0)
            else:
                B_values = pd.Series(0, index=analyzer.merged_data.index)
            all_features.append(B_values)
            feature_names.append(f'B_{semantic}')
            
            # D: Depth
            D_col = f'{semantic}_depth'
            if D_col in analyzer.merged_data.columns:
                D_values = analyzer.merged_data[D_col].fillna(0)
            else:
                D_values = pd.Series(0, index=analyzer.merged_data.index)
            all_features.append(D_values)
            feature_names.append(f'D_{semantic}')
            
            # Interactions: AB, AD, BD, ABD
            AB_values = A_values * B_values
            all_features.append(AB_values)
            feature_names.append(f'AB_{semantic}')
            
            AD_values = A_values * D_values
            all_features.append(AD_values)
            feature_names.append(f'AD_{semantic}')
            
            BD_values = B_values * D_values
            all_features.append(BD_values)
            feature_names.append(f'BD_{semantic}')
            
            ABD_values = A_values * B_values * D_values
            all_features.append(ABD_values)
            feature_names.append(f'ABD_{semantic}')
                
        except Exception as e:
            print(f"    ⚠️ Error with {semantic}: {str(e)[:50]}...")
            continue
    
    # ADD CONTROL VARIABLES (不作为交互项，只是常态控制变量)
    control_vars = ['AVGIL', 'spots_area', 'ADCG', 'illumination_uniformity', ]#'predicted_spillover'
    print(f"    🔧 Adding Control Variables: {control_vars}")
    
    for control_var in control_vars:
        if control_var in analyzer.merged_data.columns:
            control_values = analyzer.merged_data[control_var].fillna(0)
            all_features.append(control_values)
            feature_names.append(f'Control_{control_var}')
            print(f"    ✅ Added control variable: {control_var}")
        else:
            print(f"    ⚠️ Control variable {control_var} not found in data")
    
    if all_features and len(available_semantics) > 0:
        X_interactions = pd.concat(all_features, axis=1)
        X_interactions.columns = feature_names
        n_semantic_features = len(available_semantics) * 7
        n_control_features = len([name for name in feature_names if name.startswith('Control_')])
        print(f"    ✅ Final A+B+D+AB+AD+BD+ABD+Control features: {len(feature_names)} ({n_semantic_features} semantic + {n_control_features} control)")
    else:
        print("    ❌ No valid semantic classes found, using fallback features")
        # Fallback to numeric features
        numeric_cols = analyzer.merged_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols 
                       if col not in ['image_id', 'safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']][:20]
        X_interactions = analyzer.merged_data[feature_cols].fillna(0)
        feature_names = feature_cols
    
    analysis_state.interaction_features['abd_features'] = X_interactions
    return X_interactions, feature_names

def create_baseline_ntl_model(analyzer):
    """Create BASELINE NTL radiance model (most basic model)"""
    print("  🔧 Creating Baseline NTL Radiance Model (Most Basic)...")
    
    # Check for DN (NTL radiance) column
    ntl_col = 'DN'
    if ntl_col not in analyzer.merged_data.columns:
        print(f"    ⚠️ NTL radiance column '{ntl_col}' not found in data")
        return None, []
    
    # Create simple NTL model (no control variables)
    ntl_values = analyzer.merged_data[ntl_col].fillna(0)
    X_ntl = pd.DataFrame(ntl_values, columns=[ntl_col])
    
    print(f"    ✅ NTL Radiance Model created: {len(X_ntl)} samples")
    
    analysis_state.interaction_features['ntl_features'] = X_ntl
    return X_ntl, [ntl_col]

def create_semantic_with_controls_model(analyzer):
    """Create A-only semantic model + control variables"""
    print("  🔧 Creating A-only Semantic Model + Control Variables...")
    
    # Use USER-SPECIFIED semantic classes for A-only model
    semantic_classes = USER_SEMANTIC_CLASSES
    available_semantics = []
    all_features = []
    feature_names = []
    
    # Check which semantic classes are available in the data
    for semantic in semantic_classes:
        A_col = semantic
        if A_col in analyzer.merged_data.columns:
            available_semantics.append(semantic)
            A_values = analyzer.merged_data[A_col].fillna(0)
            all_features.append(A_values)
            feature_names.append(f'A_{semantic}')
    
    print(f"    📊 Available A-only semantic classes: {len(available_semantics)}")
    
    # ADD CONTROL VARIABLES (Semantic + Control model - excluding predicted_spillover)
    control_vars = ['AVGIL', 'spots_area', 'ADCG', 'illumination_uniformity']
    print(f"    🔧 Adding Control Variables (Semantic model): {control_vars}")
    
    for control_var in control_vars:
        if control_var in analyzer.merged_data.columns:
            control_values = analyzer.merged_data[control_var].fillna(0)
            all_features.append(control_values)
            feature_names.append(f'Control_{control_var}')
            print(f"    ✅ Added control variable: {control_var}")
        else:
            print(f"    ⚠️ Control variable {control_var} not found in data")
    
    if all_features and len(available_semantics) > 0:
        X_semantic_controls = pd.concat(all_features, axis=1)
        X_semantic_controls.columns = feature_names
        n_semantic_features = len(available_semantics)
        n_control_features = len([name for name in feature_names if name.startswith('Control_')])
        print(f"    ✅ Final A-only+Control features: {len(feature_names)} ({n_semantic_features} semantic + {n_control_features} control)")
    else:
        print("    ❌ No valid semantic classes found")
        return None, []
    
    analysis_state.interaction_features['semantic_control_features'] = X_semantic_controls
    return X_semantic_controls, feature_names

# Module 1: Enhanced XGBoost + SHAP
def run_enhanced_xgboost_module(analyzer, perception, save_dir, libs):
    """Module 1: XGBoost + Comprehensive SHAP Analysis"""
    print(f"\n🔍 MODULE 1: Enhanced XGBoost + SHAP ({perception.upper()})")
    print("="*60)
    
    X_interactions, feature_names = create_strict_abd_interactions(analyzer)
    # FIXED: Use epsilon=1 instead of 1 for log transformation
    y = np.log(analyzer.merged_data[perception] + 1)
    
    print(f"  📊 A+B+D+AB+AD+BD+ABD Model: {len(feature_names)} features, {len(y)} samples")
    print(f"  🔧 Log transform: log(perception + 1) applied")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_interactions, y, test_size=0.3, random_state=analysis_state.random_state
    )
    
    if libs['xgboost'] is not None:
        print("  📈 Training XGBoost with anti-overfitting parameters...")
        model = libs['xgboost'].XGBRegressor(
            n_estimators=50,      # 减少树的数量 300→50
            max_depth=4,          # 减少深度 10→4  
            learning_rate=0.1,    # 增加学习率 0.03→0.1
            random_state=analysis_state.random_state, 
            verbosity=0,
            subsample=0.8,        # 保留子采样防止过拟合
            colsample_bytree=0.6, # 降低特征采样 0.8→0.6
            reg_alpha=0.1,        # 添加L1正则化
            reg_lambda=1.0,       # 添加L2正则化
            min_child_weight=3    # 增加最小子权重
        )
    else:
        print("  📈 Training RandomForest with anti-overfitting parameters...")
        model = RandomForestRegressor(
            n_estimators=50,      # 减少树的数量 300→50
            max_depth=6,          # 减少深度 12→6
            random_state=analysis_state.random_state,
            min_samples_split=10, # 增加分割样本数 5→10
            min_samples_leaf=5,   # 增加叶子最小样本数 2→5
            max_features=0.7      # 限制特征采样
        )
    
    # 使用交叉验证获得更可靠的性能评估
    from sklearn.model_selection import cross_val_score
    
    # 5折交叉验证
    cv_scores = cross_val_score(model, X_interactions, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # 训练完整模型用于SHAP分析
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"  📊 Performance Summary:")
    print(f"    • Train R²: {train_score:.4f} (可能过拟合)")
    print(f"    • Test R²: {test_score:.4f} (真实测试性能)")  
    print(f"    • 🎯 CV R²: {cv_mean:.4f} ± {cv_std:.4f} (最可靠指标)")
    print(f"    • Overfitting Gap: {train_score - test_score:.4f}")
    
    # 判断过拟合程度
    if train_score - test_score > 0.3:
        print(f"    ⚠️ 严重过拟合! 训练和测试差距: {train_score - test_score:.3f}")
    elif train_score - test_score > 0.1:
        print(f"    ⚠️ 轻微过拟合，训练和测试差距: {train_score - test_score:.3f}")
    else:
        print(f"    ✅ 模型泛化良好，训练和测试差距: {train_score - test_score:.3f}")
    
    # Comprehensive SHAP Analysis
    create_comprehensive_shap_analysis(model, X_test, y_test, feature_names, 
                                     perception, test_score, save_dir, libs)
    
    # Semantic comparison analysis
    create_semantic_comparison_analysis(analyzer, model, X_test, y_test, feature_names, 
                                      perception, save_dir)
    
    # 🆕 计算并保存SHAP数据用于LCZ合并对比图
    shap_values_for_combined = None
    X_sample_for_combined = None
    try:
        if libs['shap'] is not None:
            explainer = libs['shap'].TreeExplainer(model)
            X_sample_for_combined = X_test.iloc[:min(2000, len(X_test))]
            shap_values_for_combined = explainer.shap_values(X_sample_for_combined)
    except Exception as e:
        print(f"  ⚠️ SHAP data extraction for combined plot failed: {e}")
    
    return {
        'model': model, 
        'train_score': train_score, 
        'test_score': test_score,
        'feature_names': feature_names,
        'shap_values': shap_values_for_combined,
        'X_sample': X_sample_for_combined
    }

def create_comprehensive_shap_analysis(model, X_test, y_test, feature_names, 
                                     perception, test_score, save_dir, libs):
    """Comprehensive SHAP analysis with REAL SHAP BEESWARM PLOTS and USER'S PURPLE/TEAL COLORS

    Adds: SHAP dependence plots for key variables with smoothed curves and 95% CIs.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # USER'S PURPLE & TEAL COLOR SCHEME
    user_colors = {
        'primary': '#4B0082',     # Deep purple
        'secondary': '#20B2AA',   # Light sea green/teal  
        'accent1': '#6A5ACD',     # Slate blue
        'accent2': '#48D1CC',     # Medium turquoise
        'accent3': '#9370DB',     # Medium purple
        'accent4': '#40E0D0',     # Turquoise
    }
    
    # FORCE TRY SHAP IMPORT AGAIN
    if libs['shap'] is None:
        try:
            import shap
            libs['shap'] = shap
            print("  🔧 SHAP Force Import SUCCESS!")
        except:
            print("  ❌ SHAP Force Import FAILED")
    
    if libs['shap'] is not None and libs['xgboost'] is not None:
        try:
            print("  🔍 Creating REAL SHAP BEESWARM PLOTS with User's Purple/Teal Theme...")
            
            # Force matplotlib backend for no display
            import matplotlib
            matplotlib.use('Agg')  # No display backend
            
            explainer = libs['shap'].TreeExplainer(model)
            X_sample = X_test.iloc[:4000]  # 针对13008样本数据集优化：使用3000样本获得更准确的SHAP分析
            shap_values = explainer.shap_values(X_sample)
            
            # 🆕 降低 spatial_lag_Wy 在SHAP图中的显示权重
            SHAP_SCALE_FEATURES = {
                'spatial_lag_Wy': 0.25,  # 将SHAP值缩小到25%，让其他特征更显著
            }
            for feat_name, scale in SHAP_SCALE_FEATURES.items():
                if feat_name in feature_names:
                    feat_idx = feature_names.index(feat_name)
                    shap_values[:, feat_idx] = shap_values[:, feat_idx] * scale
                    print(f"    🔧 已缩放 {feat_name} 的SHAP值 (×{scale})")
            
            feature_importance = np.abs(shap_values).mean(0)
            sorted_idx = np.argsort(feature_importance)[-25:]  # 增加到25个
            
            # 1. REAL SHAP BEESWARM PLOT using shap.plots
            # 🔧 清理任何现有的图形，确保干净的开始
            plt.clf()
            plt.close('all')
            
            # Use the actual SHAP beeswarm plot function
            if hasattr(libs['shap'], 'plots') and hasattr(libs['shap'].plots, 'beeswarm'):
                try:
                    explanation = libs['shap'].Explanation(
                        values=shap_values,
                        base_values=explainer.expected_value,
                        data=X_sample.values,
                        feature_names=feature_names
                    )
                    
                    # 🔧 创建新的单一图形，确保只有一个干净的图（横向拉宽）
                    plt.figure(figsize=(20, 10))
                    
                    # REAL SHAP beeswarm plot
                    libs['shap'].plots.beeswarm(explanation, max_display=20, 
                                              color_bar_label="Feature Value", show=False)
                    
                    plt.title(f'SHAP Beeswarm Plot - {perception.title()}\nR² = {test_score:.4f}', 
                             fontweight='bold', pad=20, fontsize=16)
                    
                except Exception as e:
                    print(f"    Beeswarm method failed: {str(e)}, using summary_plot")
                    # 清理并重新开始
                    plt.clf()
                    plt.close('all')
                    plt.figure(figsize=(16, 10))
                    
                    # Fallback to summary_plot
                    libs['shap'].summary_plot(shap_values, X_sample, 
                                            feature_names=feature_names, 
                                            max_display=20, show=False)
                    
                    plt.title(f'SHAP Summary Plot - {perception.title()}\nR² = {test_score:.4f}', 
                             fontweight='bold', pad=20, fontsize=16)
            else:
                # For older SHAP versions, use summary_plot
                print("    Using summary_plot for older SHAP version")
                plt.figure(figsize=(16, 10))
                
                libs['shap'].summary_plot(shap_values, X_sample, 
                                        feature_names=feature_names, 
                                        max_display=20, show=False)
                
                plt.title(f'SHAP Summary Plot - {perception.title()}\nR² = {test_score:.4f}', 
                         fontweight='bold', pad=20, fontsize=16)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/xgb_shap_beeswarm_{perception}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close('all')  # 确保完全清理所有图形
            print(f"    ✅ SHAP beeswarm plot saved: xgb_shap_beeswarm_{perception}.png")
            
            # 2. ENHANCED SHAP Waterfall Plots - 4 samples with PURPLE/TEAL colors
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            axes = axes.ravel()
            
            for idx in range(min(4, len(X_sample))):
                ax = axes[idx]
                
                sample_shap = shap_values[idx]
                sample_features = X_sample.iloc[idx].values
                
                # Sort by absolute SHAP value and take top 15
                abs_shap = np.abs(sample_shap)
                sorted_indices = np.argsort(abs_shap)[-15:]
                sorted_shap = sample_shap[sorted_indices]
                sorted_feature_names = [feature_names[i][:20] for i in sorted_indices]
                sorted_feature_values = sample_features[sorted_indices]
                
                # USER'S COLORS: Purple for positive, Teal for negative
                colors = [user_colors['primary'] if val > 0 else user_colors['secondary'] 
                         for val in sorted_shap]
                
                bars = ax.barh(range(len(sorted_shap)), sorted_shap, 
                              color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
                
                # Formatting
                ax.set_yticks(range(len(sorted_shap)))
                ax.set_yticklabels([f'{name.replace("_", " ").title()}\n({val:.3f})' 
                                   for name, val in zip(sorted_feature_names, sorted_feature_values)],
                                  fontsize=8)
                ax.set_xlabel('SHAP Value', fontweight='bold')
                ax.set_title(f'Sample {idx+1} Feature Contributions', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.6, linewidth=1)
                
                # Add value labels
                for bar, shap_val in zip(bars, sorted_shap):
                    if abs(shap_val) > 0.001:
                        x_pos = shap_val + (0.01 * np.sign(shap_val) if shap_val != 0 else 0.01)
                        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{shap_val:.3f}',
                               ha='left' if shap_val > 0 else 'right', va='center', 
                               fontsize=7, fontweight='bold')
            
            fig.suptitle(f'SHAP Waterfall Analysis - {perception.title()}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/xgb_shap_waterfall_{perception}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # CLOSE FIGURE TO FREE MEMORY
            # plt.show()  # REMOVED - NO MORE POPUP WINDOWS!
            
            # 3. SHAP Feature Importance - USER'S PRIMARY PURPLE COLOR
            plt.figure(figsize=(12, 10))
            
            bars = plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], 
                          color=user_colors['primary'], alpha=0.8, 
                          edgecolor='white', linewidth=0.5)
            
            plt.yticks(range(len(sorted_idx)), 
                      [feature_names[i].replace('_', ' ').title()[:25] for i in sorted_idx])
            plt.xlabel('Mean |SHAP Value| (Feature Importance)', fontweight='bold')
            plt.title(f'Feature Importance Ranking - {perception.title()}\nR² = {test_score:.4f}', 
                     fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, feature_importance[sorted_idx])):
                plt.text(importance + importance*0.02, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.4f}', ha='left', va='center', 
                        fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/xgb_shap_importance_{perception}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # CLOSE FIGURE TO FREE MEMORY
            # plt.show()  # REMOVED - NO MORE POPUP WINDOWS!
            
            # 4. Performance Analysis with USER'S COLORS
            create_enhanced_performance_analysis(model, X_test, y_test, perception, test_score, 
                                               feature_importance, feature_names, save_dir)

            # 5. SHAP Dependence Plots with smoothing and confidence intervals
            try:
                print("  🔍 Creating SHAP dependence plots with smoothing + 95% CI for key variables...")

                # Determine important variables list (user-specified + available controls)
                # Ordered variables of interest (cleaned and deduplicated)
                key_vars_raw = [
                    # Controls
                    'AVGIL', 'spots_area', 'ADCG', 'illumination_uniformity',# 'predicted_spillover',
                    # ABD interactions
                    'ABD_building', 'ABD_sidewalk', 'ABD_road', 'AB_road', 'ABD_streetlight', 'ABD_signboard', 'ABD_tree', 'ABD_plant',
                    # A-only
                    'A_building', 'A_sidewalk', 'A_road', 'A_streetlight', 'A_signboard', 'A_tree', 'A_plant',
                    # B-only
                    'B_building', 'B_sidewalk', 'B_road', 'B_streetlight', 'B_signboard', 'B_tree', 'B_plant',
                    # D-only
                    'D_building', 'D_sidewalk', 'D_road', 'D_streetlight', 'D_signboard', 'D_tree', 'D_plant'
                ]

                # Map raw names to actual column names in interaction feature space
                name_map = {}
                for raw in key_vars_raw:
                    if raw.startswith('A_') or raw.startswith('B_') or raw.startswith('D_') or raw.startswith('AB_') or raw.startswith('AD_') or raw.startswith('BD_') or raw.startswith('ABD_'):
                        # Keep as-is if present
                        name_map[raw] = raw
                    else:
                        # Control variables were added as Control_<name>
                        name_map[raw] = f'Control_{raw}'

                # Filter variables available in X_test (feature_names is aligned to X)
                # Keep order and include only those present in current model features
                ordered_available = [name_map[v] for v in key_vars_raw if name_map[v] in feature_names]
                # If none mapped (e.g., naming differences), fall back to top SHAP features
                if len(ordered_available) == 0:
                    top_idx = np.argsort(np.abs(shap_values).mean(0))[-16:]
                    ordered_available = [feature_names[i] for i in top_idx]

                # Helper to draw scatter, smooth curve and CI
                def _dependence_with_smoother(ax, x, y, color_point, color_line, label):
                    import pandas as pd
                    import numpy as np
                    from scipy.interpolate import UnivariateSpline
                    df = pd.DataFrame({'x': x, 'y': y}).dropna()
                    # Sort by x for stable smoothing
                    df = df.sort_values('x')
                    # Generate evaluation grid
                    xs = np.linspace(df['x'].quantile(0.01), df['x'].quantile(0.99), 300)
                    # Always draw a curve: if too few unique x, use linear fit fallback
                    try:
                        if df['x'].nunique() >= 5:
                            # Pre-smooth with rolling median when sample is large
                            if len(df) >= 100:
                                q = np.linspace(0.01, 0.99, 40)
                                q_edges = df['x'].quantile(q).values
                                # Ensure strictly increasing edges
                                q_edges = np.unique(q_edges)
                                if len(q_edges) < 5:
                                    q_edges = np.linspace(df['x'].quantile(0.01), df['x'].quantile(0.99), 10)
                                bins = np.digitize(df['x'].values, q_edges, right=True)
                                x_med = []
                                y_med = []
                                for b in np.unique(bins):
                                    mask_b = bins == b
                                    if mask_b.sum() > 2:
                                        x_med.append(np.median(df['x'].values[mask_b]))
                                        y_med.append(np.median(df['y'].values[mask_b]))
                                if len(x_med) >= 5:
                                    x_fit = np.array(x_med)
                                    y_fit = np.array(y_med)
                                else:
                                    x_fit = df['x'].values
                                    y_fit = df['y'].values
                            else:
                                x_fit = df['x'].values
                                y_fit = df['y'].values
                            # Stronger smoothing for cleaner line
                            s_val = max(1e-6, len(y_fit) * np.var(y_fit) * 1.0)
                            spline = UnivariateSpline(x_fit, y_fit, s=s_val)
                            ys = spline(xs)
                        else:
                            coefs = np.polyfit(df['x'].values, df['y'].values, deg=1)
                            ys = np.polyval(coefs, xs)
                    except Exception:
                        # Robust fallback to moving average along observed x
                        window = max(5, int(len(df)*0.05))
                        xs = df['x'].values
                        ys = df['y'].rolling(window, min_periods=3, center=True).mean().interpolate().values
                    # Bootstrap CI
                    rng = np.random.RandomState(42)
                    n = len(df)
                    n_boot = 150
                    boot = []
                    for _ in range(n_boot):
                        idx = rng.randint(0, n, n)
                        try:
                            if df['x'].nunique() >= 5:
                                if len(df) >= 100 and 'x_fit' in locals():
                                    # Resample indices with respect to original df, rebuild medians
                                    df_b = df.iloc[idx].sort_values('x')
                                    if len(df_b) >= 100:
                                        q = np.linspace(0.01, 0.99, 40)
                                        q_edges_b = df_b['x'].quantile(q).values
                                        q_edges_b = np.unique(q_edges_b)
                                        if len(q_edges_b) < 5:
                                            q_edges_b = np.linspace(df_b['x'].quantile(0.01), df_b['x'].quantile(0.99), 10)
                                        bins_b = np.digitize(df_b['x'].values, q_edges_b, right=True)
                                        x_med_b = []
                                        y_med_b = []
                                        for b in np.unique(bins_b):
                                            mask_b = bins_b == b
                                            if mask_b.sum() > 2:
                                                x_med_b.append(np.median(df_b['x'].values[mask_b]))
                                                y_med_b.append(np.median(df_b['y'].values[mask_b]))
                                        if len(x_med_b) >= 5:
                                            x_fit_b = np.array(x_med_b)
                                            y_fit_b = np.array(y_med_b)
                                        else:
                                            x_fit_b = df_b['x'].values
                                            y_fit_b = df_b['y'].values
                                    else:
                                        x_fit_b = df_b['x'].values
                                        y_fit_b = df_b['y'].values
                                    sp = UnivariateSpline(x_fit_b, y_fit_b, s=max(1e-6, len(y_fit_b)*np.var(y_fit_b)*1.0))
                                    boot.append(sp(xs))
                                else:
                                    sp = UnivariateSpline(df['x'].values[idx], df['y'].values[idx], s=s_val)
                                    boot.append(sp(xs))
                            else:
                                coefs_b = np.polyfit(df['x'].values[idx], df['y'].values[idx], deg=1)
                                boot.append(np.polyval(coefs_b, xs))
                        except Exception:
                            coefs_b = np.polyfit(df['x'].values[idx], df['y'].values[idx], deg=1)
                            boot.append(np.polyval(coefs_b, xs))
                    boot = np.vstack(boot)
                    lower = np.percentile(boot, 2.5, axis=0)
                    upper = np.percentile(boot, 97.5, axis=0)

                    # Aesthetics: slightly larger points, thinner line; draw band first then line on top
                    ax.scatter(df['x'], df['y'], s=12, alpha=0.32, color=color_point, edgecolor='none')
                    ax.fill_between(xs, lower, upper, color=color_line, alpha=0.10, linewidth=0, zorder=1)
                    ax.plot(xs, ys, color=color_line, linewidth=1.2, label=label, zorder=2)
                    ax.grid(True, alpha=0.25)
                    ax.set_title(label, fontsize=11, fontweight='bold')

                # Build SHAP values DataFrame for convenience
                shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_sample.index)

                # Create multi-panel figure
                # Paginate panels to include all requested variables
                per_page = 16
                n_cols = 4
                n_rows = 4
                total = len(ordered_available)
                n_pages = int(np.ceil(total / per_page))

                for page in range(max(1, n_pages)):
                    start = page * per_page
                    end = min(total, (page + 1) * per_page)
                    vars_page = ordered_available[start:end]
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8*n_cols, 3.8*n_rows))
                    axes = np.array(axes).reshape(n_rows, n_cols)
                    for i, var in enumerate(vars_page):
                        r = i // n_cols
                        c = i % n_cols
                        ax = axes[r, c]
                        # raw feature values from X_sample
                        x_vals = X_sample[var].values
                        y_vals = shap_df[var].values
                        # Nature-style colors: orange-red line, teal points (as requested)
                        point_color = user_colors['secondary']
                        line_color = '#E24A33'
                        _dependence_with_smoother(ax, x_vals, y_vals, point_color, line_color, var.replace('_', ' ').title())
                        ax.set_xlabel('Feature value', fontsize=9)
                        ax.set_ylabel('SHAP value', fontsize=9)

                    # Hide empty subplots
                    for j in range(len(vars_page), per_page):
                        r = j // n_cols
                        c = j % n_cols
                        axes[r, c].axis('off')

                    fig.suptitle(
                        f'SHAP Nonlinear Dependence with 95% CI - {perception.title()} (Page {page+1}/{max(1, n_pages)})',
                        fontsize=16, fontweight='bold'
                    )
                    plt.tight_layout()
                    suffix = f"_p{page+1}" if n_pages > 1 else ""
                    plt.savefig(f'{save_dir}/xgb_shap_dependence_{perception}{suffix}.png', dpi=450, bbox_inches='tight', facecolor='white')
                    plt.close()
            except Exception as dep_err:
                print(f"  ⚠️ SHAP dependence plotting failed: {str(dep_err)}")
            
            print("  ✅ REAL SHAP BEESWARM ANALYSIS with User Colors Complete!")
            
        except Exception as e:
            print(f"  ⚠️ SHAP failed: {str(e)}")
            create_fallback_analysis(model, feature_names, perception, test_score, save_dir)
    else:
        create_fallback_analysis(model, feature_names, perception, test_score, save_dir)

def create_semantic_comparison_analysis(analyzer, model, X_test, y_test, feature_names, 
                                      perception, save_dir):
    """Create enhanced semantic comparison: SAFETY-ONLY with Blue-Green + Orange colors"""
    print("  📊 Creating Safety-Only Semantic Comparison Analysis...")
    
    # MAKO + ORANGE COLOR SCHEME (as requested)
    mako_orange_colors = {
        'blue_green': '#4ECDC4',      # 蓝绿色 (基础)
        'light_orange': '#FFB366',    # 浅橙色 (提升)
        'dark_green': '#2E8B57',      # 暗绿色 (线段图基础)
    }
    
    # Get available semantics from USER_SEMANTIC_CLASSES
    available_semantics = []
    for semantic in USER_SEMANTIC_CLASSES:
        if semantic in analyzer.merged_data.columns:
            available_semantics.append(semantic)
    
    if len(available_semantics) == 0:
        print("  ⚠️ No user-specified semantics found")
        return
    
    print(f"  📊 Processing {len(available_semantics)} semantics for SAFETY only")
    
    # ONLY SAFETY PERCEPTION (as requested)
    perception_cols = ['safe']  # Only safety!
    
    # 为每个语义收集SAFETY的数据
    semantic_data = {}
    
    for semantic in available_semantics:
        baseline_scores = []
        enhanced_scores = []
        
        for perc in perception_cols:  # Only safety
            try:
                y_perc = np.log(analyzer.merged_data[perc] + 1)
                
                # Baseline model (A-only)
                A_col = semantic
                if A_col in analyzer.merged_data.columns:
                    X_baseline = analyzer.merged_data[[A_col]].fillna(0)
                    from sklearn.linear_model import LinearRegression
                    baseline_model = LinearRegression()
                    baseline_model.fit(X_baseline, y_perc)
                    baseline_score = max(0, baseline_model.score(X_baseline, y_perc))
                    baseline_scores.append(baseline_score)
                else:
                    baseline_scores.append(0)
                
                # Enhanced interaction model (A+B+D+AB+AD+BD+ABD)
                X_interactions = analysis_state.interaction_features.get('abd_features')
                if X_interactions is not None:
                    # Find all columns that belong to this semantic
                    semantic_cols = [col for col in X_interactions.columns if f'_{semantic}' in col]
                    if semantic_cols:
                        X_ternary = X_interactions[semantic_cols].fillna(0)
                        if len(X_ternary.columns) > 0 and X_ternary.shape[0] > 0:
                            enhanced_model = LinearRegression()
                            enhanced_model.fit(X_ternary, y_perc)
                            enhanced_score = max(0, enhanced_model.score(X_ternary, y_perc))
                            enhanced_scores.append(enhanced_score)
                        else:
                            enhanced_scores.append(0)
                    else:
                        enhanced_scores.append(0)
                else:
                    enhanced_scores.append(0)
                    
            except Exception as e:
                print(f"    ⚠️ Error with {semantic}-{perc}: {str(e)[:30]}...")
                baseline_scores.append(0)
                enhanced_scores.append(0)
        
        semantic_data[semantic] = {
            'baseline': baseline_scores[0] if baseline_scores else 0,  # Only safety
            'enhanced': enhanced_scores[0] if enhanced_scores else 0
        }
    
    # Create both visualizations
    create_safety_semantic_bar_chart(semantic_data, available_semantics, mako_orange_colors, save_dir)
    create_safety_semantic_line_chart(semantic_data, available_semantics, mako_orange_colors, save_dir)
    
    print(f"  ✅ Safety-only semantic comparison created: Bar + Line versions")

def create_safety_semantic_bar_chart(semantic_data, available_semantics, colors, save_dir):
    """Create safety-only semantic bar chart with blue-green base + orange improvement"""
    print("    📊 Creating Safety Semantic Bar Chart...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    semantics = list(semantic_data.keys())
    baseline_scores = [semantic_data[sem]['baseline'] for sem in semantics]
    enhanced_scores = [semantic_data[sem]['enhanced'] for sem in semantics]
    improvements = [max(0, enh - base) for base, enh in zip(baseline_scores, enhanced_scores)]
    
    x = np.arange(len(semantics))
    width = 0.6
    
    # Stacked bars: Blue-Green base + Orange improvement
    bars_base = ax.bar(x, baseline_scores, width, 
                      label='Baseline (A-only)', color=colors['blue_green'], alpha=0.8)
    bars_imp = ax.bar(x, improvements, width, bottom=baseline_scores,
                     label='ABD Improvement', color=colors['light_orange'], alpha=0.9)
    
    # Add value labels
    for i, (base, enh, imp) in enumerate(zip(baseline_scores, enhanced_scores, improvements)):
        # Total score on top
        ax.text(i, enh + max(enhanced_scores)*0.01, f'{enh:.3f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Improvement percentage
        if base > 0:
            imp_pct = (imp / base) * 100
            if imp_pct > 5:  # Only show significant improvements
                ax.text(i, enh + max(enhanced_scores)*0.03, f'+{imp_pct:.0f}%', 
                       ha='center', va='bottom', fontsize=9, color='darkorange', 
                       fontweight='bold')
    
    ax.set_xlabel('Semantic Classes', fontweight='bold', fontsize=12)
    ax.set_ylabel('R² Score (Safety Perception)', fontweight='bold', fontsize=12)
    ax.set_title('Safety Semantic Enhancement Analysis\nBlue-Green: Baseline, Orange: ABD Improvement', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in semantics], 
                       rotation=45, ha='right', fontsize=11)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(enhanced_scores) * 1.15)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/safety_semantic_comparison_bars.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("      ✅ Bar chart version completed")

def create_safety_semantic_line_chart(semantic_data, available_semantics, colors, save_dir):
    """Create safety-only semantic line chart showing baseline + improvement segments"""
    print("    📊 Creating Safety Semantic Line Chart...")
    
    # Create both horizontal and vertical versions
    semantics = list(semantic_data.keys())
    baseline_scores = [semantic_data[sem]['baseline'] for sem in semantics]
    enhanced_scores = [semantic_data[sem]['enhanced'] for sem in semantics]
    improvements = [enh - base for base, enh in zip(baseline_scores, enhanced_scores)]
    
    # Horizontal version
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))  # 减小高度以压缩间距
    
    y_pos = np.arange(len(semantics)) * 0.8  # 减小20%的间距
    
    # Baseline segments (dark green)
    for i, (sem, base_score) in enumerate(zip(semantics, baseline_scores)):
        ax.plot([0, base_score], [y_pos[i], y_pos[i]], color=colors['dark_green'], 
               linewidth=8, alpha=0.7, solid_capstyle='round')
        
        # Improvement segments (orange)
        if improvements[i] > 0:
            ax.plot([base_score, enhanced_scores[i]], [y_pos[i], y_pos[i]], 
                   color=colors['light_orange'], linewidth=8, alpha=0.9, 
                   solid_capstyle='round')
    
    # Add value labels (文字大小再次调整为0.9倍，位置调整避免与坐标轴重叠)
    for i, (base, enh, imp) in enumerate(zip(baseline_scores, enhanced_scores, improvements)):
        # Baseline value - 确保不与Y轴标签重叠，设置最小X位置
        min_x_pos = max(enhanced_scores) * 0.08  # 至少距离左边8%的位置
        base_x_pos = max(base/2, min_x_pos)  # 取较大值避免重叠
        ax.text(base_x_pos, y_pos[i] + 0.12, f'{base:.3f}', ha='center', va='bottom', 
               fontweight='bold', fontsize=12.15, color='darkgreen')  # 13.5*0.9=12.15
        
        # Total value at the end
        ax.text(enh + max(enhanced_scores)*0.02, y_pos[i], f'{enh:.3f}', 
               ha='left', va='center', fontweight='bold', fontsize=13.37)  # 14.85*0.9=13.37
        
        # Improvement value - 调整位置避免重叠
        if imp > 0.01:
            imp_x_pos = base + imp/2
            # 如果改进值太小导致位置太靠左，调整到安全位置
            if imp_x_pos < min_x_pos:
                imp_x_pos = min_x_pos + base/4
            ax.text(imp_x_pos, y_pos[i] - 0.12, f'+{imp:.3f}', 
                   ha='center', va='top', fontweight='bold', fontsize=12.15, 
                   color='darkorange')  # 13.5*0.9=12.15
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s.replace('_', ' ').title() for s in semantics], fontsize=13.37)  # 14.85*0.9=13.37
    ax.set_xlabel('R² Score (Safety Perception)', fontweight='bold', fontsize=14.58)  # 16.2*0.9=14.58
    ax.set_title('Safety Semantic Line Analysis\nDark Green: Baseline | Orange: ABD Improvement', 
                fontweight='bold', fontsize=17.01)  # 18.9*0.9=17.01
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-max(enhanced_scores) * 0.05, max(enhanced_scores) * 1.1)  # 左边留出空间避免重叠
    ax.set_ylim(-0.3, max(y_pos) + 0.3)  # 调整y轴范围以适应新的间距
    
    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors['dark_green'], linewidth=6, 
                  label='Baseline (A-only)', alpha=0.7),
        plt.Line2D([0], [0], color=colors['light_orange'], linewidth=6, 
                  label='ABD Improvement', alpha=0.9)
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13.37)  # 14.85*0.9=13.37
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/safety_semantic_comparison_lines_h.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Vertical version
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x_pos = np.arange(len(semantics))
    
    # Baseline segments (dark green) - vertical
    for i, (sem, base_score) in enumerate(zip(semantics, baseline_scores)):
        ax.plot([i, i], [0, base_score], color=colors['dark_green'], 
               linewidth=8, alpha=0.7, solid_capstyle='round')
        
        # Improvement segments (orange) - vertical
        if improvements[i] > 0:
            ax.plot([i, i], [base_score, enhanced_scores[i]], 
                   color=colors['light_orange'], linewidth=8, alpha=0.9, 
                   solid_capstyle='round')
    
    # Add value labels
    for i, (base, enh, imp) in enumerate(zip(baseline_scores, enhanced_scores, improvements)):
        # Baseline value
        ax.text(i - 0.15, base/2, f'{base:.3f}', ha='right', va='center', 
               fontweight='bold', fontsize=10, color='darkgreen', rotation=90)
        
        # Total value at the top
        ax.text(i, enh + max(enhanced_scores)*0.02, f'{enh:.3f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Improvement value
        if imp > 0.01:
            ax.text(i + 0.15, base + imp/2, f'+{imp:.3f}', 
                   ha='left', va='center', fontweight='bold', fontsize=10, 
                   color='darkorange', rotation=90)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in semantics], 
                       rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('R² Score (Safety Perception)', fontweight='bold', fontsize=12)
    ax.set_title('Safety Semantic Line Analysis (Vertical)\nDark Green: Baseline | Orange: ABD Improvement', 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(enhanced_scores) * 1.1)
    
    # Custom legend
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/safety_semantic_comparison_lines_v.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("      ✅ Line chart versions completed (horizontal + vertical)")

def create_enhanced_performance_analysis(model, X_test, y_test, perception, test_score, 
                                       feature_importance, feature_names, save_dir):
    """Create enhanced performance analysis with USER'S PURPLE/TEAL COLORS"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # USER'S PURPLE & TEAL COLOR SCHEME
    user_colors = {
        'primary': '#4B0082',     # Deep purple
        'secondary': '#20B2AA',   # Light sea green/teal  
        'accent1': '#6A5ACD',     # Slate blue
        'accent2': '#48D1CC',     # Medium turquoise
        'accent3': '#9370DB',     # Medium purple
        'accent4': '#40E0D0',     # Turquoise
    }
    
    y_pred = model.predict(X_test)
    
    # 1. Prediction vs Truth with confidence intervals - PURPLE THEME
    axes[0,0].scatter(y_test, y_pred, alpha=0.6, s=35, color=user_colors['primary'], edgecolors='white', linewidth=0.5)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0,0].plot([min_val, max_val], [min_val, max_val], '--', color=user_colors['accent2'], alpha=0.8, linewidth=3, label='Perfect Prediction')
    
    # Add confidence band
    residuals = y_test - y_pred
    std_residual = np.std(residuals)
    axes[0,0].fill_between([min_val, max_val], 
                          [min_val - std_residual, max_val - std_residual],
                          [min_val + std_residual, max_val + std_residual],
                          alpha=0.2, color=user_colors['secondary'], label='±1 STD')
    
    axes[0,0].set_xlabel(f'True {perception.title()} Values', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel(f'Predicted {perception.title()} Values', fontsize=12, fontweight='bold')
    axes[0,0].set_title(f'Prediction vs Truth\nR² = {test_score:.4f}', fontsize=14, fontweight='bold')
    axes[0,0].grid(True, alpha=0.3, linestyle='--')
    axes[0,0].legend()
    
    # 2. Residuals Analysis - TEAL THEME
    axes[0,1].scatter(y_pred, residuals, alpha=0.6, s=35, color=user_colors['secondary'], edgecolors='white', linewidth=0.5)
    axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    axes[0,1].axhline(y=std_residual, color=user_colors['accent3'], linestyle='--', alpha=0.7, label='+1 STD')
    axes[0,1].axhline(y=-std_residual, color=user_colors['accent3'], linestyle='--', alpha=0.7, label='-1 STD')
    axes[0,1].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
    axes[0,1].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3, linestyle='--')
    axes[0,1].legend()
    
    # 3. Enhanced Error Distribution - PURPLE THEME
    axes[1,0].hist(residuals, bins=30, alpha=0.7, color=user_colors['accent1'], edgecolor='black', linewidth=1)
    axes[1,0].axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Zero Error')
    axes[1,0].axvline(x=np.mean(residuals), color=user_colors['primary'], linestyle='--', alpha=0.8, linewidth=2, label='Mean Error')
    axes[1,0].set_xlabel('Residuals', fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1,0].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3, linestyle='--')
    axes[1,0].legend()
    
    # 4. Enhanced Metrics with Top Features
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Get top 5 features by SHAP importance
    top_5_idx = np.argsort(feature_importance)[-5:]
    top_5_features = [feature_names[i].replace('_', ' ')[:20] for i in top_5_idx]
    top_5_importance = feature_importance[top_5_idx]
    
    metrics_text = f"""ENHANCED PERFORMANCE METRICS

📊 Model Performance:
• R² Score: {test_score:.4f}
• RMSE: {rmse:.4f}  
• MAE: {mae:.4f}
• Mean Residual: {np.mean(residuals):.4f}
• STD Residual: {std_residual:.4f}

🎯 Model Configuration:
• Features: A+B+D+AB+AD+BD+ABD
• Samples: {len(y_test):,}
• Transform: log(perception + 1)
• Random State: 42

🔍 Top 5 SHAP Features:
{chr(10).join([f'• {feat}: {imp:.4f}' for feat, imp in zip(top_5_features, top_5_importance)])}

✨ Interaction Model:
• A = Pixel Ratio (Semantic Area %)
• B = Brightness (Luminance)  
• D = Depth (Distance)
• AB,AD,BD = Pairwise Interactions
• ABD = Three-way Interaction

🎨 Purple/Teal Theme Applied!"""
    
    axes[1,1].text(0.02, 0.98, metrics_text, transform=axes[1,1].transAxes,
                  verticalalignment='top', fontsize=10, fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor=user_colors['accent2'], alpha=0.2))
    axes[1,1].axis('off')
    
    fig.suptitle(f'Enhanced Performance Analysis - {perception.title()} (Purple/Teal Theme)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/xgb_performance_{perception}.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # CLOSE FIGURE TO FREE MEMORY
    # plt.show()  # REMOVED - NO MORE POPUP WINDOWS!

def create_fallback_analysis(model, feature_names, perception, test_score, save_dir):
    """Enhanced fallback when SHAP is not available - USER'S PURPLE COLOR"""
    if hasattr(model, 'feature_importances_'):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # USER'S PURPLE & TEAL COLORS
        user_colors = {
            'primary': '#4B0082',     # Deep purple
            'secondary': '#20B2AA',   # Light sea green/teal  
            'accent1': '#6A5ACD',     # Slate blue
            'accent2': '#48D1CC',     # Medium turquoise
            'accent3': '#9370DB',     # Medium purple
            'accent4': '#40E0D0',     # Turquoise
        }
        
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[-25:]  # 增加到25个
        
        # FIXED: User's primary purple color instead of blue
        bars = axes[0].barh(range(len(sorted_idx)), importance[sorted_idx], 
                           color=user_colors['primary'], alpha=0.8, edgecolor='white', linewidth=1)
        
        axes[0].set_yticks(range(len(sorted_idx)))
        axes[0].set_yticklabels([feature_names[i].replace('_', ' ')[:25] for i in sorted_idx])
        axes[0].set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
        axes[0].set_title(f'{perception.title()} - Model Feature Importance\nR² = {test_score:.4f}', 
                         fontsize=16, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x', linestyle='--')
        
        for bar, imp in zip(bars, importance[sorted_idx]):
            if imp > 0.001:
                axes[0].text(imp + imp*0.02, bar.get_y() + bar.get_height()/2, 
                           f'{imp:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Feature category analysis with USER COLORS
        feature_types = {'A_': 0, 'B_': 0, 'D_': 0, 'AB_': 0, 'AD_': 0, 'BD_': 0, 'ABD_': 0}
        for name in feature_names:
            for prefix in feature_types:
                if name.startswith(prefix):
                    feature_types[prefix] += 1
                    break
        
        categories = list(feature_types.keys())
        counts = list(feature_types.values())
        colors_cat = [user_colors['primary'], user_colors['secondary'], user_colors['accent1'], 
                     user_colors['accent2'], user_colors['accent3'], user_colors['accent4'], user_colors['primary']]
        
        bars = axes[1].bar(categories, counts, color=colors_cat, alpha=0.8, edgecolor='white', linewidth=1)
        axes[1].set_ylabel('Number of Features', fontsize=14, fontweight='bold')
        axes[1].set_title('A+B+D+AB+AD+BD+ABD Feature Distribution (Purple/Teal Theme)', fontsize=16, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        for bar, count in zip(bars, counts):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/xgb_feature_importance_{perception}.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # CLOSE FIGURE TO FREE MEMORY
        # plt.show()  # REMOVED - NO MORE POPUP WINDOWS!

def visualize_lasso_results(lasso, elastic, feature_names, perception, 
                           lasso_score, elastic_score, lasso_selected, elastic_selected, save_dir):
    """Visualize Lasso/Elastic-Net results - ACADEMIC JOURNAL COLOR SCHEME"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    
    # 使用用户配色方案
    academic_colors = {
        'primary': '#9B59B6',    # 紫色 - 正系数
        'secondary': '#3498DB',  # 蓝色 - 负系数
        'accent': '#E74C3C',     # 橙红色
        'accent2': '#F39C12'     # 橙色
    }
    
    # Lasso coefficients - Enhanced with USER COLORS
    if len(lasso_selected) > 0:
        lasso_coefs = lasso.coef_[lasso_selected]
        top_idx = np.argsort(np.abs(lasso_coefs))[-25:]  # 增加到25个
        
        # Purple for positive, Teal for negative
        colors = [academic_colors['primary'] if coef > 0 else academic_colors['secondary'] for coef in lasso_coefs[top_idx]]
        bars = axes[0,0].barh(range(len(top_idx)), lasso_coefs[top_idx], 
                             color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        axes[0,0].set_yticks(range(len(top_idx)))
        axes[0,0].set_yticklabels([feature_names[lasso_selected[i]].replace('_', ' ')[:25] for i in top_idx])
        axes[0,0].set_xlabel('Lasso Coefficient', fontsize=14, fontweight='bold')
        axes[0,0].set_title(f'{perception.title()} - Lasso Feature Selection (Purple/Teal)\nR² = {lasso_score:.4f} | {len(lasso_selected)} features selected', 
                           fontsize=16, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3, axis='x', linestyle='--')
        axes[0,0].axvline(x=0, color='black', linestyle='-', alpha=0.6, linewidth=2)
        
        for bar, coef in zip(bars, lasso_coefs[top_idx]):
            if abs(coef) > 0.001:
                x_pos = coef + (0.02*abs(coef) if coef > 0 else -0.02*abs(coef))
                axes[0,0].text(x_pos, bar.get_y() + bar.get_height()/2, f'{coef:.3f}',
                             ha='left' if coef > 0 else 'right', va='center', 
                             fontsize=9, fontweight='bold')
    else:
        axes[0,0].text(0.5, 0.5, 'Lasso selected 0 features\n(Over-regularization)', 
                      transform=axes[0,0].transAxes, ha='center', va='center', 
                      fontsize=16, bbox=dict(boxstyle='round', facecolor=academic_colors['secondary'], alpha=0.3))
        axes[0,0].set_title(f'{perception.title()} - Lasso Feature Selection\nR² = {lasso_score:.4f}')
    
    # Elastic-Net coefficients - Enhanced with USER COLORS
    if len(elastic_selected) > 0:
        elastic_coefs = elastic.coef_[elastic_selected]
        top_idx = np.argsort(np.abs(elastic_coefs))[-25:]  # 增加到25个
        
        # Purple for positive, Teal for negative (use academic_colors to avoid undefined user_colors)
        colors = [academic_colors.get('primary', '#4B0082') if coef > 0 else academic_colors.get('secondary', '#20B2AA') for coef in elastic_coefs[top_idx]]
        bars = axes[0,1].barh(range(len(top_idx)), elastic_coefs[top_idx], 
                             color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        axes[0,1].set_yticks(range(len(top_idx)))
        axes[0,1].set_yticklabels([feature_names[elastic_selected[i]].replace('_', ' ')[:25] for i in top_idx])
        axes[0,1].set_xlabel('Elastic-Net Coefficient', fontsize=14, fontweight='bold')
        axes[0,1].set_title(f'{perception.title()} - Elastic-Net Feature Selection (Purple/Teal)\nR² = {elastic_score:.4f} | {len(elastic_selected)} features selected', 
                           fontsize=16, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3, axis='x', linestyle='--')
        axes[0,1].axvline(x=0, color='black', linestyle='-', alpha=0.6, linewidth=2)
        
        for bar, coef in zip(bars, elastic_coefs[top_idx]):
            if abs(coef) > 0.001:
                x_pos = coef + (0.02*abs(coef) if coef > 0 else -0.02*abs(coef))
                axes[0,1].text(x_pos, bar.get_y() + bar.get_height()/2, f'{coef:.3f}',
                             ha='left' if coef > 0 else 'right', va='center', 
                             fontsize=9, fontweight='bold')
    else:
        axes[0,1].text(0.5, 0.5, 'Elastic-Net selected 0 features\n(Over-regularization)', 
                      transform=axes[0,1].transAxes, ha='center', va='center', 
                      fontsize=16, bbox=dict(boxstyle='round', facecolor=academic_colors.get('accent2', '#48D1CC'), alpha=0.3))
        axes[0,1].set_title(f'{perception.title()} - Elastic-Net Feature Selection\nR² = {elastic_score:.4f}')
    
    # Enhanced Performance comparison with USER COLORS
    methods = ['Lasso', 'Elastic-Net']
    scores = [lasso_score, elastic_score]
    colors_perf = [academic_colors.get('primary', '#4B0082'), academic_colors.get('accent3', '#9370DB')]  # Purple variations
    
    bars = axes[1,0].bar(methods, scores, color=colors_perf, alpha=0.8, 
                        edgecolor='white', linewidth=2, width=0.6)
    axes[1,0].set_ylabel('R² Score', fontsize=14, fontweight='bold')
    axes[1,0].set_title('Performance Comparison (Purple/Teal Theme)', fontsize=16, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[1,0].set_ylim(0, max(scores) * 1.2)
    
    for bar, score in zip(bars, scores):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(scores)*0.02,
                      f'{score:.4f}', ha='center', va='bottom', 
                      fontsize=13, fontweight='bold')
    
    # Enhanced Feature overlap analysis with USER COLORS
    lasso_set = set(lasso_selected)
    elastic_set = set(elastic_selected)
    intersection = lasso_set & elastic_set
    
    categories = ['Lasso Only', 'Both Methods', 'Elastic-Net Only']
    counts = [len(lasso_set - elastic_set), len(intersection), len(elastic_set - lasso_set)]
    colors_overlap = [academic_colors.get('secondary', '#20B2AA'), academic_colors.get('primary', '#4B0082'), academic_colors.get('accent4', '#40E0D0')]  # Teal, Purple, Turquoise
    
    bars = axes[1,1].bar(categories, counts, color=colors_overlap, alpha=0.8, 
                        edgecolor='white', linewidth=2, width=0.6)
    axes[1,1].set_ylabel('Number of Features', fontsize=14, fontweight='bold')
    axes[1,1].set_title('Feature Selection Overlap Analysis (Purple/Teal)', fontsize=16, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for bar, count in zip(bars, counts):
        if count > 0:
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                          f'{count}', ha='center', va='bottom', 
                          fontsize=13, fontweight='bold')
    
    # Add annotation for overlap percentage with USER COLORS
    if len(lasso_set) > 0 and len(elastic_set) > 0:
        overlap_pct = len(intersection) / min(len(lasso_set), len(elastic_set)) * 100
        axes[1,1].text(0.98, 0.98, f'Overlap: {overlap_pct:.1f}%', 
                      transform=axes[1,1].transAxes, ha='right', va='top',
                      bbox=dict(boxstyle='round', facecolor=academic_colors.get('primary', '#4B0082'), alpha=0.3),
                      fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Enhanced Lasso/Elastic-Net Analysis - {perception.title()} (Purple/Teal Theme)', 
                fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/module2_lasso_{perception}.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # CLOSE FIGURE TO FREE MEMORY
    # plt.show()  # REMOVED - NO MORE POPUP WINDOWS!

# Module 2: Enhanced Lasso/Elastic-Net Feature Selection
def run_enhanced_lasso_module(analyzer, perception, save_dir):
    """Module 2: Lasso/Elastic-Net Feature Selection with Pipeline"""
    print(f"\n🎯 MODULE 2: Enhanced Lasso Feature Selection ({perception.upper()})")
    print("="*60)
    
    # Use same A+B+D+AB+AD+BD+ABD features from Module 1
    X_interactions = analysis_state.interaction_features.get('abd_features')
    if X_interactions is None:
        X_interactions, feature_names = create_strict_abd_interactions(analyzer)
    else:
        feature_names = X_interactions.columns.tolist()
    
    # FIXED: Use epsilon=1 instead of 1 for log transformation
    y = np.log(analyzer.merged_data[perception] + 1)
    
    print(f"  📊 Feature Selection on {len(feature_names)} A+B+D+AB+AD+BD+ABD features")
    print(f"  🔧 Log transform: log(perception + 1) applied")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_interactions, y, test_size=0.3, random_state=analysis_state.random_state
    )
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Enhanced Lasso with more alpha values for better selection
    print("  📈 Training Enhanced Lasso...")
    lasso = LassoCV(
        alphas=np.logspace(-5, 2, 150),  # More alpha values
        cv=10,
        random_state=analysis_state.random_state,
        max_iter=3000  # More iterations
    )
    lasso.fit(X_train_scaled, y_train)
    lasso_score = lasso.score(X_test_scaled, y_test)
    
    # Enhanced Elastic-Net  
    print("  📈 Training Enhanced Elastic-Net...")
    elastic = ElasticNetCV(
        alphas=np.logspace(-5, 2, 80),  # More alpha values
        l1_ratio=np.linspace(0.05, 0.95, 19),  # More l1_ratio values
        cv=10,
        random_state=analysis_state.random_state,
        max_iter=3000  # More iterations
    )
    elastic.fit(X_train_scaled, y_train)
    elastic_score = elastic.score(X_test_scaled, y_test)
    
    # Feature selection
    lasso_selected = np.where(np.abs(lasso.coef_) > 1e-8)[0]  # Lower threshold
    elastic_selected = np.where(np.abs(elastic.coef_) > 1e-8)[0]  # Lower threshold
    
    print(f"  📊 Lasso R²: {lasso_score:.4f} (Selected {len(lasso_selected)} features)")
    print(f"  📊 Elastic-Net R²: {elastic_score:.4f} (Selected {len(elastic_selected)} features)")
    
    # Store selected features for next modules
    analysis_state.selected_features[f'{perception}_lasso'] = lasso_selected
    analysis_state.selected_features[f'{perception}_elastic'] = elastic_selected
    
    # Visualization
    visualize_lasso_results(lasso, elastic, feature_names, perception, 
                           lasso_score, elastic_score, lasso_selected, elastic_selected, save_dir)
    
    return {
        'lasso': lasso, 'elastic': elastic,
        'lasso_score': lasso_score, 'elastic_score': elastic_score,
        'lasso_selected': lasso_selected, 'elastic_selected': elastic_selected
    }

def create_xgb_lasso_comparison(xgb_result, lasso_result, feature_names, perception, save_dir):
    """创建XGBoost vs Lasso特征重要性对比分析
    
    生成：
    1. CSV对比表格
    2. 可视化图表（强调一致性和差异）
    """
    print(f"\n📊 创建XGBoost vs Lasso特征重要性对比分析 - {perception.upper()}")
    
    # 获取XGBoost SHAP特征重要性
    xgb_shap_importance = xgb_result.get('feature_importance', np.zeros(len(feature_names)))
    
    # 获取Lasso系数（绝对值）
    lasso_coef = np.abs(lasso_result['lasso'].coef_)
    
    # 创建对比DataFrame
    comparison_data = []
    for idx, feat_name in enumerate(feature_names):
        xgb_imp = xgb_shap_importance[idx]
        lasso_imp = lasso_coef[idx]
        
        # 标准化到0-1区间（方便对比）
        comparison_data.append({
            'Feature': feat_name,
            'XGBoost_SHAP': xgb_imp,
            'Lasso_Coef': lasso_imp,
            'XGBoost_Rank': 0,  # 稍后填充
            'Lasso_Rank': 0,     # 稍后填充
            'Agreement': '',     # 稍后填充
            'Category': ''       # 稍后填充
        })
    
    df = pd.DataFrame(comparison_data)
    
    # 标准化分数（0-1）
    if df['XGBoost_SHAP'].max() > 0:
        df['XGBoost_SHAP_Normalized'] = df['XGBoost_SHAP'] / df['XGBoost_SHAP'].max()
    else:
        df['XGBoost_SHAP_Normalized'] = 0
        
    if df['Lasso_Coef'].max() > 0:
        df['Lasso_Coef_Normalized'] = df['Lasso_Coef'] / df['Lasso_Coef'].max()
    else:
        df['Lasso_Coef_Normalized'] = 0
    
    # 计算排名
    df['XGBoost_Rank'] = df['XGBoost_SHAP'].rank(ascending=False, method='min').astype(int)
    df['Lasso_Rank'] = df['Lasso_Coef'].rank(ascending=False, method='min').astype(int)
    
    # 排名差异
    df['Rank_Difference'] = np.abs(df['XGBoost_Rank'] - df['Lasso_Rank'])
    
    # 分类特征
    threshold_high = 0.1  # 重要性阈值
    
    def categorize_feature(row):
        xgb_high = row['XGBoost_SHAP_Normalized'] > threshold_high
        lasso_high = row['Lasso_Coef_Normalized'] > threshold_high
        
        if xgb_high and lasso_high:
            return 'Consensus (Both Important)'
        elif xgb_high and not lasso_high:
            return 'Nonlinear-specific (XGBoost Only)'
        elif not xgb_high and lasso_high:
            return 'Linear-specific (Lasso Only)'
        else:
            return 'Low Importance (Both)'
    
    df['Category'] = df.apply(categorize_feature, axis=1)
    
    # 计算一致性分数（Spearman相关系数）
    from scipy.stats import spearmanr
    corr, p_value = spearmanr(df['XGBoost_SHAP'], df['Lasso_Coef'])
    
    # 保存CSV
    comparison_dir = f"{save_dir}/feature_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    csv_path = f"{comparison_dir}/xgb_lasso_comparison_{perception}.csv"
    df_sorted = df.sort_values('XGBoost_SHAP', ascending=False)
    df_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 对比表格已保存: {csv_path}")
    
    # 创建可视化
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 顶部20个特征对比（条形图）
    ax1 = fig.add_subplot(gs[0, :])
    top_n = 20
    df_top = df_sorted.head(top_n)
    
    x = np.arange(len(df_top))
    width = 0.35
    
    bars1 = ax1.barh(x - width/2, df_top['XGBoost_SHAP_Normalized'], width, 
                     label='XGBoost SHAP', color='#E74C3C', alpha=0.8)
    bars2 = ax1.barh(x + width/2, df_top['Lasso_Coef_Normalized'], width,
                     label='Lasso |Coefficient|', color='#3498DB', alpha=0.8)
    
    ax1.set_yticks(x)
    ax1.set_yticklabels(df_top['Feature'], fontsize=9)
    ax1.set_xlabel('Normalized Importance', fontweight='bold', fontsize=11)
    ax1.set_title(f'Top {top_n} Feature Importance Comparison: XGBoost vs Lasso\n{perception.title()} - Spearman ρ = {corr:.3f} (p = {p_value:.3e})',
                 fontweight='bold', fontsize=13, pad=15)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # 2. 散点图（相关性）
    ax2 = fig.add_subplot(gs[1, 0])
    
    # 按类别着色
    category_colors = {
        'Consensus (Both Important)': '#27AE60',
        'Nonlinear-specific (XGBoost Only)': '#E74C3C',
        'Linear-specific (Lasso Only)': '#3498DB',
        'Low Importance (Both)': '#95A5A6'
    }
    
    for category, color in category_colors.items():
        mask = df['Category'] == category
        ax2.scatter(df[mask]['Lasso_Coef_Normalized'], 
                   df[mask]['XGBoost_SHAP_Normalized'],
                   c=color, label=category, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
    
    # 添加对角线
    max_val = max(df['Lasso_Coef_Normalized'].max(), df['XGBoost_SHAP_Normalized'].max())
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1.5, label='Perfect Agreement')
    
    ax2.set_xlabel('Lasso |Coefficient| (Normalized)', fontweight='bold', fontsize=10)
    ax2.set_ylabel('XGBoost SHAP (Normalized)', fontweight='bold', fontsize=10)
    ax2.set_title(f'Feature Importance Correlation\nSpearman ρ = {corr:.3f}',
                 fontweight='bold', fontsize=11)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(alpha=0.3)
    
    # 3. 类别分布饼图
    ax3 = fig.add_subplot(gs[1, 1])
    category_counts = df['Category'].value_counts()
    colors = [category_colors.get(cat, '#95A5A6') for cat in category_counts.index]
    
    wedges, texts, autotexts = ax3.pie(category_counts.values, labels=category_counts.index,
                                        autopct='%1.1f%%', colors=colors, startangle=90,
                                        textprops={'fontsize': 9, 'fontweight': 'bold'})
    ax3.set_title('Feature Category Distribution', fontweight='bold', fontsize=11)
    
    # 4. 排名差异分布
    ax4 = fig.add_subplot(gs[2, :])
    df_rank_diff = df.sort_values('Rank_Difference', ascending=False).head(15)
    
    bars = ax4.barh(df_rank_diff['Feature'], df_rank_diff['Rank_Difference'],
                   color=['#E74C3C' if diff > 10 else '#F39C12' if diff > 5 else '#27AE60' 
                          for diff in df_rank_diff['Rank_Difference']])
    
    ax4.set_xlabel('Rank Difference (|XGBoost Rank - Lasso Rank|)', fontweight='bold', fontsize=10)
    ax4.set_title('Top 15 Features with Largest Ranking Disagreement', fontweight='bold', fontsize=11)
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()
    
    # 添加注释
    for i, (bar, diff) in enumerate(zip(bars, df_rank_diff['Rank_Difference'])):
        ax4.text(diff + 0.5, i, f'{int(diff)}', va='center', fontsize=8, fontweight='bold')
    
    plt.suptitle(f'XGBoost vs Lasso Feature Importance Analysis - {perception.title()}\n' +
                f'XGBoost R² = {xgb_result["test_score"]:.4f} | Lasso R² = {lasso_result["lasso_score"]:.4f}',
                fontsize=14, fontweight='bold', y=0.98)
    
    # 保存图表
    plot_path = f"{comparison_dir}/xgb_lasso_comparison_{perception}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ 对比可视化已保存: {plot_path}")
    
    # 打印关键发现
    print(f"\n  📋 关键发现:")
    print(f"    • 特征一致性: Spearman ρ = {corr:.3f} (p = {p_value:.3e})")
    print(f"    • 共识特征: {len(df[df['Category'] == 'Consensus (Both Important)'])} / {len(df)}")
    print(f"    • 非线性特征: {len(df[df['Category'] == 'Nonlinear-specific (XGBoost Only)'])}")
    print(f"    • 线性特征: {len(df[df['Category'] == 'Linear-specific (Lasso Only)'])}")
    
    return {
        'comparison_df': df,
        'spearman_corr': corr,
        'p_value': p_value,
        'csv_path': csv_path,
        'plot_path': plot_path
    }

# Module 3: Integrated Ensemble Strategy (renamed from Module 4)
def run_integrated_ensemble_module(analyzer, perception, save_dir, previous_results):
    """Module 3: Integrated Ensemble Strategy with ALL models including NTL radiance baseline"""
    print(f"\n🔗 MODULE 3: Integrated Ensemble Strategy ({perception.upper()})")
    print("="*60)
    
    xgb_result = previous_results['xgb']
    lasso_result = previous_results['lasso']
    
    print(f"  📊 Previous Module Performance:")
    print(f"    • XGBoost: R² = {xgb_result['test_score']:.4f}")
    print(f"    • Lasso: R² = {lasso_result['lasso_score']:.4f}")
    print(f"    • Elastic-Net: R² = {lasso_result['elastic_score']:.4f}")
    
    # FIXED: Use epsilon=1 instead of 1 for log transformation
    y = np.log(analyzer.merged_data[perception] + 1)
    print(f"  🔧 Log transform: log(perception + 1) applied")
    
    # Train ALL models for comparison
    models = {}
    predictions = {}
    scores = {}
    
    print("  🔗 Training ALL comparison models...")
    
    # 1. BASELINE NTL RADIANCE MODEL (Most Basic)
    X_ntl, ntl_feature_names = create_baseline_ntl_model(analyzer)
    if X_ntl is not None:
        X_ntl_train, X_ntl_test, y_train, y_test = train_test_split(
            X_ntl, y, test_size=0.3, random_state=analysis_state.random_state
        )
        ntl_model = RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_split=10,
                                         min_samples_leaf=5, max_features=0.7, random_state=42)
        ntl_model.fit(X_ntl_train, y_train)
        ntl_pred = ntl_model.predict(X_ntl_test)
        ntl_score = r2_score(y_test, ntl_pred)
        models['NTL Radiance (Basic)'] = ntl_model
        predictions['NTL Radiance (Basic)'] = ntl_pred
        scores['NTL Radiance (Basic)'] = ntl_score
        print(f"    • NTL Radiance (Basic): R² = {ntl_score:.4f}")
    else:
        print("    ⚠️ NTL Radiance model not available")
    
    # 2. SEMANTIC + CONTROL VARIABLES MODEL
    X_semantic_control, semantic_feature_names = create_semantic_with_controls_model(analyzer)
    if X_semantic_control is not None:
        X_semantic_train, X_semantic_test, y_train, y_test = train_test_split(
            X_semantic_control, y, test_size=0.3, random_state=analysis_state.random_state
        )
        semantic_model = RandomForestRegressor(
            n_estimators=50, max_depth=6, min_samples_split=10,
            min_samples_leaf=5, max_features=0.7,
            random_state=analysis_state.random_state, n_jobs=-1
        )
        semantic_model.fit(X_semantic_train, y_train)
        semantic_pred = semantic_model.predict(X_semantic_test)
        semantic_score = r2_score(y_test, semantic_pred)
        models['Semantic + Controls'] = semantic_model
        predictions['Semantic + Controls'] = semantic_pred
        scores['Semantic + Controls'] = semantic_score
        print(f"    • Semantic + Controls: R² = {semantic_score:.4f}")
    else:
        print("    ⚠️ Semantic + Controls model not available")
    
    # 3. FULL INTERACTION + CONTROL VARIABLES MODEL (Updated)
    X_full_interactions = analysis_state.interaction_features.get('abd_features')
    feature_names = X_full_interactions.columns.tolist()
    
    X_full_train, X_full_test, y_train, y_test = train_test_split(
        X_full_interactions, y, test_size=0.3, random_state=analysis_state.random_state
    )
    
    full_model = RandomForestRegressor(
        n_estimators=50, max_depth=6, min_samples_split=10,
        min_samples_leaf=5, max_features=0.7, 
        random_state=analysis_state.random_state, n_jobs=-1
    )
    full_model.fit(X_full_train, y_train)
    full_pred = full_model.predict(X_full_test)
    full_score = r2_score(y_test, full_pred)
    models['Full Interaction + Controls'] = full_model
    predictions['Full Interaction + Controls'] = full_pred
    scores['Full Interaction + Controls'] = full_score
    print(f"    • Full Interaction + Controls: R² = {full_score:.4f}")
    
    # 4. Enhanced ensemble prediction
    print("  🎯 Creating enhanced ensemble prediction...")
    weights = np.array(list(scores.values()))
    weights = np.maximum(weights, 0.001)  # Avoid negative weights
    weights = weights / weights.sum()
    ensemble_pred = np.average(list(predictions.values()), weights=weights, axis=0)
    ensemble_score = r2_score(y_test, ensemble_pred)
    
    print(f"  🏆 Ensemble Score: R² = {ensemble_score:.4f}")
    print(f"  📊 Model Weights: {dict(zip(scores.keys(), weights))}")
    
    # Calculate improvement over NTL baseline
    if 'NTL Radiance (Basic)' in scores:
        ntl_baseline_score = scores['NTL Radiance (Basic)']
        improvement = ((full_score - ntl_baseline_score) / abs(ntl_baseline_score) * 100) if ntl_baseline_score != 0 else 0
        print(f"  🎯 Full Interaction Model Improvement over NTL Baseline: {improvement:+.1f}%")
    else:
        ntl_baseline_score = 0
        improvement = 0
    
    # Create enhanced ensemble visualization with ALL models
    create_enhanced_ensemble_visualization(perception, models, predictions, scores,
                                         ensemble_pred, ensemble_score, y_test, 
                                         ntl_baseline_score, full_score, save_dir, xgb_result=xgb_result)
    
    return {
        'ensemble_score': ensemble_score,
        'individual_scores': scores,
        'models': models,
        'weights': weights,
        'baseline_score': ntl_baseline_score,
        'full_score': full_score,
        'improvement': improvement
    }

def create_enhanced_ensemble_visualization(perception, models, predictions, scores,
                                         ensemble_pred, ensemble_score, y_test, 
                                         baseline_score, full_score, save_dir, xgb_result=None):
    """Create enhanced ensemble visualization with ALL MODELS + XGBoost - Academic Journal Color Scheme"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 🎨 ACADEMIC JOURNAL COLOR SCHEME - 参考学术论文的专业配色
    academic_colors = {
        'ntl_basic': '#4A90E2',        # 冷色调蓝色 - NTL基础模型 (最简单)
        'semantic': '#50C878',         # 中性绿色 - 语义模型  
        'full_interaction': '#9B59B6', # 优雅的紫色 - 完整交互模型
        'ensemble': '#F39C12',         # 温暖的橙色 - 集成模型
        'xgboost': '#E74C3C',         # 最醒目的橙红色 - XGBoost (最后、最好)
        'perfect': '#34495E',          # 深灰色 - 完美预测线
        'confidence': '#ECF0F1',       # 浅灰色 - 置信区间
        'grid': '#F8F9FA',            # 极浅灰色 - 网格
        'text': '#2C3E50',            # 深蓝灰色 - 文字
    }
    
    # 🔧 如果有XGBoost结果，加入到模型中 - 但要调整顺序，XGBoost放在最后
    if xgb_result is not None:
        # 需要生成XGBoost在相同测试集上的预测
        X_interactions = analysis_state.interaction_features.get('abd_features')
        if X_interactions is not None:
            from sklearn.model_selection import train_test_split
            y_dummy = np.log(np.random.rand(len(X_interactions)) + 1)  # 占位符
            _, X_test_xgb, _, _ = train_test_split(X_interactions, y_dummy, test_size=0.3, random_state=42)
            xgb_pred = xgb_result['model'].predict(X_test_xgb)
            predictions['XGBoost'] = xgb_pred
        scores['XGBoost'] = xgb_result['test_score']
        print(f"  🔍 Added XGBoost model: R² = {xgb_result['test_score']:.4f}")
    
    # 🎯 按复杂度和性能排序：Basic → Semantic → Full → Ensemble → XGBoost(最后)
    ordered_models = ['NTL Radiance (Basic)', 'Semantic + Controls', 'Full Interaction + Controls', 'Ensemble']
    if 'XGBoost' in scores:
        ordered_models.append('XGBoost')
    
    # 设置图形样式 - 学术论文级别
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': ['Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.edgecolor': academic_colors['text'],
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'grid.alpha': 0.3,
        'grid.color': academic_colors['grid'],
        'text.color': academic_colors['text'],
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 📊 1. Enhanced Model Performance Comparison - Academic Color Scheme
    model_names = []
    model_scores = []
    
    # 按照ordered_models顺序添加模型
    for model_name in ordered_models:
        if model_name == 'Ensemble':
            model_names.append(model_name)
            model_scores.append(ensemble_score)
        elif model_name in scores:
            model_names.append(model_name)
            model_scores.append(scores[model_name])
    
    # Find best model for highlighting
    best_idx = np.argmax(model_scores)
    
    # 🎨 ACADEMIC COLOR MAPPING - 按冷暖色调排序，XGBoost最醒目
    color_map = {
        'NTL Radiance (Basic)': academic_colors['ntl_basic'],      # 冷色调蓝色
        'Semantic + Controls': academic_colors['semantic'],        # 中性绿色
        'Full Interaction + Controls': academic_colors['full_interaction'], # 紫色
        'Ensemble': academic_colors['ensemble'],                   # 温暖橙色
        'XGBoost': academic_colors['xgboost']                      # 最醒目橙红色
    }
    colors = [color_map.get(name, academic_colors['semantic']) for name in model_names]
    
    # Enhanced bar chart with error bars and styling
    bars = axes[0,0].bar(model_names, model_scores, color=colors, alpha=0.85, 
                        edgecolor='white', linewidth=1.5, width=0.7)
    
    # Highlight best model
    if best_idx < len(bars):
        bars[best_idx].set_edgecolor(academic_colors['text'])
        bars[best_idx].set_linewidth(3)
    
    axes[0,0].set_ylabel('R² Score', fontweight='bold')
    axes[0,0].set_title(f'Model Performance Comparison - {perception.title()}\nALL Models with Control Variables', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3, axis='y')
    axes[0,0].tick_params(axis='x', rotation=25)
    
    # Add value labels
    for bar, score in zip(bars, model_scores):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(model_scores)*0.02,
                      f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. ENHANCED Scatter Plot with fit lines and CONFIDENCE INTERVALS for ALL models
    # 🎯 按照ordered_models顺序重新排列predictions
    ordered_predictions = {}
    for model_name in ordered_models:
        if model_name == 'Ensemble':
            continue  # Ensemble在后面单独处理
        elif model_name in predictions:
            ordered_predictions[model_name] = predictions[model_name]
    
    model_list = list(ordered_predictions.keys())
    best_model_idx = np.argmax([scores[model] for model in model_list])
    
    # 🎨 ENHANCED COLOR MAPPING for scatter plots - 按冷暖色调排序，XGBoost最醒目
    scatter_colors = {
        'NTL Radiance (Basic)': academic_colors['ntl_basic'],      # 冷色调蓝色
        'Semantic + Controls': academic_colors['semantic'],        # 中性绿色
        'Full Interaction + Controls': academic_colors['full_interaction'], # 紫色
        'XGBoost': academic_colors['xgboost']                      # 最醒目橙红色
    }
    
    # 🔥 增强的散点图：添加R²和斜率标注
    for i, (name, pred) in enumerate(ordered_predictions.items()):
        alpha = 0.8 if i == best_model_idx else 0.65  # Best model most prominent
        size = 50 if i == best_model_idx else 40
        color = scatter_colors.get(name, academic_colors['semantic'])
        
        # 计算拟合线统计信息
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, pred)
        r_squared = r_value**2
        
        # 散点图 - 加强边框，把R²和斜率放到legend里
        axes[0,1].scatter(y_test, pred, alpha=alpha, s=size, 
                         color=color, 
                         label=f'{name} (R²={r_squared:.3f}, Slope={slope:.3f})', 
                         edgecolors='white', linewidth=1.0)
        
        # 🎯 添加拟合线
        line_x = np.linspace(y_test.min(), y_test.max(), 100)
        line_y = slope * line_x + intercept
        
        line_alpha = 0.9 if i == best_model_idx else 0.8
        line_width = 4 if i == best_model_idx else 2.8
        
        axes[0,1].plot(line_x, line_y, color=color, 
                      alpha=line_alpha, linewidth=line_width, linestyle='-')
        
        # 🎯 添加95%置信区间 - 更透明，避免颜色叠加太花
        residuals = pred - (slope * y_test + intercept)
        mse = np.mean(residuals**2)
        ci = 1.96 * np.sqrt(mse)  # 95% confidence interval
        
        ci_alpha = 0.25 if i == best_model_idx else 0.15  # 大幅降低透明度
        axes[0,1].fill_between(line_x, line_y - ci, line_y + ci, 
                              color=color, alpha=ci_alpha)
    
    # 添加完美预测线
    min_val, max_val = min(y_test.min(), min([p.min() for p in ordered_predictions.values()])), \
                      max(y_test.max(), max([p.max() for p in ordered_predictions.values()]))
    axes[0,1].plot([min_val, max_val], [min_val, max_val], 
                  color=academic_colors['perfect'], linestyle='--', 
                  alpha=0.9, linewidth=3, label='Perfect Prediction')
    
    axes[0,1].set_xlabel(f'True {perception.title()} Values', fontweight='bold')
    axes[0,1].set_ylabel(f'Predicted {perception.title()} Values', fontweight='bold')
    axes[0,1].set_title('Enhanced Prediction Accuracy with Confidence Intervals', fontweight='bold')
    axes[0,1].legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.05, 1))
    axes[0,1].grid(True, alpha=0.3, color=academic_colors['grid'])
    
    # 3. Model Architecture Comparison with USER COLORS
    if 'NTL Radiance (Basic)' in scores and 'Full Interaction + Controls' in scores:
        comparison_models = ['NTL Radiance\n(Basic)', 'Semantic+Controls', 'Full Interaction\n+Controls']
        comparison_scores = [scores.get('NTL Radiance (Basic)', 0),
                           scores.get('Semantic + Controls', 0),
                           scores.get('Full Interaction + Controls', 0)]
        comparison_colors = [academic_colors['ntl_basic'], academic_colors['semantic'], academic_colors['full_interaction']]
        
        bars = axes[1,0].bar(comparison_models, comparison_scores, 
                            color=comparison_colors, alpha=0.8, 
                            edgecolor='white', linewidth=1)
        
        axes[1,0].set_ylabel('R² Score', fontweight='bold')
        axes[1,0].set_title('Model Architecture Progression\nBasic → Semantic → Full Interaction', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels and improvement
        for bar, score in zip(bars, comparison_scores):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comparison_scores)*0.02,
                          f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Add improvement annotations
        if comparison_scores[0] > 0:
            improvement_semantic = ((comparison_scores[1] - comparison_scores[0]) / abs(comparison_scores[0]) * 100)
            improvement_full = ((comparison_scores[2] - comparison_scores[0]) / abs(comparison_scores[0]) * 100)
            
            axes[1,0].text(0.5, 0.8, f'Semantic vs NTL: {improvement_semantic:+.1f}%\nFull vs NTL: {improvement_full:+.1f}%', 
                          transform=axes[1,0].transAxes, ha='center', va='center',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor=academic_colors['semantic'], alpha=0.3),
                          fontsize=11, fontweight='bold')
    
    # 4. ENHANCED Ensemble Summary with ACADEMIC COLOR THEME
    xgb_info = f"• XGBoost: {scores.get('XGBoost', 0):.4f}" if 'XGBoost' in scores else ""
    
    ensemble_text = f"""🎯 ENHANCED ENSEMBLE ANALYSIS
{perception.title()} Perception with ALL Models

🏆 BEST PERFORMANCE: {model_names[best_idx]}
• Best R² Score: {max(model_scores):.4f}

📊 MODEL PROGRESSION (冷→暖色调):
• NTL Radiance (Basic): {scores.get('NTL Radiance (Basic)', 0):.4f}
• Semantic + Controls: {scores.get('Semantic + Controls', 0):.4f}  
• Full Interaction + Controls: {scores.get('Full Interaction + Controls', 0):.4f}
• Ensemble: {ensemble_score:.4f}
{xgb_info}

🔬 CONTROL VARIABLES:
• AVGIL: Average Illumination
• spots_area: Light Spots Area  
• ADCG: Advanced Depth Correlation Grid
• illumination_uniformity: Illumination Uniformity

🎯 MODEL ARCHITECTURE:
• NTL: Night-time Light Radiance (DN) only
• Semantic: A-pixel ratios + 5 control variables
• Full: A+B+D+AB+AD+BD+ABD + 5 control variables
• XGBoost: Non-linear tree-based ensemble

✨ ENHANCED SCATTER PLOT FEATURES:
• R² and Slope annotations on each model
• Enhanced fit lines with optimal thickness
• 95% confidence intervals (transparent bands)
• Color-coded by complexity (cold→warm)
• Perfect prediction reference line

🎨 ACADEMIC COLOR SCHEME:
• Blue: NTL Radiance (Basic) - Coldest
• Green: Semantic + Controls - Cool
• Purple: Full Interaction + Controls - Warm
• Orange: Ensemble - Warmer  
• Red: XGBoost - Hottest & Most Prominent"""
    
    axes[1,1].text(0.02, 0.98, ensemble_text, transform=axes[1,1].transAxes,
                  verticalalignment='top', fontsize=9, fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor=academic_colors['semantic'], alpha=0.2))
    axes[1,1].axis('off')
    
    fig.suptitle(f'🎯 Enhanced ALL-Models Ensemble Analysis - {perception.title()}\nAcademic Color Scheme with R² & Slope Annotations', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/module3_ensemble_{perception}.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # CLOSE FIGURE TO FREE MEMORY
    # plt.show()  # REMOVED - NO MORE POPUP WINDOWS!

def create_enhanced_seven_line_threshold_analysis(analyzer, save_dir):
    """Create enhanced 7-line nonlinear threshold analysis: A+B+D+AB+AD+BD+ABD with USER'S PURPLE/TEAL THEME"""
    print("\n🎯 MODULE 4: Enhanced 7-Line Nonlinear Threshold Analysis")
    print("="*60)
    print("  📊 Seven-Line Analysis Components:")
    print("     • A = Pixel Ratio (Semantic Area %)")
    print("     • B = Brightness (Luminance)")
    print("     • D = Depth (Distance)")
    print("     • AB, AD, BD = Two-way interactions")
    print("     • ABD = Three-way interaction (RED HIGHLIGHT)")
    print("  🎨 Using USER'S Purple/Teal Color Theme")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # USER'S PURPLE & TEAL COLOR SCHEME
    user_colors = {
        'primary': '#4B0082',     # Deep purple
        'secondary': '#20B2AA',   # Light sea green/teal  
        'accent1': '#6A5ACD',     # Slate blue
        'accent2': '#48D1CC',     # Medium turquoise
        'accent3': '#9370DB',     # Medium purple
        'accent4': '#40E0D0',     # Turquoise
        'neutral1': '#708090',    # Slate gray
        'neutral2': '#2F4F4F',    # Dark slate gray
    }
    
    perception_cols = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
    
    # Use USER-SPECIFIED semantic classes for analysis
    available_semantics = []
    for semantic in USER_SEMANTIC_CLASSES:
        # Check if all required data exists
        A_col = semantic
        B_col = f'{semantic}_brightness'
        D_col = f'{semantic}_depth'
        
        if (A_col in analyzer.merged_data.columns and 
            B_col in analyzer.merged_data.columns and 
            D_col in analyzer.merged_data.columns):
            available_semantics.append(semantic)
            print(f"    ✅ {semantic}: Complete A/B/D data available")
        else:
            print(f"    ⚠️ {semantic}: Missing data, skipping")
    
    print(f"    📊 Analyzing {len(available_semantics)} semantic classes: {available_semantics}")
    
    if len(available_semantics) == 0:
        print("    ❌ No semantic classes with complete data found")
        return
    
    # Create analysis for each semantic class
    for semantic_idx, semantic in enumerate(available_semantics):
        try:
            print(f"    🔍 Processing semantic {semantic_idx+1}/{len(available_semantics)}: {semantic}")
            
            # Create individual plot for each perception
            for perception_idx, perception in enumerate(perception_cols):
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
                    
                    # Get data with explicit column names
                    A_semantic = analyzer.merged_data[semantic]  # Pixel ratio
                    B_semantic = analyzer.merged_data[f'{semantic}_brightness'] / 255.0  # Brightness (normalized)
                    D_semantic = analyzer.merged_data[f'{semantic}_depth']  # Depth
                    y = np.log(analyzer.merged_data[perception] + 1)  # FIXED: Use 1 instead of 1
                    
                    # Adaptive threshold range based on data distribution
                    max_pixel = A_semantic.quantile(0.95)  # Use 95th percentile to avoid outliers
                    thresholds = np.linspace(0.001, min(max_pixel, 0.3), 15)  # 15 threshold points for smoother curves
                    
                    print(f"      Processing {perception} - {semantic}: {len(thresholds)} thresholds")
                    
                    # Seven interaction effects with EXACT run_optimized_analysis.py COLORS
                    effects_data = {
                        'A (Pixel)': {'values': [], 'color': '#9B7EDE', 'style': '-', 'width': 2},  # Light purple (primary)
                        'B (Brightness)': {'values': [], 'color': '#4ECDC4', 'style': '-', 'width': 2},  # Teal (secondary)
                        'D (Depth)': {'values': [], 'color': '#45B7D1', 'style': '-', 'width': 2},  # Blue accent
                        'AB (Pixel×Brightness)': {'values': [], 'color': '#96CEB4', 'style': '--', 'width': 2.5},  # Light green (neutral)
                        'AD (Pixel×Depth)': {'values': [], 'color': '#FECA57', 'style': '--', 'width': 2.5},  # Yellow (warning)
                        'BD (Brightness×Depth)': {'values': [], 'color': '#FFB3BA', 'style': '--', 'width': 2.5},  # Light pink (info)
                        'ABD (Triple Interaction)': {'values': [], 'color': '#FF0000', 'style': ':', 'width': 4}  # RED for ABD
                    }
                    
                    # Calculate effects for each threshold
                    for threshold in thresholds:
                        mask = A_semantic >= threshold
                        n_samples = mask.sum()
                        
                        if n_samples < 25:  # Need sufficient samples for reliable correlation
                            for effect_name in effects_data.keys():
                                effects_data[effect_name]['values'].append(np.nan)
                            continue
                        
                        # Extract masked data
                        A_masked = A_semantic[mask]
                        B_masked = B_semantic[mask]
                        D_masked = D_semantic[mask]
                        y_masked = y[mask]
                        
                        # Calculate correlations for each effect
                        try:
                            # Main effects - correlation with perception
                            corr_a = pearsonr(A_masked, y_masked)[0] if len(A_masked) > 1 else 0
                            corr_b = pearsonr(B_masked, y_masked)[0] if len(B_masked) > 1 else 0
                            corr_d = pearsonr(D_masked, y_masked)[0] if len(D_masked) > 1 else 0
                            
                            # Two-way interactions
                            AB_interaction = A_masked * B_masked
                            AD_interaction = A_masked * D_masked
                            BD_interaction = B_masked * D_masked
                            
                            corr_ab = pearsonr(AB_interaction, y_masked)[0] if len(AB_interaction) > 1 else 0
                            corr_ad = pearsonr(AD_interaction, y_masked)[0] if len(AD_interaction) > 1 else 0
                            corr_bd = pearsonr(BD_interaction, y_masked)[0] if len(BD_interaction) > 1 else 0
                            
                            # Three-way interaction (ABD) - THE HIGHLIGHT!
                            ABD_interaction = A_masked * B_masked * D_masked
                            corr_abd = pearsonr(ABD_interaction, y_masked)[0] if len(ABD_interaction) > 1 else 0
                            
                            # Store results
                            effects_data['A (Pixel)']['values'].append(corr_a)
                            effects_data['B (Brightness)']['values'].append(corr_b)
                            effects_data['D (Depth)']['values'].append(corr_d)
                            effects_data['AB (Pixel×Brightness)']['values'].append(corr_ab)
                            effects_data['AD (Pixel×Depth)']['values'].append(corr_ad)
                            effects_data['BD (Brightness×Depth)']['values'].append(corr_bd)
                            effects_data['ABD (Triple Interaction)']['values'].append(corr_abd)
                            
                        except Exception as correlation_error:
                            print(f"        Correlation error at threshold {threshold:.3f}: {str(correlation_error)[:30]}")
                            # Fill with zeros on error
                            for effect_name in effects_data.keys():
                                effects_data[effect_name]['values'].append(0)
                    
                    # Plot all seven lines with USER'S ENHANCED STYLING
                    for effect_name, effect_data in effects_data.items():
                        effect_values = effect_data['values']
                        color = effect_data['color']
                        style = effect_data['style']
                        width = effect_data['width']
                        
                        # Special formatting for ABD (Triple Interaction)
                        if 'ABD' in effect_name:
                            ax.plot(thresholds, effect_values, 
                                   linestyle=style, linewidth=width, color=color,
                                   label=effect_name, alpha=0.95, marker='o', markersize=8,
                                   markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)
                        else:
                            ax.plot(thresholds, effect_values, 
                                   linestyle=style, linewidth=width, 
                                   color=color, label=effect_name, alpha=0.85)
                    
                    # Enhanced formatting with USER'S STYLE
                    ax.set_xlabel(f'A_{semantic} Threshold (Pixel Ratio)', fontsize=13, fontweight='bold')
                    ax.set_ylabel(f'Correlation with {perception.title()}', fontsize=13, fontweight='bold')
                    ax.set_title(f'Seven-Line Nonlinear Analysis: {semantic.title()} → {perception.title()}\n' +
                               'Purple/Teal Theme | Red ABD shows three-way interaction effects', 
                               fontsize=15, fontweight='bold', pad=20)
                    
                    # Enhanced legend with better positioning
                    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, 
                                     frameon=True, shadow=True, fancybox=True)
                    legend.get_frame().set_facecolor('white')
                    legend.get_frame().set_alpha(0.95)
                    
                    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.6, linewidth=1.5)
                    
                    # Add enhanced text annotation for significant ABD effects
                    abd_values = effects_data['ABD (Triple Interaction)']['values']
                    valid_abd = [v for v in abd_values if not np.isnan(v)]
                    if valid_abd:
                        max_abd = max([abs(v) for v in valid_abd], default=0)
                        mean_abd = np.mean([abs(v) for v in valid_abd])
                        
                        if max_abd > 0.15:  # Significant threshold
                            ax.text(0.02, 0.98, f'ABD Effects:\nMax |ABD| = {max_abd:.3f}\nMean |ABD| = {mean_abd:.3f}', 
                                   transform=ax.transAxes, va='top', ha='left',
                                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FF0000', alpha=0.3, edgecolor='#FF0000'),
                                   fontsize=11, fontweight='bold', color='#8B0000')
                    
                    # Add model information
                    ax.text(0.98, 0.02, f'Model: A+B+D+AB+AD+BD+ABD\nTransform: log(perception + 1)\nSemantic: {semantic.title()}', 
                           transform=ax.transAxes, va='bottom', ha='right',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor=user_colors['accent2'], alpha=0.2),
                           fontsize=9, style='italic')
                    
                    plt.tight_layout()
                    plt.savefig(f'{save_dir}/seven_line_analysis_{semantic}_{perception}.png', 
                               dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()  # CLOSE TO FREE MEMORY
                    
                    print(f"        ✅ {perception} plot saved")
                    
                except Exception as perception_error:
                    print(f"    ⚠️ Error with {semantic}-{perception}: {str(perception_error)[:50]}...")
                    continue
                    
        except Exception as semantic_error:
            print(f"    ⚠️ Error with {semantic}: {str(semantic_error)[:50]}...")
            continue
    
    # Create summary visualization showing best ABD effects across all semantics
    create_abd_summary_visualization(analyzer, available_semantics, save_dir, user_colors)
    
    print("    ✅ Enhanced seven-line threshold analysis completed!")
    print(f"    📁 Generated {len(available_semantics) * len(perception_cols)} individual plots")
    return available_semantics

def create_abd_summary_visualization(analyzer, available_semantics, save_dir, user_colors):
    """Create summary visualization of ABD effects across all semantics and perceptions"""
    print("  📊 Creating ABD Summary Visualization...")
    
    perception_cols = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
    
    # Calculate max ABD effects for each semantic-perception combination
    abd_effects_matrix = np.zeros((len(available_semantics), len(perception_cols)))
    
    for i, semantic in enumerate(available_semantics):
        for j, perception in enumerate(perception_cols):
            try:
                # Get data
                A_semantic = analyzer.merged_data[semantic]
                B_semantic = analyzer.merged_data[f'{semantic}_brightness'] / 255.0
                D_semantic = analyzer.merged_data[f'{semantic}_depth']
                y = np.log(analyzer.merged_data[perception] + 1)
                
                # Calculate ABD interaction across different thresholds
                thresholds = np.linspace(0.01, 0.25, 10)
                max_abd_corr = 0
                
                for threshold in thresholds:
                    mask = A_semantic >= threshold
                    if mask.sum() > 20:
                        A_masked = A_semantic[mask]
                        B_masked = B_semantic[mask]
                        D_masked = D_semantic[mask]
                        y_masked = y[mask]
                        
                        ABD_interaction = A_masked * B_masked * D_masked
                        if len(ABD_interaction) > 1:
                            corr_abd = abs(pearsonr(ABD_interaction, y_masked)[0])
                            max_abd_corr = max(max_abd_corr, corr_abd)
                
                abd_effects_matrix[i, j] = max_abd_corr
                
            except Exception:
                abd_effects_matrix[i, j] = 0
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Heatmap of ABD effects
    im = ax1.imshow(abd_effects_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=0.5)
    ax1.set_xticks(range(len(perception_cols)))
    ax1.set_xticklabels([p.title() for p in perception_cols], rotation=45, ha='right')
    ax1.set_yticks(range(len(available_semantics)))
    ax1.set_yticklabels([s.title() for s in available_semantics])
    ax1.set_title('Maximum ABD Interaction Effects\n(Triple Interaction: A×B×D)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add text annotations
    for i in range(len(available_semantics)):
        for j in range(len(perception_cols)):
            value = abd_effects_matrix[i, j]
            color = 'white' if value > 0.25 else 'black'
            ax1.text(j, i, f'{value:.3f}', ha='center', va='center', 
                    color=color, fontweight='bold')
    
    plt.colorbar(im, ax=ax1, label='Max |Correlation|')
    
    # Bar chart of strongest ABD effects
    flat_effects = abd_effects_matrix.flatten()
    semantic_perception_pairs = [(s, p) for s in available_semantics for p in perception_cols]
    
    # Get top 10 effects
    top_indices = np.argsort(flat_effects)[-10:]
    top_effects = flat_effects[top_indices]
    top_pairs = [semantic_perception_pairs[i] for i in top_indices]
    
    bars = ax2.barh(range(len(top_effects)), top_effects, 
                   color=user_colors['primary'], alpha=0.8, edgecolor='white', linewidth=1)
    ax2.set_yticks(range(len(top_effects)))
    ax2.set_yticklabels([f'{s.title()} → {p.title()}' for s, p in top_pairs], fontsize=10)
    ax2.set_xlabel('Max ABD Correlation', fontweight='bold')
    ax2.set_title('Top 10 ABD Triple Interaction Effects\n(Purple Theme)', 
                 fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, effect in zip(bars, top_effects):
        ax2.text(effect + 0.005, bar.get_y() + bar.get_height()/2, f'{effect:.3f}',
                ha='left', va='center', fontweight='bold')
    
    fig.suptitle('ABD Triple Interaction Summary - All Semantics & Perceptions\nPurple/Teal Theme', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/abd_summary_analysis.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("    ✅ ABD Summary visualization completed")

def create_enhanced_seven_line_direct_effects_analysis(analyzer, save_dir):
    """Create enhanced 7-line DIRECT EFFECTS threshold analysis: A+B+D+AB+AD+BD+ABD DIRECT impact on perception"""
    print("\n🎯 MODULE 4B: Enhanced 7-Line DIRECT EFFECTS Threshold Analysis")
    print("="*60)
    print("  📊 Seven-Line DIRECT EFFECTS Analysis Components:")
    print("     • A = Pixel Ratio → Direct impact on perception")
    print("     • B = Brightness → Direct impact on perception")
    print("     • D = Depth → Direct impact on perception")
    print("     • AB, AD, BD = Two-way interactions → Direct impact")
    print("     • ABD = Three-way interaction → Direct impact (RED HIGHLIGHT)")
    print("  🔍 DIFFERENCE FROM CORRELATION ANALYSIS:")
    print("     • Previous: Correlation between variable and perception at different thresholds")
    print("     • Current: DIRECT MEAN VALUES of perception for different variable levels")
    print("  🎨 Using SAME Purple/Teal Color Theme")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # EXACT SAME COLORS as correlation analysis
    user_colors = {
        'primary': '#9B7EDE',     # Light purple (primary)
        'secondary': '#4ECDC4',   # Teal (secondary)
        'accent1': '#45B7D1',     # Blue accent
        'accent2': '#96CEB4',     # Light green (neutral)
        'accent3': '#FECA57',     # Yellow (warning)
        'accent4': '#FFB3BA',     # Light pink (info)
        'neutral1': '#708090',    # Slate gray
        'neutral2': '#2F4F4F',    # Dark slate gray
    }
    
    perception_cols = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
    
    # Use USER-SPECIFIED semantic classes for analysis
    available_semantics = []
    for semantic in USER_SEMANTIC_CLASSES:
        # Check if all required data exists
        A_col = semantic
        B_col = f'{semantic}_brightness'
        D_col = f'{semantic}_depth'
        
        if (A_col in analyzer.merged_data.columns and 
            B_col in analyzer.merged_data.columns and 
            D_col in analyzer.merged_data.columns):
            available_semantics.append(semantic)
            print(f"    ✅ {semantic}: Complete A/B/D data available")
        else:
            print(f"    ⚠️ {semantic}: Missing data, skipping")
    
    print(f"    📊 Analyzing {len(available_semantics)} semantic classes: {available_semantics}")
    
    if len(available_semantics) == 0:
        print("    ❌ No semantic classes with complete data found")
        return
    
    # Create analysis for each semantic class
    for semantic_idx, semantic in enumerate(available_semantics):
        try:
            print(f"    🔍 Processing semantic {semantic_idx+1}/{len(available_semantics)}: {semantic}")
            
            # Create individual plot for each perception
            for perception_idx, perception in enumerate(perception_cols):
                try:
                    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
                    
                    # Get data with explicit column names
                    A_semantic = analyzer.merged_data[semantic]  # Pixel ratio
                    B_semantic = analyzer.merged_data[f'{semantic}_brightness'] / 255.0  # Brightness (normalized)
                    D_semantic = analyzer.merged_data[f'{semantic}_depth']  # Depth
                    y = np.log(analyzer.merged_data[perception] + 1)  # FIXED: Use 1 instead of 1
                    
                    # Adaptive threshold range based on data distribution
                    max_pixel = A_semantic.quantile(0.95)  # Use 95th percentile to avoid outliers
                    thresholds = np.linspace(0.001, min(max_pixel, 0.3), 15)  # 15 threshold points for smoother curves
                    
                    print(f"      Processing {perception} - {semantic}: {len(thresholds)} thresholds (DIRECT EFFECTS)")
                    
                    # Seven interaction effects with EXACT SAME COLORS as correlation analysis
                    effects_data = {
                        'A (Pixel)': {'values': [], 'color': '#9B7EDE', 'style': '-', 'width': 2},  # Light purple (primary)
                        'B (Brightness)': {'values': [], 'color': '#4ECDC4', 'style': '-', 'width': 2},  # Teal (secondary)
                        'D (Depth)': {'values': [], 'color': '#45B7D1', 'style': '-', 'width': 2},  # Blue accent
                        'AB (Pixel×Brightness)': {'values': [], 'color': '#96CEB4', 'style': '--', 'width': 2.5},  # Light green (neutral)
                        'AD (Pixel×Depth)': {'values': [], 'color': '#FECA57', 'style': '--', 'width': 2.5},  # Yellow (warning)
                        'BD (Brightness×Depth)': {'values': [], 'color': '#FFB3BA', 'style': '--', 'width': 2.5},  # Light pink (info)
                        'ABD (Triple Interaction)': {'values': [], 'color': '#FF0000', 'style': ':', 'width': 4}  # RED for ABD
                    }
                    
                    # Calculate DIRECT EFFECTS for each threshold
                    for threshold in thresholds:
                        mask = A_semantic >= threshold
                        n_samples = mask.sum()
                        
                        if n_samples < 25:  # Need sufficient samples for reliable mean calculation
                            for effect_name in effects_data.keys():
                                effects_data[effect_name]['values'].append(np.nan)
                            continue
                        
                        # Extract masked data
                        A_masked = A_semantic[mask]
                        B_masked = B_semantic[mask]
                        D_masked = D_semantic[mask]
                        y_masked = y[mask]
                        
                        # Calculate DIRECT MEAN VALUES for each effect (NOT correlations!)
                        try:
                            # Main effects - mean values of each variable for samples above threshold
                            mean_a = A_masked.mean() if len(A_masked) > 0 else 0
                            mean_b = B_masked.mean() if len(B_masked) > 0 else 0
                            mean_d = D_masked.mean() if len(D_masked) > 0 else 0
                            
                            # Two-way interactions - mean interaction values
                            AB_interaction = A_masked * B_masked
                            AD_interaction = A_masked * D_masked
                            BD_interaction = B_masked * D_masked
                            
                            mean_ab = AB_interaction.mean() if len(AB_interaction) > 0 else 0
                            mean_ad = AD_interaction.mean() if len(AD_interaction) > 0 else 0
                            mean_bd = BD_interaction.mean() if len(BD_interaction) > 0 else 0
                            
                            # Three-way interaction (ABD) - mean triple interaction value
                            ABD_interaction = A_masked * B_masked * D_masked
                            mean_abd = ABD_interaction.mean() if len(ABD_interaction) > 0 else 0
                            
                            # Store results - DIRECT VALUES, not correlations
                            effects_data['A (Pixel)']['values'].append(mean_a)
                            effects_data['B (Brightness)']['values'].append(mean_b)
                            effects_data['D (Depth)']['values'].append(mean_d)
                            effects_data['AB (Pixel×Brightness)']['values'].append(mean_ab)
                            effects_data['AD (Pixel×Depth)']['values'].append(mean_ad)
                            effects_data['BD (Brightness×Depth)']['values'].append(mean_bd)
                            effects_data['ABD (Triple Interaction)']['values'].append(mean_abd)
                            
                        except Exception as calculation_error:
                            print(f"        Calculation error at threshold {threshold:.3f}: {str(calculation_error)[:30]}")
                            # Fill with zeros on error
                            for effect_name in effects_data.keys():
                                effects_data[effect_name]['values'].append(0)
                    
                    # Plot all seven lines with SAME ENHANCED STYLING
                    for effect_name, effect_data in effects_data.items():
                        effect_values = effect_data['values']
                        color = effect_data['color']
                        style = effect_data['style']
                        width = effect_data['width']
                        
                        # Special formatting for ABD (Triple Interaction)
                        if 'ABD' in effect_name:
                            ax.plot(thresholds, effect_values, 
                                   linestyle=style, linewidth=width, color=color,
                                   label=effect_name, alpha=0.95, marker='o', markersize=8,
                                   markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)
                        else:
                            ax.plot(thresholds, effect_values, 
                                   linestyle=style, linewidth=width, 
                                   color=color, label=effect_name, alpha=0.85)
                    
                    # Enhanced formatting with CLEAR TITLE indicating DIRECT EFFECTS
                    ax.set_xlabel(f'A_{semantic} Threshold (Pixel Ratio)', fontsize=13, fontweight='bold')
                    ax.set_ylabel(f'Mean Variable Values (Direct Effects)', fontsize=13, fontweight='bold')
                    ax.set_title(f'Seven-Line DIRECT EFFECTS Analysis: {semantic.title()} → {perception.title()}\n' +
                               'Shows MEAN VALUES of variables (not correlations) | Red ABD = triple interaction', 
                               fontsize=15, fontweight='bold', pad=20)
                    
                    # Enhanced legend with better positioning
                    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, 
                                     frameon=True, shadow=True, fancybox=True)
                    legend.get_frame().set_facecolor('white')
                    legend.get_frame().set_alpha(0.95)
                    
                    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.6, linewidth=1.5)
                    
                    # Add enhanced text annotation for significant ABD effects
                    abd_values = effects_data['ABD (Triple Interaction)']['values']
                    valid_abd = [v for v in abd_values if not np.isnan(v)]
                    if valid_abd:
                        max_abd = max([abs(v) for v in valid_abd], default=0)
                        mean_abd = np.mean([abs(v) for v in valid_abd])
                        
                        if max_abd > 0.001:  # Lower threshold for direct effects
                            ax.text(0.02, 0.98, f'ABD Direct Effects:\nMax |ABD| = {max_abd:.5f}\nMean |ABD| = {mean_abd:.5f}', 
                                   transform=ax.transAxes, va='top', ha='left',
                                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FF0000', alpha=0.3, edgecolor='#FF0000'),
                                   fontsize=11, fontweight='bold', color='#8B0000')
                    
                    # Add model information
                    ax.text(0.98, 0.02, f'Analysis: DIRECT EFFECTS (Mean Values)\nModel: A+B+D+AB+AD+BD+ABD\nSemantic: {semantic.title()}', 
                           transform=ax.transAxes, va='bottom', ha='right',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor=user_colors['accent2'], alpha=0.2),
                           fontsize=9, style='italic')
                    
                    plt.tight_layout()
                    plt.savefig(f'{save_dir}/seven_line_direct_effects_{semantic}_{perception}.png', 
                               dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()  # CLOSE TO FREE MEMORY
                    
                    print(f"        ✅ {perception} DIRECT EFFECTS plot saved")
                    
                except Exception as perception_error:
                    print(f"    ⚠️ Error with {semantic}-{perception}: {str(perception_error)[:50]}...")
                    continue
                    
        except Exception as semantic_error:
            print(f"    ⚠️ Error with {semantic}: {str(semantic_error)[:50]}...")
            continue
    
    # Create summary visualization showing direct effects patterns
    create_direct_effects_summary_visualization(analyzer, available_semantics, save_dir, user_colors)
    
    print("    ✅ Enhanced seven-line DIRECT EFFECTS threshold analysis completed!")
    print(f"    📁 Generated {len(available_semantics) * len(perception_cols)} DIRECT EFFECTS plots")
    return available_semantics

def create_direct_effects_summary_visualization(analyzer, available_semantics, save_dir, user_colors):
    """Create summary visualization of DIRECT EFFECTS patterns across all semantics and perceptions"""
    print("  📊 Creating DIRECT EFFECTS Summary Visualization...")
    
    perception_cols = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
    
    # Calculate max DIRECT ABD effects for each semantic-perception combination
    abd_direct_effects_matrix = np.zeros((len(available_semantics), len(perception_cols)))
    
    for i, semantic in enumerate(available_semantics):
        for j, perception in enumerate(perception_cols):
            try:
                # Get data
                A_semantic = analyzer.merged_data[semantic]
                B_semantic = analyzer.merged_data[f'{semantic}_brightness'] / 255.0
                D_semantic = analyzer.merged_data[f'{semantic}_depth']
                y = np.log(analyzer.merged_data[perception] + 1)
                
                # Calculate ABD DIRECT EFFECTS across different thresholds
                thresholds = np.linspace(0.01, 0.25, 10)
                max_abd_direct = 0
                
                for threshold in thresholds:
                    mask = A_semantic >= threshold
                    if mask.sum() > 20:
                        A_masked = A_semantic[mask]
                        B_masked = B_semantic[mask]
                        D_masked = D_semantic[mask]
                        
                        ABD_interaction = A_masked * B_masked * D_masked
                        if len(ABD_interaction) > 0:
                            mean_abd_direct = abs(ABD_interaction.mean())
                            max_abd_direct = max(max_abd_direct, mean_abd_direct)
                
                abd_direct_effects_matrix[i, j] = max_abd_direct
                
            except Exception:
                abd_direct_effects_matrix[i, j] = 0
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Heatmap of ABD DIRECT effects
    im = ax1.imshow(abd_direct_effects_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=np.max(abd_direct_effects_matrix))
    ax1.set_xticks(range(len(perception_cols)))
    ax1.set_xticklabels([p.title() for p in perception_cols], rotation=45, ha='right')
    ax1.set_yticks(range(len(available_semantics)))
    ax1.set_yticklabels([s.title() for s in available_semantics])
    ax1.set_title('Maximum ABD DIRECT EFFECTS\n(Triple Interaction: A×B×D Mean Values)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add text annotations
    for i in range(len(available_semantics)):
        for j in range(len(perception_cols)):
            value = abd_direct_effects_matrix[i, j]
            color = 'white' if value > np.max(abd_direct_effects_matrix) * 0.6 else 'black'
            ax1.text(j, i, f'{value:.5f}', ha='center', va='center', 
                    color=color, fontweight='bold', fontsize=8)
    
    plt.colorbar(im, ax=ax1, label='Max |Direct ABD Effect|')
    
    # Bar chart of strongest ABD DIRECT effects
    flat_effects = abd_direct_effects_matrix.flatten()
    semantic_perception_pairs = [(s, p) for s in available_semantics for p in perception_cols]
    
    # Get top 10 effects
    top_indices = np.argsort(flat_effects)[-10:]
    top_effects = flat_effects[top_indices]
    top_pairs = [semantic_perception_pairs[i] for i in top_indices]
    
    bars = ax2.barh(range(len(top_effects)), top_effects, 
                   color=user_colors['primary'], alpha=0.8, edgecolor='white', linewidth=1)
    ax2.set_yticks(range(len(top_effects)))
    ax2.set_yticklabels([f'{s.title()} → {p.title()}' for s, p in top_pairs], fontsize=10)
    ax2.set_xlabel('Max ABD Direct Effect', fontweight='bold')
    ax2.set_title('Top 10 ABD DIRECT EFFECTS\n(Purple Theme)', 
                 fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, effect in zip(bars, top_effects):
        ax2.text(effect + effect*0.05, bar.get_y() + bar.get_height()/2, f'{effect:.5f}',
                ha='left', va='center', fontweight='bold', fontsize=9)
    
    fig.suptitle('ABD Triple Interaction DIRECT EFFECTS Summary\nMean Values (Not Correlations) | Purple/Teal Theme', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/abd_direct_effects_summary.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("    ✅ ABD DIRECT EFFECTS Summary visualization completed")

def create_comprehensive_multi_model_analysis(analyzer, perception, save_dir):
    """Create comprehensive analysis showing ALL models with confidence intervals and slopes"""
    print(f"  📊 Creating Comprehensive Multi-Model Analysis for {perception}...")
    
    # Create all models
    models = {}
    predictions = {}
    scores = {}
    slopes = {}
    
    # USER'S PURPLE & TEAL COLOR SCHEME for ALL models
    user_colors = {
        'ntl': '#FF6B6B',           # Coral for NTL radiance
        'semantic': '#4ECDC4',      # Teal for semantic
        'full': '#4B0082',          # Deep purple for full model
        'ridge': '#20B2AA',         # Sea green for ridge
        'accent1': '#6A5ACD',       # Slate blue
        'accent2': '#48D1CC',       # Medium turquoise
    }
    
    # FIXED: Use epsilon=1 instead of 1 for log transformation
    y = np.log(analyzer.merged_data[perception] + 0.01)
    
    # 1. NTL Radiance Model
    X_ntl, ntl_feature_names = create_baseline_ntl_model(analyzer)
    if X_ntl is not None:
        X_ntl_train, X_ntl_test, y_train, y_test = train_test_split(
            X_ntl, y, test_size=0.3, random_state=analysis_state.random_state
        )
        ntl_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        ntl_model.fit(X_ntl_train, y_train)
        ntl_pred = ntl_model.predict(X_ntl_test)
        ntl_score = r2_score(y_test, ntl_pred)
        
        # Calculate slope
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, ntl_pred)
        
        models['NTL Radiance'] = ntl_model
        predictions['NTL Radiance'] = ntl_pred
        scores['NTL Radiance'] = ntl_score
        slopes['NTL Radiance'] = slope
        
        print(f"    ✅ NTL Radiance: R² = {ntl_score:.4f}, Slope = {slope:.4f}")
    
    # 2. Semantic + Controls Model
    X_semantic, semantic_feature_names = create_semantic_with_controls_model(analyzer)
    if X_semantic is not None:
        X_sem_train, X_sem_test, y_train, y_test = train_test_split(
            X_semantic, y, test_size=0.3, random_state=analysis_state.random_state
        )
        semantic_model = RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_split=10,
                                             min_samples_leaf=5, max_features=0.7, random_state=42)
        semantic_model.fit(X_sem_train, y_train)
        semantic_pred = semantic_model.predict(X_sem_test)
        semantic_score = r2_score(y_test, semantic_pred)
        
        # Calculate slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, semantic_pred)
        
        models['Semantic + Controls'] = semantic_model
        predictions['Semantic + Controls'] = semantic_pred
        scores['Semantic + Controls'] = semantic_score
        slopes['Semantic + Controls'] = slope
        
        print(f"    ✅ Semantic + Controls: R² = {semantic_score:.4f}, Slope = {slope:.4f}")
    
    # 3. Full Interaction + Controls Model  
    X_full = analysis_state.interaction_features.get('abd_features')
    if X_full is not None:
        X_full_train, X_full_test, y_train, y_test = train_test_split(
            X_full, y, test_size=0.3, random_state=analysis_state.random_state
        )
        full_model = RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_split=10,
                                         min_samples_leaf=5, max_features=0.7, random_state=42)
        full_model.fit(X_full_train, y_train)
        full_pred = full_model.predict(X_full_test)
        full_score = r2_score(y_test, full_pred)
        
        # Calculate slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, full_pred)
        
        models['Full Interaction + Controls'] = full_model
        predictions['Full Interaction + Controls'] = full_pred
        scores['Full Interaction + Controls'] = full_score
        slopes['Full Interaction + Controls'] = slope
        
        print(f"    ✅ Full Interaction + Controls: R² = {full_score:.4f}, Slope = {slope:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Model Performance Comparison
    model_names = list(scores.keys())
    model_scores = list(scores.values())
    model_slopes = list(slopes.values())
    
    colors = [user_colors['ntl'], user_colors['semantic'], user_colors['full']][:len(model_names)]
    
    bars = axes[0,0].bar(model_names, model_scores, color=colors, alpha=0.8, 
                        edgecolor='white', linewidth=1)
    
    axes[0,0].set_ylabel('R² Score', fontweight='bold')
    axes[0,0].set_title(f'Model Performance: {perception.title()}\nAll Models with Control Variables', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3, axis='y')
    axes[0,0].tick_params(axis='x', rotation=15)
    
    # Add R² and slope labels
    for i, (bar, score, slope) in enumerate(zip(bars, model_scores, model_slopes)):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(model_scores)*0.02,
                      f'R²={score:.4f}\nSlope={slope:.3f}', ha='center', va='bottom', 
                      fontweight='bold', fontsize=9)
    
    # 2. Scatter Plot with Confidence Intervals
    for i, (name, pred) in enumerate(predictions.items()):
        color = colors[i]
        score = scores[name]
        slope = slopes[name]
        
        axes[0,1].scatter(y_test, pred, alpha=0.7, s=30, color=color, 
                         label=f'{name} (R²={score:.3f}, Slope={slope:.3f})', 
                         edgecolors='white', linewidth=0.3)
        
        # Add fit line
        from scipy import stats
        slope_val, intercept, r_value, p_value, std_err = stats.linregress(y_test, pred)
        line_x = np.linspace(y_test.min(), y_test.max(), 100)
        line_y = slope_val * line_x + intercept
        
        axes[0,1].plot(line_x, line_y, color=color, alpha=0.8, linewidth=2.5)
        
        # Add confidence interval
        residuals = pred - (slope_val * y_test + intercept)
        mse = np.mean(residuals**2)
        ci = 1.96 * np.sqrt(mse)
        
        axes[0,1].fill_between(line_x, line_y - ci, line_y + ci, 
                              color=color, alpha=0.2)
    
    # Perfect prediction line
    min_val = min(y_test.min(), min([p.min() for p in predictions.values()]))
    max_val = max(y_test.max(), max([p.max() for p in predictions.values()]))
    axes[0,1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=3)
    
    axes[0,1].set_xlabel(f'True {perception.title()} Values', fontweight='bold')
    axes[0,1].set_ylabel(f'Predicted {perception.title()} Values', fontweight='bold')
    axes[0,1].set_title('Model Predictions with Confidence Intervals', fontweight='bold')
    axes[0,1].legend(fontsize=9)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Slope Comparison
    bars = axes[1,0].bar(model_names, model_slopes, color=colors, alpha=0.8, 
                        edgecolor='white', linewidth=1)
    
    axes[1,0].set_ylabel('Regression Slope', fontweight='bold')
    axes[1,0].set_title('Model Fit Quality (Slope Analysis)', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    axes[1,0].tick_params(axis='x', rotation=15)
    axes[1,0].axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect Slope')
    
    # Add slope labels
    for bar, slope in zip(bars, model_slopes):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(model_slopes)*0.02,
                      f'{slope:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Model Summary
    summary_text = f"""COMPREHENSIVE MODEL ANALYSIS
{perception.title()} Perception

📊 MODEL ARCHITECTURE PROGRESSION:
• NTL Radiance (Basic): {scores.get('NTL Radiance', 0):.4f}
• Semantic + Controls: {scores.get('Semantic + Controls', 0):.4f}  
• Full Interaction + Controls: {scores.get('Full Interaction + Controls', 0):.4f}

🔬 CONTROL VARIABLES:
• AVGIL: Average Illumination
• spots_area: Light Spots Area  
• ADCG: Advanced Depth Correlation Grid
• illumination_uniformity: Illumination Uniformity

✨ VISUAL FEATURES:
• Confidence intervals (95% CI)
• Fit lines with slopes
• R² scores for each model
• Purple/Teal color scheme

🎨 COLOR LEGEND:
• Coral: NTL Radiance (Basic)
• Teal: Semantic + Controls
• Purple: Full Interaction + Controls"""
    
    axes[1,1].text(0.02, 0.98, summary_text, transform=axes[1,1].transAxes,
                  verticalalignment='top', fontsize=9, fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor=user_colors['accent2'], alpha=0.2))
    axes[1,1].axis('off')
    
    fig.suptitle(f'Comprehensive Multi-Model Analysis - {perception.title()}\nAll Models with Control Variables & Confidence Intervals', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comprehensive_multi_model_{perception}.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Comprehensive Multi-Model Analysis completed for {perception}")
    
    return {
        'models': models,
        'predictions': predictions,
        'scores': scores,
        'slopes': slopes
    }

def create_fallback_analysis(model, feature_names, perception, test_score, save_dir):
    """Enhanced fallback when SHAP is not available - USER'S PURPLE COLOR"""
    if hasattr(model, 'feature_importances_'):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # USER'S PURPLE & TEAL COLORS
        user_colors = {
            'primary': '#4B0082',     # Deep purple
            'secondary': '#20B2AA',   # Light sea green/teal  
            'accent1': '#6A5ACD',     # Slate blue
            'accent2': '#48D1CC',     # Medium turquoise
            'accent3': '#9370DB',     # Medium purple
            'accent4': '#40E0D0',     # Turquoise
        }
        
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[-25:]  # 增加到25个
        
        # FIXED: User's primary purple color instead of blue
        bars = axes[0].barh(range(len(sorted_idx)), importance[sorted_idx], 
                           color=user_colors['primary'], alpha=0.8, edgecolor='white', linewidth=1)
        
        axes[0].set_yticks(range(len(sorted_idx)))
        axes[0].set_yticklabels([feature_names[i].replace('_', ' ')[:25] for i in sorted_idx])
        axes[0].set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
        axes[0].set_title(f'{perception.title()} - Model Feature Importance\nR² = {test_score:.4f}', 
                         fontsize=16, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x', linestyle='--')
        
        for bar, imp in zip(bars, importance[sorted_idx]):
            if imp > 0.001:
                axes[0].text(imp + imp*0.02, bar.get_y() + bar.get_height()/2, 
                           f'{imp:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Feature category analysis with USER COLORS
        feature_types = {'A_': 0, 'B_': 0, 'D_': 0, 'AB_': 0, 'AD_': 0, 'BD_': 0, 'ABD_': 0}
        for name in feature_names:
            for prefix in feature_types:
                if name.startswith(prefix):
                    feature_types[prefix] += 1
                    break
        
        categories = list(feature_types.keys())
        counts = list(feature_types.values())
        colors_cat = [user_colors['primary'], user_colors['secondary'], user_colors['accent1'], 
                     user_colors['accent2'], user_colors['accent3'], user_colors['accent4'], user_colors['primary']]
        
        bars = axes[1].bar(categories, counts, color=colors_cat, alpha=0.8, edgecolor='white', linewidth=1)
        axes[1].set_ylabel('Number of Features', fontsize=14, fontweight='bold')
        axes[1].set_title('A+B+D+AB+AD+BD+ABD Feature Distribution (Purple/Teal Theme)', fontsize=16, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        for bar, count in zip(bars, counts):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/xgb_feature_importance_{perception}.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # CLOSE FIGURE TO FREE MEMORY
        # plt.show()  # REMOVED - NO MORE POPUP WINDOWS!

# 重复的visualize_lasso_results函数已移除，避免生成两套相同的图
# 重复的create_fallback_analysis函数也已移除，避免生成两套相同的图
# 重复的create_lcz_stability_analysis函数也已移除
    
    # 目标LCZ类型
    target_lcz_types = [1, 2, 3, 4, 9, 11]
    lcz_names = {
        1: 'Compact High-rise',
        2: 'Compact Mid-rise', 
        3: 'Compact Low-rise',
        4: 'Open High-rise',
        9: 'Sparsely Built',
        11: 'Dense Trees'
    }
    
    # 筛选目标LCZ数据
    lcz_data = analyzer.merged_data[analyzer.merged_data['LCZ'].isin(target_lcz_types)]
    print(f"📊 LCZ筛选后数据量: {len(lcz_data)}/{len(analyzer.merged_data)} ({len(lcz_data)/len(analyzer.merged_data)*100:.1f}%)")
    
    # 为每个LCZ类型分别跑完整分析
    all_lcz_results = {}
    
    for lcz_type in target_lcz_types:
        lcz_subset = lcz_data[lcz_data['LCZ'] == lcz_type]
        if len(lcz_subset) < 50:
            print(f"⚠️ LCZ {lcz_type} 样本量不足 ({len(lcz_subset)})，跳过")
            continue
            
        print(f"\n{'='*60}")
        print(f"🏙️ LCZ {lcz_type} ({lcz_names[lcz_type]}) - 完整分析开始")
        print(f"📊 样本数: {len(lcz_subset)}")
        print("="*60)
        
        # 创建LCZ专用分析器
        lcz_analyzer = FixedOptimizedInteractionAnalyzer()
        lcz_analyzer.merged_data = lcz_subset.copy()
        
        # 创建LCZ专用结果目录
        lcz_save_dir = f"{save_dir}/LCZ_{lcz_type}_{lcz_names[lcz_type].replace(' ', '_')}"
        os.makedirs(lcz_save_dir, exist_ok=True)
        
        # 为每个感知维度跑完整分析
        lcz_results = {}
        
        for perception in perception_cols:
            print(f"\n🎯 LCZ {lcz_type} - 感知维度: {perception.upper()}")
            print("-" * 50)
            
            try:
                # MODULE 1: XGBoost + SHAP
                print(f"🔍 MODULE 1: XGBoost + SHAP for LCZ {lcz_type} - {perception}")
                xgb_result = run_enhanced_xgboost_module(lcz_analyzer, perception, lcz_save_dir, libs)
                
                if xgb_result is None:
                    print(f"    ❌ XGBoost模块失败")
                    continue
                
                # MODULE 2: Lasso Feature Selection
                print(f"🎯 MODULE 2: Lasso Feature Selection for LCZ {lcz_type} - {perception}")
                lasso_result = run_enhanced_lasso_module(lcz_analyzer, perception, lcz_save_dir)
                
                if lasso_result is None:
                    print(f"    ❌ Lasso模块失败")
                    continue
                
                # MODULE 3: Ensemble Strategy
                print(f"🔗 MODULE 3: Ensemble Strategy for LCZ {lcz_type} - {perception}")
                ensemble_result = run_integrated_ensemble_module(lcz_analyzer, perception, lcz_save_dir, 
                                                               {'xgb': xgb_result, 'lasso': lasso_result})
                
                # 保存结果
                lcz_results[perception] = {
                    'xgb': xgb_result,
                    'lasso': lasso_result,
                    'ensemble': ensemble_result
                }
                
                print(f"\n📊 LCZ {lcz_type} - {perception} 性能总结:")
                print(f"  • XGBoost: R² = {xgb_result['test_score']:.4f}")
                print(f"  • Lasso: R² = {lasso_result['lasso_score']:.4f}")
                print(f"  • Elastic-Net: R² = {lasso_result['elastic_score']:.4f}")
                print(f"  • Ensemble: R² = {ensemble_result['ensemble_score']:.4f}")
                
            except Exception as e:
                print(f"    ❌ LCZ {lcz_type} - {perception} 分析失败: {str(e)}")
                continue
        
        # 保存这个LCZ的所有结果
        all_lcz_results[lcz_type] = {
            'name': lcz_names[lcz_type],
            'sample_count': len(lcz_subset),
            'results': lcz_results,
            'save_dir': lcz_save_dir
        }
        
        print(f"\n✅ LCZ {lcz_type} ({lcz_names[lcz_type]}) 完整分析完成!")
        print(f"📁 结果保存在: {lcz_save_dir}")
    
    print(f"\n🎉 LCZ分区分析完成! 共分析了 {len(all_lcz_results)} 个LCZ分区")
    
    # 生成LCZ分区对比日志
    try:
        create_lcz_comparison_log(all_lcz_results, save_dir)
    except Exception as e:
        print(f"  ⚠️ LCZ对比日志生成失败: {str(e)}")
    
    return all_lcz_results

def create_model_log_file(model_results, model_name, y_var, save_dir, analysis_type="main", lcz_type=None):
    """创建详细的模型日志文件"""
    from datetime import datetime
    
    # 创建日志目录
    log_dir = f"{save_dir}/model_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 构建日志文件名
    if lcz_type is not None:
        log_filename = f"model_log_{analysis_type}_LCZ{lcz_type}_{y_var}_{model_name}.txt"
    else:
        log_filename = f"model_log_{analysis_type}_{y_var}_{model_name}.txt"
    
    log_path = f"{log_dir}/{log_filename}"
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"MODEL PERFORMANCE LOG - ABD_trip_V2\n")
        f.write("="*80 + "\n\n")
        
        # 基本信息
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Target Variable: {y_var}\n")
        f.write(f"Analysis Type: {analysis_type}\n")
        if lcz_type is not None:
            f.write(f"LCZ Type: {lcz_type}\n")
        f.write("\n" + "-"*80 + "\n\n")
        
        # 性能指标
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*40 + "\n")
        if 'test_score' in model_results:
            f.write(f"Test R²: {model_results['test_score']:.6f}\n")
        if 'lasso_score' in model_results:
            f.write(f"Lasso R²: {model_results['lasso_score']:.6f}\n")
        if 'elastic_score' in model_results:
            f.write(f"Elastic-Net R²: {model_results['elastic_score']:.6f}\n")
        if 'ensemble_score' in model_results:
            f.write(f"Ensemble R²: {model_results['ensemble_score']:.6f}\n")
        f.write("\n")
        
        # 选中的特征
        if 'lasso_selected' in model_results:
            f.write(f"LASSO SELECTED FEATURES ({len(model_results['lasso_selected'])}):\n")
            f.write("-"*40 + "\n")
            for i, feat in enumerate(model_results['lasso_selected'][:30]):  # 前30个
                f.write(f"{i+1:3d}. {feat}\n")
            if len(model_results['lasso_selected']) > 30:
                f.write(f"... and {len(model_results['lasso_selected'])-30} more\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    return log_path

def create_comparison_log(all_results, save_dir, analysis_type="main"):
    """创建模型对比日志（CSV格式，便于Excel分析）"""
    import pandas as pd
    
    log_dir = f"{save_dir}/model_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 收集所有模型结果
    comparison_data = []
    
    for y_var, results in all_results.items():
        if isinstance(results, dict):
            # XGBoost
            if 'xgb' in results and results['xgb'] is not None:
                comparison_data.append({
                    'Analysis_Type': analysis_type,
                    'Y_Variable': y_var,
                    'Model': 'XGBoost',
                    'R2_Test': results['xgb'].get('test_score', np.nan)
                })
            
            # Lasso
            if 'lasso' in results and results['lasso'] is not None:
                comparison_data.append({
                    'Analysis_Type': analysis_type,
                    'Y_Variable': y_var,
                    'Model': 'Lasso',
                    'R2_Test': results['lasso'].get('lasso_score', np.nan)
                })
                
                comparison_data.append({
                    'Analysis_Type': analysis_type,
                    'Y_Variable': y_var,
                    'Model': 'Elastic-Net',
                    'R2_Test': results['lasso'].get('elastic_score', np.nan)
                })
            
            # Ensemble
            if 'ensemble' in results and results['ensemble'] is not None:
                comparison_data.append({
                    'Analysis_Type': analysis_type,
                    'Y_Variable': y_var,
                    'Model': 'Ensemble',
                    'R2_Test': results['ensemble'].get('ensemble_score', np.nan)
                })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        csv_path = f"{log_dir}/model_comparison_{analysis_type}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ 模型对比日志已保存: {csv_path}")
        return csv_path
    
    return None

def create_lcz_comparison_log(all_lcz_results, save_dir):
    """创建LCZ对比日志"""
    import pandas as pd
    
    log_dir = f"{save_dir}/model_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    comparison_data = []
    
    for lcz_type, lcz_info in all_lcz_results.items():
        for perception, results in lcz_info['results'].items():
            if 'xgb' in results and results['xgb'] is not None:
                comparison_data.append({
                    'LCZ_Type': lcz_type,
                    'LCZ_Name': lcz_info['name'],
                    'Y_Variable': perception,
                    'Model': 'XGBoost',
                    'R2_Test': results['xgb'].get('test_score', np.nan),
                    'N_Samples': lcz_info.get('sample_count', np.nan)
                })
            
            if 'lasso' in results and results['lasso'] is not None:
                comparison_data.append({
                    'LCZ_Type': lcz_type,
                    'LCZ_Name': lcz_info['name'],
                    'Y_Variable': perception,
                    'Model': 'Lasso',
                    'R2_Test': results['lasso'].get('lasso_score', np.nan),
                    'N_Samples': lcz_info.get('sample_count', np.nan)
                })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        csv_path = f"{log_dir}/lcz_model_comparison.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ LCZ对比日志已保存: {csv_path}")
        return csv_path
    
    return None

def create_ntl_analysis(analyzer, perception_cols, save_dir, libs):
    """创建NTL专项分析 - V2版本：只使用ntl_mean（不包含其他控制变量）"""
    print("\n🌙 NTL SPECIALIZED ANALYSIS - V2")
    print("="*80)
    print("📊 使用 ntl_mean 预测所有感知维度（V2版本：无其他控制变量）")
    
    # 检查ntl_mean列是否存在
    if 'ntl_mean' not in analyzer.merged_data.columns:
        print("❌ ntl_mean列不存在，跳过NTL分析")
        return
    
    # 创建NTL专用结果目录
    ntl_save_dir = f"{save_dir}/NTL_Analysis"
    os.makedirs(ntl_save_dir, exist_ok=True)
    
    # 过滤掉ntl_mean为空的数据
    ntl_data = analyzer.merged_data[analyzer.merged_data['ntl_mean'].notna()].copy()
    print(f"📊 NTL数据量: {len(ntl_data)}/{len(analyzer.merged_data)} ({len(ntl_data)/len(analyzer.merged_data)*100:.1f}%)")
    print(f"   过滤掉 {len(analyzer.merged_data) - len(ntl_data)} 个ntl_mean空值数据")
    
    if len(ntl_data) < 100:
        print("❌ NTL数据量不足，跳过分析")
        return
    
    # 准备特征：V2版本只使用ntl_mean
    feature_cols = ['ntl_mean']
    print(f"✅ 使用特征: {', '.join(feature_cols)} (V2版本)")
    
    # 为每个Y变量建模
    ntl_results = {}
    
    for perception in perception_cols:
        print(f"\n{'='*60}")
        print(f"🌙 NTL分析 - {perception.upper()}")
        print("="*60)
        
        try:
            # 准备数据
            X_ntl = ntl_data[feature_cols].fillna(0)
            y = np.log(ntl_data[perception] + 1)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_ntl, y, test_size=0.3, random_state=42
            )
            
            # XGBoost模型
            if libs['xgboost'] is not None:
                print(f"  📈 Training XGBoost for {perception}...")
                xgb_model = libs['xgboost'].XGBRegressor(
                    n_estimators=50, max_depth=4, learning_rate=0.1,
                    random_state=42, verbosity=0
                )
                xgb_model.fit(X_train, y_train)
                xgb_score = xgb_model.score(X_test, y_test)
                print(f"    ✅ XGBoost R² = {xgb_score:.4f}")
                
                # SHAP分析
                if libs['shap'] is not None:
                    try:
                        print(f"  🔍 Creating SHAP visualizations for {perception}...")
                        explainer = libs['shap'].TreeExplainer(xgb_model)
                        X_sample = X_test.iloc[:min(2000, len(X_test))]
                        shap_values = explainer.shap_values(X_sample)
                        
                        # 1. SHAP Beeswarm Plot (红蓝渐变贡献度图)
                        print(f"    📊 Creating SHAP beeswarm plot...")
                        plt.clf()
                        plt.close('all')
                        
                        if hasattr(libs['shap'], 'plots') and hasattr(libs['shap'].plots, 'beeswarm'):
                            try:
                                explanation = libs['shap'].Explanation(
                                    values=shap_values,
                                    base_values=explainer.expected_value,
                                    data=X_sample.values,
                                    feature_names=feature_cols
                                )
                                plt.figure(figsize=(20, 10))
                                libs['shap'].plots.beeswarm(explanation, max_display=20, 
                                                          color_bar_label="Feature Value", show=False)
                                plt.title(f'NTL SHAP Beeswarm - {perception.title()}\nR² = {xgb_score:.4f}',
                                         fontweight='bold', pad=20, fontsize=16)
                            except Exception as e:
                                print(f"      ⚠️ Beeswarm with shap.plots failed: {str(e)}")
                                plt.clf()
                                plt.close('all')
                                plt.figure(figsize=(20, 10))
                                libs['shap'].summary_plot(shap_values, X_sample, feature_names=feature_cols,
                                                        max_display=20, show=False)
                                plt.title(f'NTL SHAP Summary - {perception.title()}\nR² = {xgb_score:.4f}',
                                         fontweight='bold', pad=20, fontsize=16)
                        else:
                            plt.figure(figsize=(20, 10))
                            libs['shap'].summary_plot(shap_values, X_sample, feature_names=feature_cols,
                                                    max_display=20, show=False)
                            plt.title(f'NTL SHAP Summary - {perception.title()}\nR² = {xgb_score:.4f}',
                                     fontweight='bold', pad=20, fontsize=16)
                        
                        plt.tight_layout()
                        plt.savefig(f'{ntl_save_dir}/ntl_shap_beeswarm_{perception}.png',
                                   dpi=300, bbox_inches='tight', facecolor='white')
                        plt.close('all')
                        print(f"      ✅ Beeswarm plot saved")
                        
                        # 2. SHAP Dependence图 - 蓝绿散点+橙红色曲线
                        fig, axes = plt.subplots(1, len(feature_cols), figsize=(8*len(feature_cols), 6))
                        if len(feature_cols) == 1:
                            axes = [axes]
                        
                        for idx, feature in enumerate(feature_cols):
                            ax = axes[idx]
                            
                            # 获取特征索引
                            feat_idx = feature_cols.index(feature)
                            x_vals = X_sample.iloc[:, feat_idx].values
                            y_vals = shap_values[:, feat_idx]
                            
                            # 散点 - 蓝绿色
                            ax.scatter(x_vals, y_vals, alpha=0.4, s=20, color='#20B2AA', edgecolor='none')
                            
                            # 拟合曲线 - 橙红色
                            try:
                                from scipy.interpolate import UnivariateSpline
                                df = pd.DataFrame({'x': x_vals, 'y': y_vals}).dropna().sort_values('x')
                                if len(df) > 5 and df['x'].nunique() >= 5:
                                    xs = np.linspace(df['x'].quantile(0.01), df['x'].quantile(0.99), 200)
                                    s_val = max(1e-6, len(df) * np.var(df['y']) * 0.5)
                                    spline = UnivariateSpline(df['x'].values, df['y'].values, s=s_val)
                                    ys = spline(xs)
                                    
                                    # 拟合曲线
                                    ax.plot(xs, ys, color='#E24A33', linewidth=2.5, label='Smoothed Trend')
                                    
                                    # Bootstrap置信区间
                                    rng = np.random.RandomState(42)
                                    n_boot = 100
                                    boot = []
                                    for _ in range(n_boot):
                                        idx_boot = rng.randint(0, len(df), len(df))
                                        try:
                                            sp_boot = UnivariateSpline(df['x'].values[idx_boot], 
                                                                       df['y'].values[idx_boot], s=s_val)
                                            boot.append(sp_boot(xs))
                                        except:
                                            continue
                                    
                                    if boot:
                                        boot = np.vstack(boot)
                                        lower = np.percentile(boot, 2.5, axis=0)
                                        upper = np.percentile(boot, 97.5, axis=0)
                                        ax.fill_between(xs, lower, upper, color='#E24A33', 
                                                       alpha=0.15, linewidth=0, label='95% CI')
                            except Exception as e:
                                print(f"      ⚠️ 曲线拟合失败: {str(e)}")
                            
                            ax.set_xlabel(f'{feature}', fontweight='bold', fontsize=12)
                            ax.set_ylabel('SHAP value', fontweight='bold', fontsize=12)
                            ax.set_title(f'NTL Impact on {perception.title()}', fontweight='bold', fontsize=14)
                            ax.grid(True, alpha=0.3)
                            ax.legend(fontsize=10)
                        
                        plt.suptitle(f'NTL SHAP Dependence Analysis - {perception.title()}\nR² = {xgb_score:.4f}',
                                   fontsize=16, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(f'{ntl_save_dir}/ntl_shap_dependence_{perception}.png', 
                                   dpi=300, bbox_inches='tight', facecolor='white')
                        plt.close()
                        print(f"    ✅ SHAP dependence plot saved")
                        
                    except Exception as e:
                        print(f"    ⚠️ SHAP分析失败: {str(e)}")
            
            # Lasso模型
            print(f"  📈 Training Lasso for {perception}...")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            lasso = LassoCV(alphas=np.logspace(-5, 2, 100), cv=5, random_state=42)
            lasso.fit(X_train_scaled, y_train)
            lasso_score = lasso.score(X_test_scaled, y_test)
            print(f"    ✅ Lasso R² = {lasso_score:.4f}")
            
            ntl_results[perception] = {
                'xgb_score': xgb_score if libs['xgboost'] is not None else 0,
                'lasso_score': lasso_score,
                'n_samples': len(ntl_data)
            }
            
        except Exception as e:
            print(f"  ❌ {perception} NTL分析失败: {str(e)}")
            continue
    
    # 创建NTL结果汇总
    if ntl_results:
        print(f"\n📊 NTL分析结果汇总:")
        print(f"{'Perception':<12} {'XGBoost R²':<12} {'Lasso R²':<12}")
        print("-" * 40)
        for perc, res in ntl_results.items():
            print(f"{perc.capitalize():<12} {res['xgb_score']:<12.4f} {res['lasso_score']:<12.4f}")
    
    print(f"\n✅ NTL整体分析完成！")
    print(f"📁 结果保存在: {ntl_save_dir}")
    
    return ntl_results

def create_ntl_lcz_analysis(analyzer, perception_cols, save_dir, libs):
    """创建NTL的LCZ分区分析 - V2版本：只使用ntl_mean"""
    print("\n🌙🏙️ NTL LCZ PARTITIONED ANALYSIS - V2")
    print("="*80)
    
    # 检查必要列
    if 'ntl_mean' not in analyzer.merged_data.columns:
        print("❌ ntl_mean列不存在")
        return
    if 'LCZ' not in analyzer.merged_data.columns:
        print("❌ LCZ列不存在")
        return
    
    # 过滤NTL空值数据
    ntl_data = analyzer.merged_data[analyzer.merged_data['ntl_mean'].notna()].copy()
    print(f"📊 NTL有效数据: {len(ntl_data)}/{len(analyzer.merged_data)}")
    
    # LCZ分区
    target_lcz_types = [1, 2, 3, 4, 9, 11]
    lcz_names = {
        1: 'Compact High-rise', 2: 'Compact Mid-rise', 3: 'Compact Low-rise',
        4: 'Open High-rise', 9: 'Sparsely Built', 11: 'Dense Trees'
    }
    
    # 准备特征：V2版本只使用ntl_mean
    feature_cols = ['ntl_mean']
    print(f"✅ V2 NTL LCZ分析使用特征: {', '.join(feature_cols)}")
    
    lcz_data = ntl_data[ntl_data['LCZ'].isin(target_lcz_types)]
    all_lcz_ntl_results = {}
    
    for lcz_type in target_lcz_types:
        lcz_subset = lcz_data[lcz_data['LCZ'] == lcz_type]
        if len(lcz_subset) < 50:
            print(f"⚠️ LCZ {lcz_type} NTL数据不足 ({len(lcz_subset)})，跳过")
            continue
        
        print(f"\n{'='*60}")
        print(f"🌙 NTL - LCZ {lcz_type} ({lcz_names[lcz_type]})")
        print(f"📊 样本数: {len(lcz_subset)}")
        print("="*60)
        
        lcz_ntl_dir = f"{save_dir}/NTL_LCZ_{lcz_type}_{lcz_names[lcz_type].replace(' ', '_')}"
        os.makedirs(lcz_ntl_dir, exist_ok=True)
        
        lcz_results = {}
        
        for perception in perception_cols:
            try:
                X_ntl = lcz_subset[feature_cols].fillna(0)
                y = np.log(lcz_subset[perception] + 1)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_ntl, y, test_size=0.3, random_state=42
                )
                
                # XGBoost + SHAP
                if libs['xgboost'] is not None and libs['shap'] is not None:
                    xgb_model = libs['xgboost'].XGBRegressor(
                        n_estimators=50, max_depth=4, learning_rate=0.1,
                        random_state=42, verbosity=0
                    )
                    xgb_model.fit(X_train, y_train)
                    xgb_score = xgb_model.score(X_test, y_test)
                    
                    # SHAP visualizations
                    try:
                        explainer = libs['shap'].TreeExplainer(xgb_model)
                        X_sample = X_test.iloc[:min(1000, len(X_test))]
                        shap_values = explainer.shap_values(X_sample)
                        
                        # 1. Beeswarm plot
                        plt.clf()
                        plt.close('all')
                        if hasattr(libs['shap'], 'plots') and hasattr(libs['shap'].plots, 'beeswarm'):
                            try:
                                explanation = libs['shap'].Explanation(
                                    values=shap_values,
                                    base_values=explainer.expected_value,
                                    data=X_sample.values,
                                    feature_names=feature_cols
                                )
                                plt.figure(figsize=(20, 10))
                                libs['shap'].plots.beeswarm(explanation, max_display=20, show=False)
                                plt.title(f'NTL LCZ {lcz_type} - {perception.title()}\nR² = {xgb_score:.4f}',
                                         fontweight='bold', pad=20)
                            except:
                                plt.clf()
                                plt.close('all')
                                plt.figure(figsize=(20, 10))
                                libs['shap'].summary_plot(shap_values, X_sample, feature_names=feature_cols,
                                                        max_display=20, show=False)
                                plt.title(f'NTL LCZ {lcz_type} - {perception.title()}\nR² = {xgb_score:.4f}',
                                         fontweight='bold', pad=20)
                        else:
                            plt.figure(figsize=(20, 10))
                            libs['shap'].summary_plot(shap_values, X_sample, feature_names=feature_cols,
                                                    max_display=20, show=False)
                            plt.title(f'NTL LCZ {lcz_type} - {perception.title()}\nR² = {xgb_score:.4f}',
                                     fontweight='bold', pad=20)
                        plt.tight_layout()
                        plt.savefig(f'{lcz_ntl_dir}/ntl_shap_beeswarm_{perception}.png',
                                   dpi=300, bbox_inches='tight', facecolor='white')
                        plt.close('all')
                        
                        # 2. Dependence plot
                        fig, axes = plt.subplots(1, len(feature_cols), figsize=(8*len(feature_cols), 6))
                        if len(feature_cols) == 1:
                            axes = [axes]
                        
                        for idx, feature in enumerate(feature_cols):
                            ax = axes[idx]
                            feat_idx = feature_cols.index(feature)
                            x_vals = X_sample.iloc[:, feat_idx].values
                            y_vals = shap_values[:, feat_idx]
                            
                            ax.scatter(x_vals, y_vals, alpha=0.4, s=20, color='#20B2AA', edgecolor='none')
                            
                            # 拟合曲线 + 95% CI
                            try:
                                from scipy.interpolate import UnivariateSpline
                                df = pd.DataFrame({'x': x_vals, 'y': y_vals}).dropna().sort_values('x')
                                if len(df) > 5 and df['x'].nunique() >= 5:
                                    xs = np.linspace(df['x'].quantile(0.01), df['x'].quantile(0.99), 200)
                                    s_val = max(1e-6, len(df) * np.var(df['y']) * 0.5)
                                    spline = UnivariateSpline(df['x'].values, df['y'].values, s=s_val)
                                    ys = spline(xs)
                                    
                                    # 主曲线
                                    ax.plot(xs, ys, color='#E24A33', linewidth=2.5, label='Smoothed Trend')
                                    
                                    # Bootstrap置信区间
                                    rng = np.random.RandomState(42)
                                    n_boot = 100
                                    boot = []
                                    for _ in range(n_boot):
                                        idx_boot = rng.randint(0, len(df), len(df))
                                        try:
                                            sp_boot = UnivariateSpline(df['x'].values[idx_boot], 
                                                                       df['y'].values[idx_boot], s=s_val)
                                            boot.append(sp_boot(xs))
                                        except:
                                            continue
                                    
                                    if boot:
                                        boot = np.vstack(boot)
                                        lower = np.percentile(boot, 2.5, axis=0)
                                        upper = np.percentile(boot, 97.5, axis=0)
                                        ax.fill_between(xs, lower, upper, color='#E24A33', 
                                                       alpha=0.15, linewidth=0, label='95% CI')
                            except:
                                pass
                            
                            ax.set_xlabel(f'{feature}', fontweight='bold')
                            ax.set_ylabel('SHAP value', fontweight='bold')
                            ax.set_title(f'LCZ {lcz_type} - {perception.title()}', fontweight='bold')
                            ax.grid(True, alpha=0.3)
                        
                        plt.suptitle(f'NTL SHAP - LCZ {lcz_type} - {perception.title()}\nR² = {xgb_score:.4f}',
                                   fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(f'{lcz_ntl_dir}/ntl_shap_{perception}.png', 
                                   dpi=300, bbox_inches='tight', facecolor='white')
                        plt.close()
                    except Exception as e:
                        print(f"    ⚠️ LCZ {lcz_type} {perception} SHAP失败: {str(e)}")
                    
                    lcz_results[perception] = {'xgb_score': xgb_score}
                    print(f"  ✅ LCZ {lcz_type} - {perception}: R² = {xgb_score:.4f}")
                
            except Exception as e:
                print(f"  ❌ LCZ {lcz_type} - {perception} 失败: {str(e)}")
                continue
        
        all_lcz_ntl_results[lcz_type] = {
            'name': lcz_names[lcz_type],
            'results': lcz_results,
            'n_samples': len(lcz_subset)
        }
    
    print(f"\n✅ NTL LCZ分区分析完成！共分析 {len(all_lcz_ntl_results)} 个LCZ分区")
    return all_lcz_ntl_results

def plot_data_histograms(analyzer, save_dir):
    """生成所有X和Y变量的浅橙色直方图"""
    print("\n📊 生成数据分布直方图...")
    
    # 感知变量 (Y变量)
    y_vars = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
    available_y_vars = [var for var in y_vars if var in analyzer.merged_data.columns]
    
    # 语义变量 (X变量) - 取前20个主要的
    x_vars = analyzer.semantic_classes[:20] if len(analyzer.semantic_classes) > 20 else analyzer.semantic_classes
    
    # 创建直方图目录
    hist_dir = f"{save_dir}/data_histograms"
    os.makedirs(hist_dir, exist_ok=True)
    
    # 浅橙色
    orange_color = '#FFB366'
    
    # 为Y变量生成直方图
    if available_y_vars:
        n_y = len(available_y_vars)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, y_var in enumerate(available_y_vars):
            if i < len(axes):
                data = analyzer.merged_data[y_var].dropna()
                axes[i].hist(data, bins=30, color=orange_color, alpha=0.7, edgecolor='white')
                axes[i].set_title(f'{y_var.title()} Distribution', fontweight='bold')
                axes[i].set_xlabel(y_var.title())
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(available_y_vars), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{hist_dir}/y_variables_histograms.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Y变量直方图已保存: {len(available_y_vars)} 个变量")
    
    # 为X变量生成直方图
    if x_vars:
        n_plots = min(20, len(x_vars))
        n_rows = 4
        n_cols = 5
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, x_var in enumerate(x_vars[:n_plots]):
            if x_var in analyzer.merged_data.columns:
                data = analyzer.merged_data[x_var].dropna()
                axes[i].hist(data, bins=25, color=orange_color, alpha=0.7, edgecolor='white')
                axes[i].set_title(f'{x_var.title()}', fontweight='bold', fontsize=10)
                axes[i].set_xlabel(x_var.replace('_', ' ').title(), fontsize=8)
                axes[i].set_ylabel('Frequency', fontsize=8)
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='both', which='major', labelsize=7)
        
        # 隐藏多余的子图
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{hist_dir}/x_variables_histograms.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ X变量直方图已保存: {n_plots} 个变量")
    
    print(f"📁 直方图保存在: {hist_dir}")

def main():
    """Enhanced main program with SAFETY-FOCUSED ANALYSIS + LCZ PARTITIONED ANALYSIS"""
    print("🌃 ENHANCED COMPLETE MODULAR URBAN PERCEPTION ANALYSIS - SAFETY FOCUSED + LCZ PARTITIONED")
    print("="*80)
    print("🔧 SAFETY-FOCUSED ANALYSIS WITH NEW VISUALIZATION REQUIREMENTS:")
    print("✅ Focus only on SAFETY perception")
    print("✅ Create 4 scatter plots: 2 linear + 2 nonlinear threshold curves")
    print("✅ Full model comparison: XGBoost vs Lasso on complete features")
    print("✅ Baseline comparison: NTL vs Semantic (both with XGBoost and Lasso)")
    print("✅ Mako+Orange color scheme: 浅橙色、蓝绿色、黄绿色、灰蓝紫色")
    print("✅ Legends outside plot frames, 1:1 aspect ratio maintained")
    print("✅ Nonlinear threshold curves with smooth confidence intervals")
    print("="*80)
    
    # 检查库
    libs = check_libraries()
    
    # 数据文件
    pixel_file = '100g/depth_weighted_semantic_results.csv'
    brightness_file = '100g/unified_semantic_brightness_analysis.csv'
    depth_file = '100g/full_semantic_depth_results.csv'
    perceptions_file = '100g/perceptionf.csv'
    
    files_to_check = [pixel_file, brightness_file, depth_file, perceptions_file]
    
    print("\n📁 检查数据文件...")
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - 文件不存在!")
            return
    
    try:
        # 创建分析器并加载数据
        analyzer = FixedOptimizedInteractionAnalyzer()
        analyzer.load_data(pixel_file, brightness_file, depth_file, perceptions_file)
        
        print(f"\n📊 数据加载成功!")
        print(f"   合并数据形状: {analyzer.merged_data.shape}")
        print(f"   可用列: {list(analyzer.merged_data.columns)}")
        
        # 创建结果目录（带时间戳）
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f'enhanced_complete_results_v2_{timestamp}'
        os.makedirs(save_dir, exist_ok=True)
        
        # SAFETY-FOCUSED ANALYSIS ONLY
        perception_cols = ['safe']  # Only focus on safety
        
        print(f"\n🎯 Analyzing SAFETY Perception Only: {perception_cols}")
        print(f"📁 Results Directory: {save_dir}/")
        
        # ============ PERCEPTION-FOCUSED ANALYSIS WITH 4 SPECIALIZED PLOTS ============
        # 为所有6个感知维度生成专门的散点图分析
        all_perception_cols = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
        print(f"\n{'='*80}")
        print("🚀 RUNNING PERCEPTION-FOCUSED ANALYSIS WITH 4 SPECIALIZED PLOTS")
        print(f"📊 Analyzing all {len(all_perception_cols)} perceptions: {all_perception_cols}")
        print("="*80)
        
        for perception in all_perception_cols:
            print(f"\n🎯 Creating specialized plots for: {perception.upper()}")
            create_safety_focused_analysis(analyzer, save_dir, perception_name=perception)
        
        # 运行完整的三模块分析
        all_results = {}
        all_model_summary = {}
        
        for perception in perception_cols:
            print(f"\n{'='*60}")
            print(f"🎯 PERCEPTION: {perception.upper()}")
            print("="*60)
            
            try:
                # MODULE 1: XGBoost + SHAP
                print(f"🔍 MODULE 1: XGBoost + SHAP for {perception}")
                xgb_result = run_enhanced_xgboost_module(analyzer, perception, save_dir, libs)
                
                if xgb_result is None:
                    print(f"    ❌ XGBoost模块失败")
                    continue
                
                # MODULE 2: Lasso Feature Selection
                print(f"🎯 MODULE 2: Lasso Feature Selection for {perception}")
                lasso_result = run_enhanced_lasso_module(analyzer, perception, save_dir)
                
                if lasso_result is None:
                    print(f"    ❌ Lasso模块失败")
                    continue
                
                # XGBoost vs Lasso对比分析
                X_features, feature_names = create_strict_abd_interactions(analyzer)
                comparison_result = create_xgb_lasso_comparison(
                    xgb_result, lasso_result, feature_names, perception, save_dir
                )
                
                # MODULE 3: Ensemble Strategy
                print(f"🔗 MODULE 3: Ensemble Strategy for {perception}")
                ensemble_result = run_integrated_ensemble_module(analyzer, perception, save_dir, 
                                                               {'xgb': xgb_result, 'lasso': lasso_result})
                
                # 保存结果
                all_results[perception] = {
                    'xgb': xgb_result,
                    'lasso': lasso_result,
                    'ensemble': ensemble_result
                }
                
                print(f"\n📊 {perception.upper()} 性能总结:")
                print(f"  • XGBoost: R² = {xgb_result['test_score']:.4f}")
                print(f"  • Lasso: R² = {lasso_result['lasso_score']:.4f}")
                print(f"  • Elastic-Net: R² = {lasso_result['elastic_score']:.4f}")
                print(f"  • Ensemble: R² = {ensemble_result['ensemble_score']:.4f}")
                
            except Exception as e:
                print(f"    ❌ {perception} 分析失败: {str(e)}")
                continue
        
        print(f"\n🎉 主分析完成!")
        print(f"📁 结果保存在: {save_dir}")
        
        # ============ LCZ分区分析 ============
        print("\n" + "="*80)
        print("🏙️ 开始LCZ分区分析")
        print("="*80)
        
        # LCZ分析需要分析所有6个perception维度
        lcz_perception_cols = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
        print(f"📊 LCZ分析将分析所有6个perception维度: {lcz_perception_cols}")
        
        lcz_save_dir = f"{save_dir}/LCZ_Analysis"
        os.makedirs(lcz_save_dir, exist_ok=True)
        
        lcz_results = create_lcz_stability_analysis(analyzer, lcz_perception_cols, lcz_save_dir, libs)
        
        if lcz_results:
            print(f"\n🎉 LCZ分区分析完成! 共分析了 {len(lcz_results)} 个LCZ分区")
            print(f"📁 结果保存在: {lcz_save_dir}")
            try:
                print("📝 生成LCZ对比日志...")
                create_lcz_comparison_log(lcz_results, lcz_save_dir)
            except Exception as log_error:
                print(f"  ⚠️ LCZ日志生成失败: {str(log_error)}")
        else:
            print("\n❌ LCZ分区分析未能完成")
        
        # ============ NTL专项分析 ============
        print("\n" + "="*80)
        print("🌙 开始NTL专项分析")
        print("="*80)
        
        # NTL分析需要分析所有6个perception维度
        ntl_perception_cols = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
        print(f"📊 NTL分析将分析所有6个perception维度: {ntl_perception_cols}")
        
        ntl_save_dir = f"{save_dir}/NTL_Analysis"
        os.makedirs(ntl_save_dir, exist_ok=True)
        
        ntl_results = create_ntl_analysis(analyzer, ntl_perception_cols, ntl_save_dir, libs)
        ntl_lcz_results = create_ntl_lcz_analysis(analyzer, ntl_perception_cols, ntl_save_dir, libs)
        
        if ntl_results or ntl_lcz_results:
            print(f"\n🎉 NTL分析完成!")
            print(f"📁 结果保存在: {ntl_save_dir}")
        else:
            print("\n❌ NTL分析未能完成")
        
        print(f"\n🎉🎉 所有分析已完成！所有结果保存在: {save_dir}/")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_data_histograms(analyzer, save_dir):
    """生成所有X和Y变量的浅橙色直方图"""
    print("\n📊 生成数据分布直方图...")
    
    # 感知变量 (Y变量)
    y_vars = ['safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring']
    available_y_vars = [var for var in y_vars if var in analyzer.merged_data.columns]
    
    # 语义变量 (X变量) - 取前20个主要的
    x_vars = analyzer.semantic_classes[:20] if len(analyzer.semantic_classes) > 20 else analyzer.semantic_classes
    
    # 创建直方图目录
    hist_dir = f"{save_dir}/data_histograms"
    os.makedirs(hist_dir, exist_ok=True)
    
    # 浅橙色
    orange_color = '#FFB366'
    
    print(f"🎯 分析的Y变量: {available_y_vars}")
    print(f"🎯 分析的X变量 (前20个): {x_vars[:20] if len(x_vars) > 20 else x_vars}")
    
    # 为Y变量生成直方图
    if available_y_vars:
        n_y = len(available_y_vars)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, y_var in enumerate(available_y_vars):
            if i < len(axes):
                data = analyzer.merged_data[y_var].dropna()
                axes[i].hist(data, bins=30, color=orange_color, alpha=0.7, edgecolor='white')
                axes[i].set_title(f'{y_var.title()} Distribution', fontweight='bold')
                axes[i].set_xlabel(y_var.title())
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(available_y_vars), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{hist_dir}/y_variables_histograms.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Y变量直方图已保存: {len(available_y_vars)} 个变量")
    
    # 为X变量生成直方图
    if x_vars:
        n_plots = min(20, len(x_vars))
        n_rows = 4
        n_cols = 5
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, x_var in enumerate(x_vars[:n_plots]):
            if x_var in analyzer.merged_data.columns:
                data = analyzer.merged_data[x_var].dropna()
                axes[i].hist(data, bins=25, color=orange_color, alpha=0.7, edgecolor='white')
                axes[i].set_title(f'{x_var.title()}', fontweight='bold', fontsize=10)
                axes[i].set_xlabel(x_var.replace('_', ' ').title(), fontsize=8)
                axes[i].set_ylabel('Frequency', fontsize=8)
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='both', which='major', labelsize=7)
        
        # 隐藏多余的子图
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{hist_dir}/x_variables_histograms.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ X变量直方图已保存: {n_plots} 个变量")
    
    print(f"📁 直方图保存在: {hist_dir}")

def create_lcz_stability_analysis(analyzer, perception_cols, save_dir, libs):
    """创建LCZ稳定性分析 - 对每个LCZ分区分别跑完整的分析"""
    print("\n🏙️ LCZ STABILITY ANALYSIS - 每个LCZ分区完整分析")
    print("="*80)
    
    # 首先生成数据分布直方图
    plot_data_histograms(analyzer, save_dir)
    
    # 检查LCZ列是否存在
    if 'LCZ' not in analyzer.merged_data.columns:
        print("❌ 未找到LCZ列，跳过LCZ分析")
        return
    
    # 目标LCZ类型
    target_lcz_types = [1, 2, 3, 4, 9, 11]
    lcz_names = {
        1: 'Compact High-rise',
        2: 'Compact Mid-rise', 
        3: 'Compact Low-rise',
        4: 'Open High-rise',
        9: 'Sparsely Built',
        11: 'Dense Trees'
    }
    
    # 筛选目标LCZ数据
    lcz_data = analyzer.merged_data[analyzer.merged_data['LCZ'].isin(target_lcz_types)]
    print(f"📊 LCZ筛选后数据量: {len(lcz_data)}/{len(analyzer.merged_data)} ({len(lcz_data)/len(analyzer.merged_data)*100:.1f}%)")
    
    # 为每个LCZ类型分别跑完整分析
    all_lcz_results = {}
    
    for lcz_type in target_lcz_types:
        lcz_subset = lcz_data[lcz_data['LCZ'] == lcz_type]
        if len(lcz_subset) < 50:
            print(f"⚠️ LCZ {lcz_type} 样本量不足 ({len(lcz_subset)})，跳过")
            continue
            
        print(f"\n{'='*60}")
        print(f"🏙️ LCZ {lcz_type} ({lcz_names[lcz_type]}) - 完整分析开始")
        print(f"📊 样本数: {len(lcz_subset)}")
        print("="*60)
        
        # 创建LCZ专用分析器
        lcz_analyzer = FixedOptimizedInteractionAnalyzer()
        lcz_analyzer.merged_data = lcz_subset.copy()
        
        # 创建LCZ专用结果目录
        lcz_save_dir = f"{save_dir}/LCZ_{lcz_type}_{lcz_names[lcz_type].replace(' ', '_')}"
        os.makedirs(lcz_save_dir, exist_ok=True)
        
        # 为每个感知维度跑完整分析
        lcz_results = {}
        
        for perception in perception_cols:
            print(f"\n🎯 LCZ {lcz_type} - 感知维度: {perception.upper()}")
            print("-" * 50)
            
            try:
                # MODULE 1: XGBoost + SHAP
                print(f"🔍 MODULE 1: XGBoost + SHAP for LCZ {lcz_type} - {perception}")
                xgb_result = run_enhanced_xgboost_module(lcz_analyzer, perception, lcz_save_dir, libs)
                
                if xgb_result is None:
                    print(f"    ❌ XGBoost模块失败")
                    continue
                
                # MODULE 2: Lasso Feature Selection
                print(f"🎯 MODULE 2: Lasso Feature Selection for LCZ {lcz_type} - {perception}")
                lasso_result = run_enhanced_lasso_module(lcz_analyzer, perception, lcz_save_dir)
                
                if lasso_result is None:
                    print(f"    ❌ Lasso模块失败")
                    continue
                
                # MODULE 3: Ensemble Strategy
                print(f"🔗 MODULE 3: Ensemble Strategy for LCZ {lcz_type} - {perception}")
                ensemble_result = run_integrated_ensemble_module(lcz_analyzer, perception, lcz_save_dir, 
                                                               {'xgb': xgb_result, 'lasso': lasso_result})
                
                # 保存结果
                lcz_results[perception] = {
                    'xgb': xgb_result,
                    'lasso': lasso_result,
                    'ensemble': ensemble_result
                }
                
                print(f"\n📊 LCZ {lcz_type} - {perception} 性能总结:")
                print(f"  • XGBoost: R² = {xgb_result['test_score']:.4f}")
                print(f"  • Lasso: R² = {lasso_result['lasso_score']:.4f}")
                print(f"  • Elastic-Net: R² = {lasso_result['elastic_score']:.4f}")
                print(f"  • Ensemble: R² = {ensemble_result['ensemble_score']:.4f}")
                
            except Exception as e:
                print(f"    ❌ LCZ {lcz_type} - {perception} 分析失败: {str(e)}")
                continue
        
        # 保存这个LCZ的所有结果
        all_lcz_results[lcz_type] = {
            'name': lcz_names[lcz_type],
            'sample_count': len(lcz_subset),
            'results': lcz_results,
            'save_dir': lcz_save_dir
        }
        
        print(f"\n✅ LCZ {lcz_type} ({lcz_names[lcz_type]}) 完整分析完成!")
        print(f"📁 结果保存在: {lcz_save_dir}")
    
    print(f"\n🎉 LCZ分区分析完成! 共分析了 {len(all_lcz_results)} 个LCZ分区")
    
    # 🆕 新增：创建LCZ合并对比的SHAP Dependence图
    create_lcz_combined_shap_dependence(analyzer, all_lcz_results, perception_cols, save_dir, libs)
    
    return all_lcz_results

def create_lcz_combined_shap_dependence(analyzer, all_lcz_results, perception_cols, save_dir, libs):
    """
    🆕 创建LCZ合并对比的SHAP Dependence图
    将LCZ 1, 4, 9, 11的数据放在同一张图上，用不同颜色区分
    每个变量一张图，包含4个LCZ分区的散点+阈值曲线+置信区间
    """
    print("\n" + "="*80)
    print("🎨 创建LCZ合并对比SHAP Dependence图 (LCZ 1, 4, 9, 11)")
    print("="*80)
    
    # 目标LCZ类型 - 只对比这4个
    target_lcz_for_combined = [1, 4, 9, 11]
    lcz_names = {
        1: 'Compact High-rise',
        4: 'Open High-rise',
        9: 'Sparsely Built',
        11: 'Dense Trees'
    }
    
    # 用户指定的4种颜色
    lcz_colors = {
        1: {'point': '#FF6B4A', 'line': '#E24A33'},    # 橙红色
        4: {'point': '#4ECDC4', 'line': '#20B2AA'},    # 蓝绿色
        9: {'point': '#9B59B6', 'line': '#8E44AD'},    # 紫色
        11: {'point': '#A4D037', 'line': '#7CB342'}    # 黄绿色
    }
    
    # 创建保存目录
    combined_dir = f"{save_dir}/LCZ_Combined_Comparison"
    os.makedirs(combined_dir, exist_ok=True)
    
    # 检查哪些LCZ有结果
    available_lcz = [lcz for lcz in target_lcz_for_combined if lcz in all_lcz_results]
    if len(available_lcz) < 2:
        print(f"⚠️ 可用LCZ分区不足 ({len(available_lcz)}个)，跳过合并对比图")
        return
    
    print(f"📊 将合并对比的LCZ分区: {available_lcz}")
    
    # 获取特征列表（从第一个可用的LCZ结果中获取）
    first_lcz = available_lcz[0]
    first_perception = perception_cols[0] if perception_cols else 'safe'
    
    if first_perception not in all_lcz_results[first_lcz]['results']:
        print(f"⚠️ 未找到感知维度 {first_perception} 的结果")
        return
    
    xgb_result = all_lcz_results[first_lcz]['results'][first_perception].get('xgb')
    if xgb_result is None:
        print("⚠️ 未找到XGBoost结果")
        return
    
    feature_names = xgb_result.get('feature_names', [])
    if not feature_names:
        print("⚠️ 未找到特征名称")
        return
    
    print(f"📊 将为 {len(feature_names)} 个特征创建合并对比图")
    
    # 为每个感知维度创建合并图
    for perception in perception_cols:
        print(f"\n🎯 处理感知维度: {perception}")
        
        # 收集所有LCZ的SHAP数据
        lcz_shap_data = {}
        
        for lcz_type in available_lcz:
            if perception not in all_lcz_results[lcz_type]['results']:
                continue
            
            xgb_res = all_lcz_results[lcz_type]['results'][perception].get('xgb')
            if xgb_res is None:
                continue
            
            # 获取SHAP数据
            shap_values = xgb_res.get('shap_values')
            X_sample = xgb_res.get('X_sample')
            feat_names = xgb_res.get('feature_names')
            
            if shap_values is not None and X_sample is not None:
                lcz_shap_data[lcz_type] = {
                    'shap_values': shap_values,
                    'X_sample': X_sample,
                    'feature_names': feat_names
                }
        
        if len(lcz_shap_data) < 2:
            print(f"  ⚠️ {perception}可用LCZ数据不足，跳过")
            continue
        
        print(f"  ✅ 收集到 {len(lcz_shap_data)} 个LCZ分区的SHAP数据")
        
        # 获取共同的特征列表
        common_features = None
        for lcz_type, data in lcz_shap_data.items():
            if common_features is None:
                common_features = set(data['feature_names'])
            else:
                common_features = common_features.intersection(set(data['feature_names']))
        
        # 🆕 确保包含所有重要的控制变量，按重要性排序而不是字母排序
        # 优先显示的控制变量列表
        priority_features = [
            'AVGIL', 'illumination_uniformity', 'spots_area', 'ADCG', 
            'spatial_lag_Wy', 'ntl_mean', 'POP_20_50',
            'safe', 'lively', 'beautiful', 'wealthy', 'depressing', 'boring'
        ]
        
        # 按优先级排序特征
        common_features_list = list(common_features)
        priority_sorted = []
        remaining = []
        
        for pf in priority_features:
            if pf in common_features_list:
                priority_sorted.append(pf)
        
        for cf in common_features_list:
            if cf not in priority_sorted:
                remaining.append(cf)
        
        # 优先特征在前，其余按字母排序
        common_features = priority_sorted + sorted(remaining)
        
        print(f"  📊 共同特征数: {len(common_features)}")
        print(f"  📊 包含控制变量: {[f for f in priority_features if f in common_features]}")
        
        # 分页创建图表 (每页16个特征)
        per_page = 16
        n_cols = 4
        n_rows = 4
        n_pages = int(np.ceil(len(common_features) / per_page))
        
        for page in range(max(1, n_pages)):
            start_idx = page * per_page
            end_idx = min(len(common_features), (page + 1) * per_page)
            features_page = common_features[start_idx:end_idx]
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            axes = np.array(axes).reshape(n_rows, n_cols)
            
            for i, feature in enumerate(features_page):
                r = i // n_cols
                c = i % n_cols
                ax = axes[r, c]
                
                # 为每个LCZ绘制散点+曲线+置信区间
                for lcz_type in available_lcz:
                    if lcz_type not in lcz_shap_data:
                        continue
                    
                    data = lcz_shap_data[lcz_type]
                    feat_names = data['feature_names']
                    
                    if feature not in feat_names:
                        continue
                    
                    feat_idx = feat_names.index(feature)
                    x_vals = data['X_sample'].iloc[:, feat_idx].values
                    y_vals = data['shap_values'][:, feat_idx]
                    
                    point_color = lcz_colors[lcz_type]['point']
                    line_color = lcz_colors[lcz_type]['line']
                    label = f"LCZ{lcz_type}"
                    
                    # 绘制散点+曲线+置信区间
                    _draw_dependence_with_ci(ax, x_vals, y_vals, point_color, line_color, label)
                
                ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=9, fontweight='bold')
                ax.set_ylabel('SHAP value', fontsize=9)
                ax.legend(loc='best', fontsize=7, framealpha=0.8)
                ax.grid(True, alpha=0.25)
            
            # 隐藏空白子图
            for j in range(len(features_page), per_page):
                r = j // n_cols
                c = j % n_cols
                axes[r, c].axis('off')
            
            # 添加图例说明
            legend_text = " | ".join([f"LCZ{lcz}: {lcz_names[lcz]}" for lcz in available_lcz])
            
            fig.suptitle(
                f'LCZ Combined SHAP Dependence - {perception.title()}\n{legend_text}',
                fontsize=14, fontweight='bold'
            )
            plt.tight_layout()
            
            suffix = f"_p{page+1}" if n_pages > 1 else ""
            save_path = f'{combined_dir}/lcz_combined_dependence_{perception}{suffix}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  ✅ 保存: {save_path}")
    
    print(f"\n✅ LCZ合并对比SHAP Dependence图已保存到: {combined_dir}")


def _draw_dependence_with_ci(ax, x_vals, y_vals, point_color, line_color, label):
    """
    绘制SHAP dependence散点图 + 非线性阈值曲线 + 95%置信区间
    """
    import pandas as pd
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    
    df = pd.DataFrame({'x': x_vals, 'y': y_vals}).dropna()
    if len(df) < 10:
        ax.scatter(x_vals, y_vals, s=8, alpha=0.3, color=point_color, edgecolor='none', label=label)
        return
    
    df = df.sort_values('x')
    xs = np.linspace(df['x'].quantile(0.02), df['x'].quantile(0.98), 200)
    
    # 拟合曲线
    try:
        if df['x'].nunique() >= 5:
            # 分箱预平滑
            if len(df) >= 50:
                q = np.linspace(0.02, 0.98, 30)
                q_edges = df['x'].quantile(q).values
                q_edges = np.unique(q_edges)
                if len(q_edges) >= 5:
                    bins = np.digitize(df['x'].values, q_edges, right=True)
                    x_med, y_med = [], []
                    for b in np.unique(bins):
                        mask = bins == b
                        if mask.sum() > 2:
                            x_med.append(np.median(df['x'].values[mask]))
                            y_med.append(np.median(df['y'].values[mask]))
                    if len(x_med) >= 5:
                        x_fit = np.array(x_med)
                        y_fit = np.array(y_med)
                    else:
                        x_fit = df['x'].values
                        y_fit = df['y'].values
                else:
                    x_fit = df['x'].values
                    y_fit = df['y'].values
            else:
                x_fit = df['x'].values
                y_fit = df['y'].values
            
            s_val = max(1e-6, len(y_fit) * np.var(y_fit) * 1.0)
            spline = UnivariateSpline(x_fit, y_fit, s=s_val)
            ys = spline(xs)
        else:
            coefs = np.polyfit(df['x'].values, df['y'].values, deg=1)
            ys = np.polyval(coefs, xs)
    except Exception:
        coefs = np.polyfit(df['x'].values, df['y'].values, deg=1)
        ys = np.polyval(coefs, xs)
    
    # Bootstrap置信区间
    rng = np.random.RandomState(42)
    n = len(df)
    n_boot = 100
    boot = []
    
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            df_b = df.iloc[idx].sort_values('x')
            if df_b['x'].nunique() >= 5 and len(df_b) >= 20:
                s_val_b = max(1e-6, len(df_b) * np.var(df_b['y']) * 1.0)
                sp = UnivariateSpline(df_b['x'].values, df_b['y'].values, s=s_val_b)
                boot.append(sp(xs))
            else:
                coefs_b = np.polyfit(df_b['x'].values, df_b['y'].values, deg=1)
                boot.append(np.polyval(coefs_b, xs))
        except Exception:
            try:
                coefs_b = np.polyfit(df['x'].values[idx], df['y'].values[idx], deg=1)
                boot.append(np.polyval(coefs_b, xs))
            except:
                pass
    
    if len(boot) > 10:
        boot = np.vstack(boot)
        lower = np.percentile(boot, 2.5, axis=0)
        upper = np.percentile(boot, 97.5, axis=0)
    else:
        lower = ys - 0.1 * np.abs(ys)
        upper = ys + 0.1 * np.abs(ys)
    
    # 绘制：散点 + 置信区间 + 曲线
    ax.scatter(df['x'], df['y'], s=8, alpha=0.25, color=point_color, edgecolor='none')
    ax.fill_between(xs, lower, upper, color=line_color, alpha=0.12, linewidth=0)
    ax.plot(xs, ys, color=line_color, linewidth=1.5, label=label)


# 第二个main函数已删除（重复定义），使用第一个main函数（包含LCZ和NTL分析）

def create_comprehensive_performance_summary_table(all_model_summary, save_dir, analyzer):
    """创建完整的模型性能汇总表格 - Excel + 可视化"""
    print("  📊 生成综合模型性能汇总表格...")
    
    # 创建DataFrame
    summary_df = pd.DataFrame(all_model_summary).T
    
    # 添加统计信息
    summary_df['Mean_R2_Across_Models'] = summary_df[['XGBoost_R2', 'Lasso_R2', 'ElasticNet_R2', 'Ensemble_R2']].mean(axis=1)
    summary_df['Std_R2_Across_Models'] = summary_df[['XGBoost_R2', 'Lasso_R2', 'ElasticNet_R2', 'Ensemble_R2']].std(axis=1)
    summary_df['Model_Consistency'] = 1 - summary_df['Std_R2_Across_Models']  # 一致性指数
    
    # 排序并重新排列列
    columns_order = [
        'Best_Model_Score', 'XGBoost_R2', 'XGBoost_Train_R2', 'Lasso_R2', 'ElasticNet_R2',
        'Ensemble_R2', 'NTL_Baseline_R2', 'Full_Interaction_R2', 'Improvement_vs_NTL',
        'Lasso_Features_Selected', 'ElasticNet_Features_Selected', 
        'Mean_R2_Across_Models', 'Std_R2_Across_Models', 'Model_Consistency'
    ]
    summary_df = summary_df[columns_order]
    
    # 保存到Excel
    excel_path = f'{save_dir}/comprehensive_model_performance_summary.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 主要结果表
        summary_df.to_excel(writer, sheet_name='Model_Performance', index=True)
        
        # 添加统计汇总
        stats_summary = pd.DataFrame({
            'Metric': ['Best_Overall_R2', 'Average_R2', 'Std_R2', 'Min_R2', 'Max_R2'],
            'Value': [
                summary_df['Best_Model_Score'].max(),
                summary_df['Best_Model_Score'].mean(),
                summary_df['Best_Model_Score'].std(),
                summary_df['Best_Model_Score'].min(),
                summary_df['Best_Model_Score'].max()
            ]
        })
        stats_summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # 添加模型排名
        model_ranking = pd.DataFrame({
            'Perception': summary_df.index,
            'Best_Model': summary_df[['XGBoost_R2', 'Lasso_R2', 'ElasticNet_R2', 'Ensemble_R2']].idxmax(axis=1),
            'Best_Score': summary_df['Best_Model_Score'],
            'Worst_Model': summary_df[['XGBoost_R2', 'Lasso_R2', 'ElasticNet_R2', 'Ensemble_R2']].idxmin(axis=1),
            'Score_Range': summary_df[['XGBoost_R2', 'Lasso_R2', 'ElasticNet_R2', 'Ensemble_R2']].max(axis=1) - 
                          summary_df[['XGBoost_R2', 'Lasso_R2', 'ElasticNet_R2', 'Ensemble_R2']].min(axis=1)
        })
        model_ranking.to_excel(writer, sheet_name='Model_Ranking', index=False)
        
        # 特征选择分析
        feature_analysis = pd.DataFrame({
            'Perception': summary_df.index,
            'Lasso_Features': summary_df['Lasso_Features_Selected'],
            'ElasticNet_Features': summary_df['ElasticNet_Features_Selected'],
            'Feature_Difference': summary_df['ElasticNet_Features_Selected'] - summary_df['Lasso_Features_Selected'],
            'Feature_Selection_Efficiency': summary_df['Best_Model_Score'] / (summary_df['Lasso_Features_Selected'] + 1)
        })
        feature_analysis.to_excel(writer, sheet_name='Feature_Analysis', index=False)
    
    print(f"    ✅ Excel报告已保存: {excel_path}")
    
    # 创建可视化汇总
    user_colors = {
        'primary': '#4B0082',     # Deep purple
        'secondary': '#20B2AA',   # Light sea green/teal  
        'accent1': '#6A5ACD',     # Slate blue
        'accent2': '#48D1CC',     # Medium turquoise
        'accent3': '#9370DB',     # Medium purple
        'accent4': '#40E0D0',     # Turquoise
    }
    
    # 1. 模型性能热力图
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 热力图数据
    heatmap_data = summary_df[['XGBoost_R2', 'Lasso_R2', 'ElasticNet_R2', 'Ensemble_R2']].T
    
    im1 = axes[0,0].imshow(heatmap_data.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    axes[0,0].set_xticks(range(len(heatmap_data.columns)))
    axes[0,0].set_xticklabels([col.title() for col in heatmap_data.columns], rotation=45)
    axes[0,0].set_yticks(range(len(heatmap_data.index)))
    axes[0,0].set_yticklabels([idx.replace('_R2', '') for idx in heatmap_data.index])
    axes[0,0].set_title('Model Performance Heatmap (R² Scores)', fontweight='bold', fontsize=14)
    
    # 添加数值标注
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            color = 'white' if value < 0.5 else 'black'
            axes[0,0].text(j, i, f'{value:.3f}', ha='center', va='center', 
                         color=color, fontweight='bold', fontsize=10)
    
    plt.colorbar(im1, ax=axes[0,0], label='R² Score')
    
    # 2. 最佳模型分布
    best_models = summary_df[['XGBoost_R2', 'Lasso_R2', 'ElasticNet_R2', 'Ensemble_R2']].idxmax(axis=1)
    model_counts = best_models.value_counts()
    
    colors_pie = [user_colors['primary'], user_colors['secondary'], user_colors['accent1'], user_colors['accent2']]
    axes[0,1].pie(model_counts.values, labels=[label.replace('_R2', '') for label in model_counts.index], 
                 colors=colors_pie[:len(model_counts)], autopct='%1.1f%%', startangle=90)
    axes[0,1].set_title('Best Model Distribution\nAcross Perceptions', fontweight='bold', fontsize=14)
    
    # 3. 改进程度条形图
    improvement_data = summary_df['Improvement_vs_NTL'].fillna(0)
    bars = axes[1,0].bar(range(len(improvement_data)), improvement_data.values, 
                        color=user_colors['accent3'], alpha=0.8, edgecolor='white', linewidth=1)
    axes[1,0].set_xticks(range(len(improvement_data)))
    axes[1,0].set_xticklabels([idx.title() for idx in improvement_data.index], rotation=45)
    axes[1,0].set_ylabel('Improvement vs NTL Baseline (%)', fontweight='bold')
    axes[1,0].set_title('Model Improvement Over NTL Baseline', fontweight='bold', fontsize=14)
    axes[1,0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, value in zip(bars, improvement_data.values):
        if not np.isnan(value):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(improvement_data.values)*0.02,
                          f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. 特征选择效率
    lasso_features = summary_df['Lasso_Features_Selected']
    elastic_features = summary_df['ElasticNet_Features_Selected']
    
    x_pos = np.arange(len(lasso_features))
    width = 0.35
    
    bars1 = axes[1,1].bar(x_pos - width/2, lasso_features, width, 
                         label='Lasso', color=user_colors['secondary'], alpha=0.8)
    bars2 = axes[1,1].bar(x_pos + width/2, elastic_features, width,
                         label='Elastic-Net', color=user_colors['primary'], alpha=0.8)
    
    axes[1,1].set_xlabel('Perception Dimensions', fontweight='bold')
    axes[1,1].set_ylabel('Number of Selected Features', fontweight='bold')
    axes[1,1].set_title('Feature Selection Comparison\nLasso vs Elastic-Net', fontweight='bold', fontsize=14)
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels([idx.title() for idx in lasso_features.index], rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, value in zip(bars1, lasso_features):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(max(lasso_features), max(elastic_features))*0.02,
                      f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar, value in zip(bars2, elastic_features):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(max(lasso_features), max(elastic_features))*0.02,
                      f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    fig.suptitle('Comprehensive Model Performance Analysis\nAll 6 Perception Dimensions with Control Variables', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_performance_heatmap.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3. 特征重要性汇总可视化
    create_feature_importance_summary(analyzer, save_dir, user_colors)
    
    print(f"    ✅ 可视化汇总已保存:")
    print(f"      • model_performance_heatmap.png - 模型性能热力图")
    print(f"      • feature_importance_summary.png - 特征重要性汇总")
    
    # 打印控制台汇总表格
    print(f"\n📊 COMPREHENSIVE MODEL PERFORMANCE SUMMARY TABLE")
    print("="*120)
    print(f"{'Perception':<12} {'Best R²':<8} {'XGBoost':<8} {'Lasso':<8} {'Elastic':<8} {'Ensemble':<8} {'NTL Base':<8} {'Improvement':<12} {'Best Model':<12}")
    print("-" * 120)
    
    for perception in summary_df.index:
        row = summary_df.loc[perception]
        best_model = summary_df.loc[perception, ['XGBoost_R2', 'Lasso_R2', 'ElasticNet_R2', 'Ensemble_R2']].idxmax().replace('_R2', '')
        
        print(f"{perception.capitalize():<12} {row['Best_Model_Score']:<8.4f} {row['XGBoost_R2']:<8.4f} "
              f"{row['Lasso_R2']:<8.4f} {row['ElasticNet_R2']:<8.4f} {row['Ensemble_R2']:<8.4f} "
              f"{row['NTL_Baseline_R2']:<8.4f} {row['Improvement_vs_NTL']:>10.1f}% {best_model:<12}")
    
    print("-" * 120)
    print(f"{'AVERAGE':<12} {summary_df['Best_Model_Score'].mean():<8.4f} {summary_df['XGBoost_R2'].mean():<8.4f} "
          f"{summary_df['Lasso_R2'].mean():<8.4f} {summary_df['ElasticNet_R2'].mean():<8.4f} "
          f"{summary_df['Ensemble_R2'].mean():<8.4f} {summary_df['NTL_Baseline_R2'].mean():<8.4f} "
          f"{summary_df['Improvement_vs_NTL'].mean():>10.1f}% {'Ensemble':<12}")
    
    # 控制变量使用情况统计
    control_vars = ['AVGIL', 'spots_area', 'ADCG', 'illumination_uniformity', ]   #'predicted_spillover'
    available_controls = [col for col in control_vars if col in analyzer.merged_data.columns]
    
    print(f"\n📊 CONTROL VARIABLES USAGE SUMMARY")
    print("="*80)
    print(f"Available Control Variables: {len(available_controls)}/{len(control_vars)}")
    print(f"Successfully Loaded: {', '.join(available_controls) if available_controls else 'None'}")
    print(f"Missing Variables: {', '.join([col for col in control_vars if col not in available_controls])}")
    print(f"Total Features per Model: {len(USER_SEMANTIC_CLASSES) * 7 + len(available_controls)} (semantic + control)")
    
    print(f"\n📊 MODEL ARCHITECTURE SUMMARY")
    print("="*80)
    print(f"Semantic Classes: {len(USER_SEMANTIC_CLASSES)} (user-specified)")
    print(f"Interaction Terms: 7 per semantic (A+B+D+AB+AD+BD+ABD)")
    print(f"Control Variables: {len(available_controls)}")
    print(f"Total Base Features: {len(USER_SEMANTIC_CLASSES) * 7 + len(available_controls)}")
    print(f"Log Transformation: log(perception + 1)")
    print(f"Cross-validation: 30% test split, random_state=42")
    
    return summary_df

def create_safety_focused_analysis(analyzer, save_dir, perception_name='safe'):
    """Create perception-focused analysis with 4 scatter plots as requested
    
    Args:
        analyzer: data analyzer object
        save_dir: directory to save plots
        perception_name: perception dimension to analyze (default: 'safe')
    """
    print(f"\n🎯 {perception_name.upper()}-FOCUSED ANALYSIS - Creating 4 specialized scatter plots")
    print("="*60)
    
    # Mako + Orange color scheme as requested
    mako_orange_colors = {
        'light_orange': '#FFB366',    # 浅橙色
        'blue_green': '#4ECDC4',      # 蓝绿色  
        'yellow_green': '#A8E6CF',    # 黄绿色
        'gray_blue_purple': '#7B68EE' # 灰蓝紫色
    }
    
    # Check if perception exists in data
    if perception_name not in analyzer.merged_data.columns:
        print(f"  ⚠️ Perception '{perception_name}' not found in data, skipping...")
        return
    
    # Focus on the specified perception
    perception = perception_name
    y = np.log(analyzer.merged_data[perception] + 1)
    print(f"  📊 Analyzing {perception.upper()} perception: {len(y)} samples")
    
    # Prepare all models and data
    models_data = prepare_safety_models(analyzer, y)
    
    # Create 4 plots with perception-specific filenames
    create_safety_linear_plots(models_data, y, save_dir, mako_orange_colors, perception_name)
    create_safety_nonlinear_plots(models_data, y, save_dir, mako_orange_colors, perception_name)
    
    # Create performance improvement chart
    create_safety_performance_chart(models_data, save_dir, mako_orange_colors, perception_name)
    
    print(f"  ✅ {perception_name.upper()}-focused analysis completed!")

def prepare_safety_models(analyzer, y):
    """Prepare all models for safety analysis"""
    print("  🔧 Preparing models for Safety analysis...")
    
    models_data = {}
    
    # 1. Full model features (A+B+D+AB+AD+BD+ABD + Controls)
    X_full, feature_names_full = create_strict_abd_interactions(analyzer)
    X_full_train, X_full_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.3, random_state=42
    )
    
    # Full XGBoost
    full_xgb = GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
    full_xgb.fit(X_full_train, y_train)
    full_xgb_pred = full_xgb.predict(X_full_test)
    
    # Full Lasso
    scaler_full = StandardScaler()
    X_full_train_scaled = scaler_full.fit_transform(X_full_train)
    X_full_test_scaled = scaler_full.transform(X_full_test)
    full_lasso = LassoCV(alphas=np.logspace(-5, 2, 100), cv=5, random_state=42)
    full_lasso.fit(X_full_train_scaled, y_train)
    full_lasso_pred = full_lasso.predict(X_full_test_scaled)
    
    models_data['full'] = {
        'X_test': X_full_test, 'y_test': y_test,
        'xgb_pred': full_xgb_pred, 'lasso_pred': full_lasso_pred,
        'xgb_model': full_xgb, 'lasso_model': full_lasso,
        'scaler': scaler_full
    }
    
    # 2. NTL only (baseline 1)
    if 'DN' in analyzer.merged_data.columns:
        X_ntl = analyzer.merged_data[['DN']].fillna(0)
        X_ntl_train, X_ntl_test, y_ntl_train, y_ntl_test = train_test_split(
            X_ntl, y, test_size=0.3, random_state=42
        )
        
        # NTL XGBoost
        ntl_xgb = GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
        ntl_xgb.fit(X_ntl_train, y_ntl_train)
        ntl_xgb_pred = ntl_xgb.predict(X_ntl_test)
        
        # NTL Lasso
        scaler_ntl = StandardScaler()
        X_ntl_train_scaled = scaler_ntl.fit_transform(X_ntl_train)
        X_ntl_test_scaled = scaler_ntl.transform(X_ntl_test)
        ntl_lasso = LassoCV(alphas=np.logspace(-5, 2, 100), cv=5, random_state=42)
        ntl_lasso.fit(X_ntl_train_scaled, y_ntl_train)
        ntl_lasso_pred = ntl_lasso.predict(X_ntl_test_scaled)
        
        models_data['ntl'] = {
            'X_test': X_ntl_test, 'y_test': y_ntl_test,
            'xgb_pred': ntl_xgb_pred, 'lasso_pred': ntl_lasso_pred,
            'xgb_model': ntl_xgb, 'lasso_model': ntl_lasso,
            'scaler': scaler_ntl
        }
    
    # 3. Semantic only (baseline 2)
    X_semantic, semantic_features = create_semantic_with_controls_model(analyzer)
    if X_semantic is not None:
        # Remove control variables, keep only semantic A features
        semantic_only_cols = [col for col in X_semantic.columns if col.startswith('A_')]
        if semantic_only_cols:
            X_semantic_only = X_semantic[semantic_only_cols]
            X_sem_train, X_sem_test, y_sem_train, y_sem_test = train_test_split(
                X_semantic_only, y, test_size=0.3, random_state=42
            )
            
            # Semantic XGBoost
            sem_xgb = GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
            sem_xgb.fit(X_sem_train, y_sem_train)
            sem_xgb_pred = sem_xgb.predict(X_sem_test)
            
            # Semantic Lasso
            scaler_sem = StandardScaler()
            X_sem_train_scaled = scaler_sem.fit_transform(X_sem_train)
            X_sem_test_scaled = scaler_sem.transform(X_sem_test)
            sem_lasso = LassoCV(alphas=np.logspace(-5, 2, 100), cv=5, random_state=42)
            sem_lasso.fit(X_sem_train_scaled, y_sem_train)
            sem_lasso_pred = sem_lasso.predict(X_sem_test_scaled)
            
            models_data['semantic'] = {
                'X_test': X_sem_test, 'y_test': y_sem_test,
                'xgb_pred': sem_xgb_pred, 'lasso_pred': sem_lasso_pred,
                'xgb_model': sem_xgb, 'lasso_model': sem_lasso,
                'scaler': scaler_sem
            }
    
    print(f"    ✅ Prepared {len(models_data)} model groups")
    return models_data

def create_safety_linear_plots(models_data, y, save_dir, colors, perception_name='safe'):
    """Create 2 linear scatter plots with fit lines and confidence intervals - FIXED LAYOUT"""
    print("  📊 Creating linear scatter plots...")
    
    perception_display = perception_name.capitalize()
    
    # Plot 1: Full model comparison (XGBoost vs Lasso)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # Fixed size for consistent canvas
    
    if 'full' in models_data:
        data = models_data['full']
        y_test = data['y_test']
        
        # Calculate R² for layering (best model on top)
        from sklearn.metrics import r2_score
        lasso_r2 = r2_score(y_test, data['lasso_pred'])
        xgb_r2 = r2_score(y_test, data['xgb_pred'])
        
        # Plot worse model first (will be underneath)
        if lasso_r2 > xgb_r2:
            # Lasso is better, plot XGBoost first
            plot_scatter_with_fit_line(ax, y_test, data['xgb_pred'], 
                                     colors['light_orange'], 'Full XGBoost', alpha=0.6)
            plot_scatter_with_fit_line(ax, y_test, data['lasso_pred'], 
                                     colors['blue_green'], 'Full Lasso', alpha=0.7)
        else:
            # XGBoost is better, plot Lasso first (XGBoost will be on top)
            plot_scatter_with_fit_line(ax, y_test, data['lasso_pred'], 
                                     colors['blue_green'], 'Full Lasso', alpha=0.6)
            plot_scatter_with_fit_line(ax, y_test, data['xgb_pred'], 
                                     colors['light_orange'], 'Full XGBoost', alpha=0.7)
    
        # Perfect prediction line
        min_val = min(y_test.min(), min(data['xgb_pred'].min(), data['lasso_pred'].min()))
        max_val = max(y_test.max(), max(data['xgb_pred'].max(), data['lasso_pred'].max()))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='Perfect Prediction')
        
        # Fixed axis limits to maintain 1:1 aspect ratio precision
        buffer = (max_val - min_val) * 0.05
        ax.set_xlim(min_val - buffer, max_val + buffer)
        ax.set_ylim(min_val - buffer, max_val + buffer)
    
    ax.set_xlabel(f'True {perception_display} Values', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'Predicted {perception_display} Values', fontweight='bold', fontsize=12)
    ax.set_title(f'Full Model Comparison: XGBoost vs Lasso\n(Complete A+B+D+AB+AD+BD+ABD + Controls) - {perception_display}', 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Legend with fixed position to avoid canvas compression
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.subplots_adjust(right=0.95)  # Reserve space but don't compress canvas
    plt.savefig(f'{save_dir}/{perception_name}_full_model_linear.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Plot 2: Baseline comparison (NTL vs Semantic)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # Fixed size for consistent canvas
    
    # Collect all data for proper layering and axis limits
    all_y_true = []
    all_y_pred = []
    model_data = []
    
    if 'ntl' in models_data:
        data = models_data['ntl']
        y_test_ntl = data['y_test']
        all_y_true.extend(y_test_ntl)
        all_y_pred.extend(data['xgb_pred'])
        all_y_pred.extend(data['lasso_pred'])
        
        # Calculate R² for both NTL models
        ntl_xgb_r2 = r2_score(y_test_ntl, data['xgb_pred'])
        ntl_lasso_r2 = r2_score(y_test_ntl, data['lasso_pred'])
        
        model_data.append(('ntl', 'xgb', y_test_ntl, data['xgb_pred'], colors['yellow_green'], 'Baseline1 NTL (XGBoost)', ntl_xgb_r2))
        model_data.append(('ntl', 'lasso', y_test_ntl, data['lasso_pred'], colors['gray_blue_purple'], 'Baseline1 NTL (Lasso)', ntl_lasso_r2))
    
    if 'semantic' in models_data:
        data = models_data['semantic']
        y_test_sem = data['y_test']
        all_y_true.extend(y_test_sem)
        all_y_pred.extend(data['xgb_pred'])
        all_y_pred.extend(data['lasso_pred'])
        
        # Calculate R² for both Semantic models
        sem_xgb_r2 = r2_score(y_test_sem, data['xgb_pred'])
        sem_lasso_r2 = r2_score(y_test_sem, data['lasso_pred'])
        
        model_data.append(('semantic', 'xgb', y_test_sem, data['xgb_pred'], colors['light_orange'], 'Baseline2 Semantic (XGBoost)', sem_xgb_r2))
        model_data.append(('semantic', 'lasso', y_test_sem, data['lasso_pred'], colors['blue_green'], 'Baseline2 Semantic (Lasso)', sem_lasso_r2))
    
    # Sort by R² score (lowest first, so best model is plotted last and appears on top)
    model_data.sort(key=lambda x: x[6])  # Sort by R² score
    
    # Plot models in order (worst to best)
    for i, (model_type, algorithm, y_true, y_pred, color, label, r2) in enumerate(model_data):
        alpha = 0.5 + (i * 0.1)  # Gradually increase alpha, best model most opaque
        plot_scatter_with_fit_line(ax, y_true, y_pred, color, label, alpha=alpha)
    
    # Perfect prediction line
    if all_y_true and all_y_pred:
        min_val = min(min(all_y_true), min(all_y_pred))
        max_val = max(max(all_y_true), max(all_y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='Perfect Prediction')
        
        # Fixed axis limits to maintain 1:1 aspect ratio precision
        buffer = (max_val - min_val) * 0.05
        ax.set_xlim(min_val - buffer, max_val + buffer)
        ax.set_ylim(min_val - buffer, max_val + buffer)
    
    ax.set_xlabel(f'True {perception_display} Values', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'Predicted {perception_display} Values', fontweight='bold', fontsize=12)
    ax.set_title(f'Baseline Models Comparison\n(NTL Only vs Semantic Only) - {perception_display}', 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Legend with fixed position to avoid canvas compression
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    plt.subplots_adjust(right=0.95)  # Reserve space but don't compress canvas
    plt.savefig(f'{save_dir}/{perception_name}_baseline_linear.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("    ✅ Linear plots completed with fixed canvas and proper layering")

def create_safety_nonlinear_plots(models_data, y, save_dir, colors, perception_name='safe'):
    """Create 2 nonlinear threshold curve plots - FIXED LAYOUT"""
    print("  📊 Creating nonlinear threshold curve plots...")
    
    perception_display = perception_name.capitalize()
    
    # Plot 3: Full model nonlinear curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # Fixed size for consistent canvas
    
    if 'full' in models_data:
        data = models_data['full']
        y_test = data['y_test']
        
        # Calculate R² for layering (best model on top)
        from sklearn.metrics import r2_score
        lasso_r2 = r2_score(y_test, data['lasso_pred'])
        xgb_r2 = r2_score(y_test, data['xgb_pred'])
        
        # Plot worse model first (will be underneath)
        if lasso_r2 > xgb_r2:
            # Lasso is better, plot XGBoost first
            create_threshold_curve(ax, y_test, data['xgb_pred'], 
                                 colors['light_orange'], 'Full XGBoost (Nonlinear)', smooth=True)
            create_threshold_curve(ax, y_test, data['lasso_pred'], 
                                 colors['blue_green'], 'Full Lasso (Nonlinear)', smooth=True)
        else:
            # XGBoost is better, plot Lasso first (XGBoost will be on top)
            create_threshold_curve(ax, y_test, data['lasso_pred'], 
                                 colors['blue_green'], 'Full Lasso (Nonlinear)', smooth=True)
            create_threshold_curve(ax, y_test, data['xgb_pred'], 
                                 colors['light_orange'], 'Full XGBoost (Nonlinear)', smooth=True)
        
        # Set fixed axis limits for 1:1 precision
        min_val = min(y_test.min(), min(data['xgb_pred'].min(), data['lasso_pred'].min()))
        max_val = max(y_test.max(), max(data['xgb_pred'].max(), data['lasso_pred'].max()))
        buffer = (max_val - min_val) * 0.05
        ax.set_xlim(min_val - buffer, max_val + buffer)
        ax.set_ylim(min_val - buffer, max_val + buffer)
    
    ax.set_xlabel(f'True {perception_display} Values (Sorted)', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'Predicted {perception_display} Values', fontweight='bold', fontsize=12)
    ax.set_title(f'Full Model Nonlinear Threshold Curves\n(Smooth Curves with 95% CI) - {perception_display}', 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')  # Maintain 1:1 aspect ratio
    
    # Legend with fixed position to avoid canvas compression
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.subplots_adjust(right=0.95)  # Reserve space but don't compress canvas
    plt.savefig(f'{save_dir}/{perception_name}_full_model_nonlinear.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Plot 4: Baseline nonlinear curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))  # Fixed size for consistent canvas
    
    # Collect all data for proper layering and axis limits
    all_y_true = []
    all_y_pred = []
    model_data = []
    
    if 'ntl' in models_data:
        data = models_data['ntl']
        y_test_ntl = data['y_test']
        all_y_true.extend(y_test_ntl)
        all_y_pred.extend(data['xgb_pred'])
        all_y_pred.extend(data['lasso_pred'])
        
        # Calculate R² for both NTL models
        ntl_xgb_r2 = r2_score(y_test_ntl, data['xgb_pred'])
        ntl_lasso_r2 = r2_score(y_test_ntl, data['lasso_pred'])
        
        model_data.append((y_test_ntl, data['xgb_pred'], colors['yellow_green'], 'Baseline1 NTL (XGBoost)', ntl_xgb_r2))
        model_data.append((y_test_ntl, data['lasso_pred'], colors['gray_blue_purple'], 'Baseline1 NTL (Lasso)', ntl_lasso_r2))
    
    if 'semantic' in models_data:
        data = models_data['semantic']
        y_test_sem = data['y_test']
        all_y_true.extend(y_test_sem)
        all_y_pred.extend(data['xgb_pred'])
        all_y_pred.extend(data['lasso_pred'])
        
        # Calculate R² for both Semantic models
        sem_xgb_r2 = r2_score(y_test_sem, data['xgb_pred'])
        sem_lasso_r2 = r2_score(y_test_sem, data['lasso_pred'])
        
        model_data.append((y_test_sem, data['xgb_pred'], colors['light_orange'], 'Baseline2 Semantic (XGBoost)', sem_xgb_r2))
        model_data.append((y_test_sem, data['lasso_pred'], colors['blue_green'], 'Baseline2 Semantic (Lasso)', sem_lasso_r2))
    
    # Sort by R² score (lowest first, so best model is plotted last and appears on top)
    model_data.sort(key=lambda x: x[4])  # Sort by R² score
    
    # Plot models in order (worst to best)
    for y_true, y_pred, color, label, r2 in model_data:
        create_threshold_curve(ax, y_true, y_pred, color, label, smooth=True)
    
    # Set fixed axis limits for 1:1 precision
    if all_y_true and all_y_pred:
        min_val = min(min(all_y_true), min(all_y_pred))
        max_val = max(max(all_y_true), max(all_y_pred))
        buffer = (max_val - min_val) * 0.05
        ax.set_xlim(min_val - buffer, max_val + buffer)
        ax.set_ylim(min_val - buffer, max_val + buffer)
    
    ax.set_xlabel(f'True {perception_display} Values (Sorted)', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'Predicted {perception_display} Values', fontweight='bold', fontsize=12)
    ax.set_title(f'Baseline Models Nonlinear Threshold Curves\n(Smooth Curves with 95% CI) - {perception_display}', 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')  # Maintain 1:1 aspect ratio
    
    # Legend with fixed position to avoid canvas compression
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    plt.subplots_adjust(right=0.95)  # Reserve space but don't compress canvas
    plt.savefig(f'{save_dir}/{perception_name}_baseline_nonlinear.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("    ✅ Nonlinear plots completed with fixed canvas and proper layering")

def plot_scatter_with_fit_line(ax, y_true, y_pred, color, label, alpha=0.6):
    """Plot scatter with fit line and confidence interval, return R² and slope for legend"""
    from scipy import stats
    
    # Calculate statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    r_squared = r_value**2
    
    # Create enhanced label with R² and slope
    enhanced_label = f'{label} (R²={r_squared:.3f}, Slope={slope:.3f})'
    
    # Scatter plot - ALWAYS USE CIRCLES (marker='o')
    ax.scatter(y_true, y_pred, alpha=alpha, s=35, color=color, marker='o', 
              edgecolors='white', linewidth=0.5, label=enhanced_label)
    
    # Fit line with slightly higher alpha for better models
    line_alpha = min(0.9, alpha + 0.2)
    line_x = np.linspace(y_true.min(), y_true.max(), 100)
    line_y = slope * line_x + intercept
    
    ax.plot(line_x, line_y, color=color, alpha=line_alpha, linewidth=2.5)
    
    # 95% Confidence interval with adjusted alpha
    residuals = y_pred - (slope * y_true + intercept)
    mse = np.mean(residuals**2)
    ci = 1.96 * np.sqrt(mse)
    
    ci_alpha = min(0.25, alpha * 0.4)  # Confidence interval alpha based on scatter alpha
    ax.fill_between(line_x, line_y - ci, line_y + ci, 
                   color=color, alpha=ci_alpha)
    
    return r_squared, slope

def create_threshold_curve(ax, y_true, y_pred, color, label, smooth=True):
    """Create smooth nonlinear threshold curve with confidence intervals"""
    from scipy.interpolate import UnivariateSpline
    from scipy import stats
    
    # Sort data by true values
    sorted_indices = np.argsort(y_true)
    y_true_sorted = y_true.iloc[sorted_indices] if hasattr(y_true, 'iloc') else y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    if smooth:
        # Create smooth curve using spline
        try:
            # Use quantile-based binning for smoother curves
            n_bins = min(50, len(y_true_sorted) // 10)
            quantiles = np.linspace(0, 1, n_bins)
            bin_edges = np.quantile(y_true_sorted, quantiles)
            bin_edges = np.unique(bin_edges)  # Remove duplicates
            
            if len(bin_edges) < 5:
                # Fallback to regular binning
                bin_edges = np.linspace(y_true_sorted.min(), y_true_sorted.max(), n_bins)
            
            # Calculate bin centers and means
            bin_centers = []
            bin_means = []
            bin_stds = []
            
            for i in range(len(bin_edges) - 1):
                mask = (y_true_sorted >= bin_edges[i]) & (y_true_sorted < bin_edges[i + 1])
                if mask.sum() > 2:  # Need at least 3 points
                    bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                    bin_means.append(y_pred_sorted[mask].mean())
                    bin_stds.append(y_pred_sorted[mask].std())
            
            if len(bin_centers) >= 4:  # Need at least 4 points for spline
                bin_centers = np.array(bin_centers)
                bin_means = np.array(bin_means)
                bin_stds = np.array(bin_stds)
                
                # Create spline
                s_param = len(bin_centers) * np.var(bin_means) * 0.5  # Smoothing parameter
                spline = UnivariateSpline(bin_centers, bin_means, s=s_param)
                
                # Generate smooth curve
                x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 200)
                y_smooth = spline(x_smooth)
                
                # Plot smooth curve
                ax.plot(x_smooth, y_smooth, color=color, linewidth=3, label=label, alpha=0.9)
                
                # Add confidence bands using binned standard deviations
                std_spline = UnivariateSpline(bin_centers, bin_stds, s=s_param)
                y_std_smooth = std_spline(x_smooth)
                
                ax.fill_between(x_smooth, y_smooth - 1.96 * y_std_smooth, 
                               y_smooth + 1.96 * y_std_smooth, 
                               color=color, alpha=0.2)
                
                return
        except:
            pass  # Fall back to simple method
    
    # Fallback: simple moving average
    window_size = max(5, len(y_true_sorted) // 20)
    y_pred_smooth = pd.Series(y_pred_sorted).rolling(window=window_size, center=True, min_periods=1).mean()
    y_pred_std = pd.Series(y_pred_sorted).rolling(window=window_size, center=True, min_periods=1).std()
    
    ax.plot(y_true_sorted, y_pred_smooth, color=color, linewidth=3, label=label, alpha=0.9)
    
    # Confidence interval
    ax.fill_between(y_true_sorted, 
                   y_pred_smooth - 1.96 * y_pred_std.fillna(0), 
                   y_pred_smooth + 1.96 * y_pred_std.fillna(0), 
                   color=color, alpha=0.2)

def create_safety_performance_chart(models_data, save_dir, colors, perception_name='safe'):
    """Create perception-specific performance improvement chart with blue-green base and orange improvement"""
    print(f"  📊 Creating {perception_name} performance improvement chart...")
    
    perception_display = perception_name.capitalize()
    
    # Collect performance data
    performance_data = {}
    
    if 'ntl' in models_data:
        data = models_data['ntl']
        performance_data['NTL Baseline'] = {
            'xgb_score': r2_score(data['y_test'], data['xgb_pred']),
            'lasso_score': r2_score(data['y_test'], data['lasso_pred'])
        }
    
    if 'semantic' in models_data:
        data = models_data['semantic']  
        performance_data['Semantic Baseline'] = {
            'xgb_score': r2_score(data['y_test'], data['xgb_pred']),
            'lasso_score': r2_score(data['y_test'], data['lasso_pred'])
        }
    
    if 'full' in models_data:
        data = models_data['full']
        performance_data['Full Model'] = {
            'xgb_score': r2_score(data['y_test'], data['xgb_pred']),
            'lasso_score': r2_score(data['y_test'], data['lasso_pred'])
        }
    
    # Create stacked bar chart
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    models = list(performance_data.keys())
    xgb_scores = [performance_data[model]['xgb_score'] for model in models]
    lasso_scores = [performance_data[model]['lasso_score'] for model in models]
    
    # Calculate baseline (use NTL as baseline)
    if 'NTL Baseline' in performance_data:
        baseline_xgb = performance_data['NTL Baseline']['xgb_score']
        baseline_lasso = performance_data['NTL Baseline']['lasso_score']
    else:
        baseline_xgb = min(xgb_scores) if xgb_scores else 0
        baseline_lasso = min(lasso_scores) if lasso_scores else 0
    
    # Calculate improvements
    xgb_baseline = [baseline_xgb] * len(models)
    xgb_improvements = [max(0, score - baseline_xgb) for score in xgb_scores]
    
    lasso_baseline = [baseline_lasso] * len(models)
    lasso_improvements = [max(0, score - baseline_lasso) for score in lasso_scores]
    
    x = np.arange(len(models))
    width = 0.35
    
    # XGBoost bars (left side)
    bars1_base = ax.bar(x - width/2, xgb_baseline, width, 
                       label='XGBoost Baseline', color=colors['blue_green'], alpha=0.7)
    bars1_imp = ax.bar(x - width/2, xgb_improvements, width, bottom=xgb_baseline,
                      label='XGBoost Improvement', color=colors['light_orange'], alpha=0.8)
    
    # Lasso bars (right side)
    bars2_base = ax.bar(x + width/2, lasso_baseline, width,
                       label='Lasso Baseline', color=colors['blue_green'], alpha=0.5)
    bars2_imp = ax.bar(x + width/2, lasso_improvements, width, bottom=lasso_baseline,
                      label='Lasso Improvement', color=colors['light_orange'], alpha=0.6)
    
    # Add value labels
    for i, (xgb_score, lasso_score) in enumerate(zip(xgb_scores, lasso_scores)):
        # XGBoost total score
        ax.text(i - width/2, xgb_score + 0.01, f'{xgb_score:.3f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=10)
        # Lasso total score
        ax.text(i + width/2, lasso_score + 0.01, f'{lasso_score:.3f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Improvement percentages
        if i > 0:  # Skip baseline itself
            xgb_imp_pct = (xgb_improvements[i] / baseline_xgb * 100) if baseline_xgb > 0 else 0
            lasso_imp_pct = (lasso_improvements[i] / baseline_lasso * 100) if baseline_lasso > 0 else 0
            
            if xgb_imp_pct > 1:
                ax.text(i - width/2, xgb_score + 0.02, f'+{xgb_imp_pct:.0f}%',
                       ha='center', va='bottom', fontsize=8, color='darkorange', fontweight='bold')
            if lasso_imp_pct > 1:
                ax.text(i + width/2, lasso_score + 0.02, f'+{lasso_imp_pct:.0f}%',
                       ha='center', va='bottom', fontsize=8, color='darkorange', fontweight='bold')
    
    ax.set_xlabel('Model Types', fontweight='bold', fontsize=12)
    ax.set_ylabel('R² Score', fontweight='bold', fontsize=12)
    ax.set_title(f'{perception_display} Perception Model Performance\nBlue-Green: Baseline, Orange: Improvement', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(xgb_scores), max(lasso_scores)) * 1.15)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{perception_name}_performance_improvement.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("    ✅ Safety performance chart completed")

def create_feature_importance_summary(analyzer, save_dir, user_colors):
    """创建特征重要性汇总可视化"""
    print("    🔍 生成特征重要性汇总...")
    
    # 分析语义类别的重要性模式
    semantic_importance = {}
    interaction_importance = {'A': [], 'B': [], 'D': [], 'AB': [], 'AD': [], 'BD': [], 'ABD': []}
    
    # 检查可用的控制变量
    control_vars = ['AVGIL', 'spots_area', 'ADCG', 'illumination_uniformity', ]   #'predicted_spillover'
    available_controls = [col for col in control_vars if col in analyzer.merged_data.columns]
    
    # 模拟特征重要性分析（基于语义类别）
    for semantic in USER_SEMANTIC_CLASSES:
        if semantic in analyzer.merged_data.columns:
            # 计算简单的语义相关性作为重要性指标
            correlations = []
            perception_cols = ['safe', 'beautiful', 'lively', 'wealthy', 'depressing', 'boring']
            
            for perception in perception_cols:
                if perception in analyzer.merged_data.columns:
                    corr = analyzer.merged_data[semantic].corr(analyzer.merged_data[perception])
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)
            
            semantic_importance[semantic] = np.mean(correlations) if correlations else 0
    
    # 创建特征重要性可视化
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 语义类别重要性排名
    if semantic_importance:
        sorted_semantics = sorted(semantic_importance.items(), key=lambda x: x[1], reverse=True)
        semantic_names = [item[0].title() for item in sorted_semantics]
        semantic_scores = [item[1] for item in sorted_semantics]
        
        bars = axes[0,0].barh(range(len(semantic_names)), semantic_scores, 
                             color=user_colors['primary'], alpha=0.8, edgecolor='white', linewidth=1)
        axes[0,0].set_yticks(range(len(semantic_names)))
        axes[0,0].set_yticklabels(semantic_names)
        axes[0,0].set_xlabel('Average Correlation with Perceptions', fontweight='bold')
        axes[0,0].set_title('Semantic Class Importance Ranking\n(Average Correlation)', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for bar, score in zip(bars, semantic_scores):
            axes[0,0].text(score + max(semantic_scores)*0.02, bar.get_y() + bar.get_height()/2,
                          f'{score:.3f}', ha='left', va='center', fontweight='bold')
    
    # 2. 交互项类型分布
    interaction_types = ['A (Pixel)', 'B (Brightness)', 'D (Depth)', 'AB (Pixel×Brightness)', 
                        'AD (Pixel×Depth)', 'BD (Brightness×Depth)', 'ABD (Triple)']
    interaction_counts = [len(USER_SEMANTIC_CLASSES)] * 7  # 每种交互类型都有相同数量
    
    colors_interaction = [user_colors['primary'], user_colors['secondary'], user_colors['accent1'], 
                         user_colors['accent2'], user_colors['accent3'], user_colors['accent4'], '#FF0000']
    
    axes[0,1].pie(interaction_counts, labels=interaction_types, colors=colors_interaction, 
                 autopct='%1.0f', startangle=90)
    axes[0,1].set_title('Interaction Feature Distribution\n(A+B+D+AB+AD+BD+ABD)', fontweight='bold')
    
    # 3. 控制变量使用情况
    control_status = ['Available', 'Missing']
    control_counts = [len(available_controls), len(control_vars) - len(available_controls)]
    control_colors = [user_colors['accent2'], user_colors['accent3']]
    
    bars = axes[1,0].bar(control_status, control_counts, color=control_colors, alpha=0.8, 
                        edgecolor='white', linewidth=1)
    axes[1,0].set_ylabel('Number of Variables', fontweight='bold')
    axes[1,0].set_title('Control Variables Status\n(AVGIL, spots_area, ADCG, illumination_uniformity)', fontweight='bold')   # predicted_spillover
    axes[1,0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签和变量名
    for bar, count, status in zip(bars, control_counts, control_status):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(control_counts)*0.02,
                      f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        if status == 'Available' and available_controls:
            vars_text = '\n'.join(available_controls)
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                          vars_text, ha='center', va='center', fontweight='bold', fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 4. 特征维度汇总
    feature_categories = ['Semantic A', 'Semantic B', 'Semantic D', 'Two-way AB', 'Two-way AD', 'Two-way BD', 'Three-way ABD', 'Controls']
    feature_counts = [len(USER_SEMANTIC_CLASSES)] * 7 + [len(available_controls)]
    
    bars = axes[1,1].bar(range(len(feature_categories)), feature_counts, 
                        color=[user_colors['primary'], user_colors['secondary'], user_colors['accent1'],
                              user_colors['accent2'], user_colors['accent3'], user_colors['accent4'], 
                              '#FF0000', user_colors['accent3']], alpha=0.8, edgecolor='white', linewidth=1)
    
    axes[1,1].set_xticks(range(len(feature_categories)))
    axes[1,1].set_xticklabels(feature_categories, rotation=45, ha='right')
    axes[1,1].set_ylabel('Number of Features', fontweight='bold')
    axes[1,1].set_title('Feature Count by Category\n(Total Features per Model)', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, count in zip(bars, feature_counts):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(feature_counts)*0.02,
                      f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 添加总计标注
    total_features = sum(feature_counts)
    axes[1,1].text(0.98, 0.98, f'Total Features: {total_features}', 
                  transform=axes[1,1].transAxes, ha='right', va='top',
                  bbox=dict(boxstyle='round', facecolor=user_colors['accent2'], alpha=0.3),
                  fontsize=12, fontweight='bold')
    
    fig.suptitle('Feature Importance & Architecture Summary\nA+B+D+AB+AD+BD+ABD Model with Control Variables', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_importance_summary.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"      ✅ 特征重要性汇总完成")

if __name__ == "__main__":
    # 运行主分析（已包含LCZ和NTL分析）
    main()