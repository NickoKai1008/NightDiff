#!/usr/bin/env python3
"""
现场验证数据分析：回归和残差分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# 尝试导入地图相关库
try:
    import contextily as ctx
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_MAP_LIBS = True
except ImportError:
    HAS_MAP_LIBS = False
    print("⚠ 警告: contextily 或 geopandas 未安装，将无法添加OSM底图。")
    print("   安装命令: pip install contextily geopandas")

def safe_log_transform(data, add_constant=1):
    """安全的对数变换"""
    return np.log(data + add_constant)

def calculate_point_density(x, y, method='kde'):
    """计算点的密度
    
    Parameters:
    -----------
    x, y : array-like
        点的坐标
    method : str
        'kde' 使用核密度估计，'nearest' 使用最近邻距离
    
    Returns:
    --------
    density : array
        每个点的密度值（归一化到0-1）
    """
    from scipy.spatial.distance import cdist
    
    if method == 'kde':
        try:
            from scipy.stats import gaussian_kde
            # 使用KDE计算密度
            kde = gaussian_kde(np.vstack([x, y]))
            density = kde(np.vstack([x, y]))
            # 归一化到0-1
            density = (density - density.min()) / (density.max() - density.min() + 1e-10)
        except:
            # 如果KDE失败，使用最近邻方法
            method = 'nearest'
    
    if method == 'nearest':
        # 使用最近邻距离的倒数作为密度
        points = np.column_stack([x, y])
        # 计算到第k个最近邻的距离（k=5）
        k = min(5, len(points) - 1)
        distances = []
        for i, point in enumerate(points):
            dists = np.sqrt(np.sum((points - point)**2, axis=1))
            dists = np.sort(dists)[1:k+1]  # 排除自身，取前k个
            distances.append(np.mean(dists))
        distances = np.array(distances)
        # 距离越小，密度越大，所以取倒数
        density = 1 / (distances + 1e-10)
        # 归一化到0-1
        density = (density - density.min()) / (density.max() - density.min() + 1e-10)
    
    return density

def create_density_colormap():
    """返回viridis颜色映射：密度高->viridis黄色端，密度低->viridis紫色端"""
    # 直接使用matplotlib的viridis colormap
    import matplotlib.cm as cm
    return cm.get_cmap('viridis')

def calculate_confidence_interval(x, y, model, confidence=0.95):
    """计算回归线的置信区间"""
    n = len(x)
    x_pred = np.linspace(x.min(), x.max(), 100)
    y_pred = model.predict(x_pred.reshape(-1, 1))
    
    # 计算残差
    y_fitted = model.predict(x.reshape(-1, 1))
    residuals = y - y_fitted
    mse = np.mean(residuals**2)
    
    # 计算标准误差
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean)**2)
    se = np.sqrt(mse * (1/n + (x_pred - x_mean)**2 / sxx))
    
    # 计算置信区间
    alpha = 1 - confidence
    from scipy import stats
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    ci = t_val * se
    
    return x_pred, y_pred, ci

def plot_regression_analysis(df, x_col, y_col, title, filename):
    """绘制回归分析图"""
    # 准备数据
    data = df[[x_col, y_col, 'ID']].dropna()
    x = safe_log_transform(data[x_col].values)
    y = safe_log_transform(data[y_col].values)
    
    # 计算点的密度
    density = calculate_point_density(x, y, method='kde')
    cmap_density = create_density_colormap()
    
    # 创建图形（只显示一个图）
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    scatter = ax.scatter(x, y, c=density, cmap=cmap_density, alpha=0.7, s=150, edgecolors='white', linewidth=0.5)
    
    # 拟合回归模型
    X = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # 绘制回归线
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, 'red', linewidth=2, alpha=0.8, label='Linear Fit')
    
    # 添加置信区间
    try:
        x_pred, y_pred_ci, ci = calculate_confidence_interval(x, y, model)
        ax.fill_between(x_pred, y_pred_ci - ci, y_pred_ci + ci, 
                        color='red', alpha=0.2, label='95% CI')
    except:
        pass
    
    ax.set_xlabel(f'{x_col} (Log)', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'{y_col} (Log)', fontsize=16, fontweight='bold')
    ax.set_title(f'{title}\nR² = {r2:.4f}, n = {len(x)}', fontsize=18, fontweight='bold')
    
    # 添加图例
    ax.legend(loc='upper left', fontsize=14)
    
    # 添加颜色条（密度）
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Point Density (High=Yellow, Low=Purple)', fontsize=12, fontweight='bold')
    
    # 美化图形
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ {title}: R² = {r2:.4f}, n = {len(x)}")
    
    return {
        'r2': r2,
        'n': len(x),
        'model': model,
        'x': x,
        'y': y
    }

def analyze_residuals_spatial(df, x_col, y_col, title_prefix):
    """残差空间分析"""
    print(f"\n{'='*60}")
    print(f"残差空间分析: {title_prefix}")
    print('='*60)
    
    # 准备数据（包含DN和LV列，如果存在）
    required_cols = [x_col, y_col, 'ID', 'X', 'Y']
    optional_cols = ['DN', 'LV']
    # 去重，避免x_col或y_col已经在required_cols中时重复添加
    all_cols = required_cols + [col for col in optional_cols if col not in required_cols]
    available_cols = [col for col in all_cols if col in df.columns]
    
    # 只对必需的列进行dropna，确保x和y的长度一致
    data = df[available_cols].dropna(subset=[x_col, y_col, 'ID', 'X', 'Y']).copy()
    
    # 再次检查，确保x_col和y_col列中没有NaN（虽然dropna应该已经处理了）
    mask = data[[x_col, y_col]].notna().all(axis=1)
    data = data[mask].copy()
    
    # 确保x和y的长度一致
    x_values = data[x_col].values
    y_values = data[y_col].values
    
    # 检查长度
    if len(x_values) != len(y_values):
        print(f"  ⚠ 错误: x和y的长度不一致 (x={len(x_values)}, y={len(y_values)})")
        print(f"  x_col={x_col}, y_col={y_col}")
        print(f"  数据形状: {data.shape}")
        raise ValueError(f"数据长度不一致: x={len(x_values)}, y={len(y_values)}")
    
    x = safe_log_transform(x_values)
    y = safe_log_transform(y_values)
    
    # 拟合模型
    X = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # 残差计算：残差 = 真实值 - 预测值
    # 正残差（红色）：真实值 > 预测值，模型低估了真实值
    # 负残差（蓝色）：真实值 < 预测值，模型高估了真实值
    residuals_log = y - y_pred  # 残差（log空间）
    
    # 获取原始y值（用于MAE计算）
    y_orig = data[y_col].values
    
    # 转换回原始空间（用于解释）
    y_pred_orig = np.exp(y_pred)
    residuals_orig = y_orig - y_pred_orig
    
    # 创建数据框（保留所有列，包括DN和LV）
    df_clean = data.copy()
    df_clean['residual_log'] = residuals_log
    df_clean['residual_orig'] = residuals_orig
    df_clean['y_pred_log'] = y_pred
    df_clean['y_pred_orig'] = y_pred_orig
    df_clean['y_clean'] = y  # log空间的y值
    
    # 计算每个点的MAE（绝对残差）
    df_clean['MAE_log'] = np.abs(residuals_log)
    df_clean['MAE_orig'] = np.abs(residuals_orig)
    
    # 输出包含残差、MAE和经纬度坐标的CSV
    output_columns = ['ID', 'X', 'Y', 'residual_log', 'residual_orig', 
                     'MAE_log', 'MAE_orig', 'y_pred_log', 'y_pred_orig']
    
    # 添加原始变量列（如果存在）
    if x_col in df_clean.columns:
        output_columns.append(x_col)
    if y_col in df_clean.columns:
        output_columns.append(y_col)
    
    # 添加DN和LV列（如果存在）
    if 'DN' in df_clean.columns:
        output_columns.append('DN')
    if 'LV' in df_clean.columns:
        output_columns.append('LV')
    
    # 只选择存在的列
    available_columns = [col for col in output_columns if col in df_clean.columns]
    df_output = df_clean[available_columns].copy()
    
    # 保存CSV
    filename = f'residual_data_{title_prefix.replace(" ", "_").replace("/", "_")}.csv'
    df_output.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✓ 残差数据CSV已保存: {filename} (共 {len(df_output)} 条记录)")
    
    # 1. 残差地图
    plot_residual_map(df_clean, title_prefix, x_col)
    
    # 2. 残差密度栅格化
    plot_residual_density(df_clean, title_prefix, x_col)
    
    # 3. 按DN level和LV分组的统计
    residual_stats = calculate_residual_stats_by_group(df_clean, title_prefix)
    
    # 4. 小提琴图
    plot_residual_violin(df_clean, title_prefix, x_col)
    
    return df_clean, residual_stats

def plot_residual_map(df_clean, title_prefix, x_col):
    """绘制残差地图（带OSM底图）"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 创建颜色映射：残差 = 真实值 - 预测值
    # 红色(正残差) = 真实值 > 预测值 = 模型低估了真实值 (Underestimated)
    # 蓝色(负残差) = 真实值 < 预测值 = 模型高估了真实值 (Overestimated)
    # 灰色 = 残差接近0
    residuals = df_clean['residual_log'].values
    
    # 归一化残差到[-1, 1]范围
    max_abs_residual = np.max(np.abs(residuals))
    if max_abs_residual > 0:
        normalized_residuals = residuals / max_abs_residual
    else:
        normalized_residuals = residuals
    
    # 根据残差大小调整点的大小
    sizes = 12 + 8 * np.abs(normalized_residuals)
    
    # 如果可以使用地图库，添加OSM底图
    if HAS_MAP_LIBS:
        try:
            # 创建GeoDataFrame（WGS84坐标系）
            geometry = [Point(xy) for xy in zip(df_clean['X'], df_clean['Y'])]
            gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs='EPSG:4326')
            
            # 转换为Web Mercator投影（OSM底图使用的坐标系）
            gdf_mercator = gdf.to_crs(epsg=3857)
            
            # 计算地图边界（添加一些边距）
            bounds = gdf_mercator.total_bounds
            margin_x = (bounds[2] - bounds[0]) * 0.1
            margin_y = (bounds[3] - bounds[1]) * 0.1
            xlim = [bounds[0] - margin_x, bounds[2] + margin_x]
            ylim = [bounds[1] - margin_y, bounds[3] + margin_y]
            
            # 设置坐标轴范围
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
            # 添加OSM底图（黑白灰风格，更透明）
            try:
                # 尝试使用黑白灰风格的底图
                # 优先使用Stamen.Toner（黑白），如果失败则使用CartoDB.Positron（浅灰白）
                try:
                    ctx.add_basemap(ax, 
                                  crs=gdf_mercator.crs,
                                  source=ctx.providers.Stamen.Toner,
                                  alpha=0.4,  # 更透明，让残差点更突出
                                  zoom='auto')
                    print("  ✓ OSM底图已添加（黑白风格，高透明度）")
                except:
                    # 如果Stamen.Toner不可用，使用CartoDB.Positron（浅灰白风格）
                    try:
                        ctx.add_basemap(ax, 
                                      crs=gdf_mercator.crs,
                                      source=ctx.providers.CartoDB.Positron,
                                      alpha=0.4,  # 更透明，让残差点更突出
                                      zoom='auto')
                        print("  ✓ OSM底图已添加（灰白风格，高透明度）")
                    except:
                        # 最后尝试Stamen.TonerLite（浅色黑白）
                        ctx.add_basemap(ax, 
                                      crs=gdf_mercator.crs,
                                      source=ctx.providers.Stamen.TonerLite,
                                      alpha=0.4,  # 更透明，让残差点更突出
                                      zoom='auto')
                        print("  ✓ OSM底图已添加（浅色黑白风格，高透明度）")
            except Exception as e:
                print(f"  ⚠ 无法加载OSM底图: {e}")
                # 如果无法加载底图，继续使用普通散点图
            
            # 在Web Mercator坐标系中绘制散点
            scatter = ax.scatter(gdf_mercator.geometry.x, 
                               gdf_mercator.geometry.y,
                               c=normalized_residuals, 
                               s=sizes,
                               cmap='RdBu_r', 
                               alpha=0.7, 
                               edgecolors='black', 
                               linewidth=0.3,
                               vmin=-1, vmax=1,
                               zorder=10)  # 确保散点在地图上方
            
            ax.set_xlabel('Longitude (Web Mercator)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Latitude (Web Mercator)', fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            
        except Exception as e:
            print(f"  ⚠ 地图处理出错，使用普通散点图: {e}")
            # 如果出错，回退到普通散点图
            scatter = ax.scatter(df_clean['X'], df_clean['Y'], 
                               c=normalized_residuals, 
                               s=sizes,
                               cmap='RdBu_r', 
                               alpha=0.7, 
                               edgecolors='black', 
                               linewidth=0.3,
                               vmin=-1, vmax=1)
            ax.set_xlabel('Longitude (WGS84)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Latitude (WGS84)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    else:
        # 如果没有地图库，使用普通散点图
        scatter = ax.scatter(df_clean['X'], df_clean['Y'], 
                           c=normalized_residuals, 
                           s=sizes,
                           cmap='RdBu_r', 
                           alpha=0.7, 
                           edgecolors='black', 
                           linewidth=0.3,
                           vmin=-1, vmax=1)
        ax.set_xlabel('Longitude (WGS84)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latitude (WGS84)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    ax.set_title(f'{title_prefix} - Residual Map\nRed: Underestimated (真实值>预测值), Blue: Overestimated (真实值<预测值), Gray: Near Zero', 
                fontsize=16, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized Residual (Log Space)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    filename = f'residual_map_{title_prefix.replace(" ", "_").replace("/", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 残差地图已保存: {filename}")

def plot_residual_density(df_clean, title_prefix, x_col):
    """绘制残差密度栅格化结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 准备栅格化数据
    x_coords = df_clean['X'].values
    y_coords = df_clean['Y'].values
    residuals = df_clean['residual_log'].values
    
    # 创建网格
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # 扩展边界
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    # 创建网格
    grid_size = 50
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # 使用高斯核密度估计进行栅格化
    from scipy.stats import gaussian_kde
    
    # 残差密度
    try:
        kde_residual = gaussian_kde(np.vstack([x_coords, y_coords]), weights=residuals)
        zi_residual = kde_residual(np.vstack([xi_grid.ravel(), yi_grid.ravel()]))
        zi_residual = zi_residual.reshape(xi_grid.shape)
        
        # 平滑处理
        zi_residual = gaussian_filter(zi_residual, sigma=1.5)
        
        im1 = ax1.contourf(xi_grid, yi_grid, zi_residual, levels=20, cmap='RdBu_r', alpha=0.8)
        ax1.scatter(x_coords, y_coords, c=residuals, s=20, cmap='RdBu_r', 
                   alpha=0.6, edgecolors='black', linewidth=0.3, vmin=residuals.min(), vmax=residuals.max())
        ax1.set_xlabel('Longitude (WGS84)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Latitude (WGS84)', fontsize=14, fontweight='bold')
        ax1.set_title('Residual Density (KDE)', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Residual Density')
    except:
        # 如果KDE失败，使用简单的网格统计
        zi_residual = np.zeros_like(xi_grid)
        for i in range(len(xi)-1):
            for j in range(len(yi)-1):
                mask = (x_coords >= xi[i]) & (x_coords < xi[i+1]) & \
                       (y_coords >= yi[j]) & (y_coords < yi[j+1])
                if np.sum(mask) > 0:
                    zi_residual[j, i] = np.mean(residuals[mask])
        
        zi_residual = gaussian_filter(zi_residual, sigma=1.5)
        im1 = ax1.contourf(xi_grid, yi_grid, zi_residual, levels=20, cmap='RdBu_r', alpha=0.8)
        ax1.scatter(x_coords, y_coords, c=residuals, s=20, cmap='RdBu_r', 
                   alpha=0.6, edgecolors='black', linewidth=0.3, vmin=residuals.min(), vmax=residuals.max())
        ax1.set_xlabel('Longitude (WGS84)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Latitude (WGS84)', fontsize=14, fontweight='bold')
        ax1.set_title('Residual Density (Grid Average)', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Mean Residual')
    
    # 残差绝对值密度
    abs_residuals = np.abs(residuals)
    try:
        kde_abs = gaussian_kde(np.vstack([x_coords, y_coords]), weights=abs_residuals)
        zi_abs = kde_abs(np.vstack([xi_grid.ravel(), yi_grid.ravel()]))
        zi_abs = zi_abs.reshape(xi_grid.shape)
        zi_abs = gaussian_filter(zi_abs, sigma=1.5)
        
        im2 = ax2.contourf(xi_grid, yi_grid, zi_abs, levels=20, cmap='YlOrRd', alpha=0.8)
        ax2.scatter(x_coords, y_coords, c=abs_residuals, s=20, cmap='YlOrRd', 
                   alpha=0.6, edgecolors='black', linewidth=0.3)
        ax2.set_xlabel('Longitude (WGS84)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Latitude (WGS84)', fontsize=14, fontweight='bold')
        ax2.set_title('Absolute Residual Density (KDE)', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Absolute Residual Density')
    except:
        zi_abs = np.zeros_like(xi_grid)
        for i in range(len(xi)-1):
            for j in range(len(yi)-1):
                mask = (x_coords >= xi[i]) & (x_coords < xi[i+1]) & \
                       (y_coords >= yi[j]) & (y_coords < yi[j+1])
                if np.sum(mask) > 0:
                    zi_abs[j, i] = np.mean(abs_residuals[mask])
        
        zi_abs = gaussian_filter(zi_abs, sigma=1.5)
        im2 = ax2.contourf(xi_grid, yi_grid, zi_abs, levels=20, cmap='YlOrRd', alpha=0.8)
        ax2.scatter(x_coords, y_coords, c=abs_residuals, s=20, cmap='YlOrRd', 
                   alpha=0.6, edgecolors='black', linewidth=0.3)
        ax2.set_xlabel('Longitude (WGS84)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Latitude (WGS84)', fontsize=14, fontweight='bold')
        ax2.set_title('Absolute Residual Density (Grid Average)', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Mean Absolute Residual')
    
    plt.tight_layout()
    filename = f'residual_density_{title_prefix.replace(" ", "_").replace("/", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 残差密度图已保存: {filename}")

def calculate_residual_stats_by_group(df_clean, title_prefix):
    """按DN level和LV分组计算残差统计"""
    stats_list = []
    
    # 按DN分组
    if 'DN' in df_clean.columns:
        dn_groups = df_clean.groupby('DN')
        for dn, group in dn_groups:
            mean_residual = group['residual_log'].mean()
            mae = mean_absolute_error(group['y_pred_log'], group['y_clean'])
            stats_list.append({
                'Group_Type': 'DN_Level',
                'Group_Value': dn,
                'Mean_Residual': mean_residual,
                'MAE': mae,
                'N': len(group)
            })
    
    # 按LV分组（作为POI type）
    if 'LV' in df_clean.columns:
        lv_groups = df_clean.groupby('LV')
        for lv, group in lv_groups:
            if pd.notna(lv):
                mean_residual = group['residual_log'].mean()
                mae = mean_absolute_error(group['y_pred_log'], group['y_clean'])
                stats_list.append({
                    'Group_Type': 'POI_Type_LV',
                    'Group_Value': lv,
                    'Mean_Residual': mean_residual,
                    'MAE': mae,
                    'N': len(group)
                })
    
    stats_df = pd.DataFrame(stats_list)
    
    # 打印统计结果
    print(f"\n按分组统计的残差结果:")
    print('='*80)
    print(f"{'Group Type':<15} {'Group Value':<15} {'Mean Residual':<15} {'MAE':<15} {'N':<10}")
    print('-'*80)
    for _, row in stats_df.iterrows():
        print(f"{row['Group_Type']:<15} {str(row['Group_Value']):<15} {row['Mean_Residual']:>14.4f} {row['MAE']:>14.4f} {row['N']:>10}")
    
    # 保存到CSV
    filename = f'residual_stats_by_group_{title_prefix.replace(" ", "_").replace("/", "_")}.csv'
    stats_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n✓ 分组统计已保存: {filename}")
    
    return stats_df

def plot_residual_violin(df_clean, title_prefix, x_col):
    """绘制残差小提琴图"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. 按DN分组的残差分布
    if 'DN' in df_clean.columns:
        ax1 = axes[0, 0]
        # 选择样本量足够的DN组
        dn_counts = df_clean['DN'].value_counts()
        valid_dns = dn_counts[dn_counts >= 5].index[:10]  # 最多显示10个组
        df_dn = df_clean[df_clean['DN'].isin(valid_dns)].copy()
        
        if len(df_dn) > 0:
            sns.violinplot(data=df_dn, x='DN', y='residual_log', ax=ax1, palette='Set2')
            ax1.set_xlabel('DN Level', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Residual (Log Space)', fontsize=14, fontweight='bold')
            ax1.set_title('Residual Distribution by DN Level', fontsize=14, fontweight='bold')
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax1.tick_params(axis='x', rotation=45)
    
    # 2. 按LV分组的残差分布
    if 'LV' in df_clean.columns:
        ax2 = axes[0, 1]
        df_lv = df_clean[df_clean['LV'].notna()].copy()
        if len(df_lv) > 0:
            sns.violinplot(data=df_lv, x='LV', y='residual_log', ax=ax2, palette='Set3')
            ax2.set_xlabel('POI Type (LV)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Residual (Log Space)', fontsize=14, fontweight='bold')
            ax2.set_title('Residual Distribution by POI Type (LV)', fontsize=14, fontweight='bold')
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # 3. 残差直方图
    ax3 = axes[1, 0]
    ax3.hist(df_clean['residual_log'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
    ax3.set_xlabel('Residual (Log Space)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax3.set_title('Residual Distribution Histogram', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q图（检验残差正态性）
    ax4 = axes[1, 1]
    stats.probplot(df_clean['residual_log'], dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'residual_violin_{title_prefix.replace(" ", "_").replace("/", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 小提琴图已保存: {filename}")

def main():
    """主函数 - 现场验证数据分析"""
    print("=== 现场验证数据分析 ===")
    
    # 读取数据
    df = pd.read_csv('HK_citywid_onsite_data.csv')
    
    variables = ['Avg_DN', 'Avg_AVGIL', 'DN']
    y_variable = 'lux'
    
    print(f"\n{'='*60}")
    print(f"数据分析: {', '.join(variables)} vs {y_variable}")
    print('='*60)
    
    results = []
    
    for x_var in variables:
        print(f"\n分析 {x_var} vs {y_variable}...")
        
        title = f'{x_var} vs {y_variable}'
        filename = f'{x_var}_vs_{y_variable}.png'
        
        # 回归分析
        result = plot_regression_analysis(df, x_var, y_variable, title, filename)
        result['variable'] = x_var
        results.append(result)
        
        # 残差空间分析
        print(f"\n进行残差空间分析: {title}...")
        df_residual, residual_stats = analyze_residuals_spatial(
            df, x_var, y_variable, title)
    
    # 生成汇总报告
    print(f"\n{'='*60}")
    print("数据分析结果汇总")
    print('='*60)
    print("| 变量     | R²     | n      |")
    print("|----------|--------|--------|")
    
    for result in results:
        print(f"| {result['variable']:8s} | {result['r2']:6.4f} | {result['n']:6d} |")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('analysis_summary.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: analysis_summary.csv")
    
    # 计算平均R²
    avg_r2 = np.mean([r['r2'] for r in results])
    print(f"\n平均R²: {avg_r2:.4f}")

if __name__ == "__main__":
    main()
