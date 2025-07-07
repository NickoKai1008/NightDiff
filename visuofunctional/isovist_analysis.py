#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isovist Analysis Script
Visibility analysis based on building, POI and road data

Features:
1. Generate observation points at 50m intervals along road network
2. Calculate isovist geometry for each point using building edges as obstructions
3. Expand isovist geometry by 3 meters
4. Analyze dominant POI types and diversity (Shannon entropy) within each isovist
5. Output result shapefiles
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union, transform
from shapely.affinity import scale
import math
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def create_isovist_polygon(observer_point, obstacles, max_distance=200, num_rays=72):
    """
    创建isovist多边形
    
    Parameters:
    observer_point: Point - 观测点
    obstacles: GeoDataFrame - 障碍物（建筑物）
    max_distance: float - 最大视距（米）
    num_rays: int - 射线数量
    
    Returns:
    Polygon - isovist多边形
    """
    rays_end_points = []
    
    # 计算射线角度
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    
    for angle in angles:
        # 计算射线终点
        end_x = observer_point.x + max_distance * np.cos(angle)
        end_y = observer_point.y + max_distance * np.sin(angle)
        ray = LineString([observer_point, Point(end_x, end_y)])
        
        # 找到最近的障碍物交点
        min_distance = max_distance
        intersection_point = Point(end_x, end_y)
        
        for _, obstacle in obstacles.iterrows():
            if obstacle.geometry.intersects(ray):
                intersection = ray.intersection(obstacle.geometry)
                if intersection.geom_type == 'Point':
                    dist = observer_point.distance(intersection)
                    if dist < min_distance and dist > 0.1:  # 避免自相交
                        min_distance = dist
                        intersection_point = intersection
                elif intersection.geom_type == 'MultiPoint':
                    for pt in intersection.geoms:
                        dist = observer_point.distance(pt)
                        if dist < min_distance and dist > 0.1:
                            min_distance = dist
                            intersection_point = pt
        
        rays_end_points.append(intersection_point)
    
    # 创建isovist多边形
    if len(rays_end_points) >= 3:
        isovist = Polygon([(pt.x, pt.y) for pt in rays_end_points])
        if not isovist.is_valid:
            isovist = isovist.buffer(0)  # 修复无效几何
        return isovist
    else:
        return None

def generate_points_along_roads(roads_gdf, interval=50):
    """
    沿道路网络生成等间距点
    
    Parameters:
    roads_gdf: GeoDataFrame - 道路数据
    interval: float - 点间距（米）
    
    Returns:
    GeoDataFrame - 生成的点
    """
    points = []
    point_id = 1
    
    for idx, road in roads_gdf.iterrows():
        geom = road.geometry
        
        if geom.geom_type == 'LineString':
            lines = [geom]
        elif geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
        else:
            continue
            
        for line in lines:
            line_length = line.length
            num_points = int(line_length / interval) + 1
            
            for i in range(num_points):
                distance = i * interval
                if distance <= line_length:
                    point = line.interpolate(distance)
                    points.append({
                        'point_id': point_id,
                        'road_id': idx,
                        'distance_along_road': distance,
                        'geometry': point
                    })
                    point_id += 1
    
    return gpd.GeoDataFrame(points, crs=roads_gdf.crs)

def calculate_shannon_entropy(poi_types):
    """
    Calculate Shannon entropy
    
    Parameters:
    poi_types: list - List of POI types
    
    Returns:
    float - Shannon entropy value
    """
    if not poi_types:
        return 0
    
    counts = Counter(poi_types)
    total = len(poi_types)
    
    entropy = 0
    for count in counts.values():
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)
    
    return entropy

def get_dominant_poi_type(poi_types):
    """
    Get dominant POI type
    
    Parameters:
    poi_types: list - List of POI types
    
    Returns:
    str - Dominant POI type
    """
    if not poi_types:
        return "None"
    
    counts = Counter(poi_types)
    return counts.most_common(1)[0][0]

def main():
    print("Starting Isovist analysis...")
    
    # 1. Load data
    print("1. Loading shapefile data...")
    
    # Load building data (EPSG:3857)
    buildings_gdf = gpd.read_file("ad/ad.shp")
    print(f"   Building data: {len(buildings_gdf)} records, CRS: {buildings_gdf.crs}")
    
    # Load POI data (EPSG:4326)
    poi_gdf = gpd.read_file("poi/poi.shp")
    print(f"   POI data: {len(poi_gdf)} records, CRS: {poi_gdf.crs}")
    
    # Load road data (EPSG:4326)
    roads_gdf = gpd.read_file("rd/rd.shp")
    print(f"   Road data: {len(roads_gdf)} records, CRS: {roads_gdf.crs}")
    
    # 2. Unify coordinate system to EPSG:3857 (Mercator projection)
    print("2. Unifying coordinate system to EPSG:3857...")
    
    if poi_gdf.crs != 'EPSG:3857':
        poi_gdf = poi_gdf.to_crs('EPSG:3857')
    
    if roads_gdf.crs != 'EPSG:3857':
        roads_gdf = roads_gdf.to_crs('EPSG:3857')
    
    if buildings_gdf.crs != 'EPSG:3857':
        buildings_gdf = buildings_gdf.to_crs('EPSG:3857')
    
    print("   Coordinate system unification completed")
    
    # 3. Generate observation points along roads
    print("3. Generating observation points along road network (50m intervals)...")
    observation_points = generate_points_along_roads(roads_gdf, interval=50)
    print(f"   Generated observation points: {len(observation_points)}")
    
    # 4. Calculate isovist geometry for each point
    print("4. Computing isovist geometries...")
    
    isovists = []
    
    for idx, point_row in observation_points.iterrows():
        if idx % 100 == 0:
            print(f"   Processing progress: {idx}/{len(observation_points)}")
        
        point = point_row.geometry
        point_id = point_row['point_id']
        
        # Calculate isovist
        isovist = create_isovist_polygon(point, buildings_gdf, max_distance=200, num_rays=72)
        
        if isovist is not None:
            # Expand isovist by 3 meters
            expanded_isovist = isovist.buffer(3)
            
            # Find POI within isovist
            poi_in_isovist = poi_gdf[poi_gdf.geometry.within(expanded_isovist)]
            
            # Get POI type field (assuming field name is 'type' or similar)
            poi_type_column = None
            for col in poi_gdf.columns:
                if col.lower() in ['type', 'category', 'class', 'poi_type', 'function']:
                    poi_type_column = col
                    break
            
            if poi_type_column is None and len(poi_gdf.columns) > 1:
                # If no clear type field found, use first non-geometry field
                poi_type_column = [col for col in poi_gdf.columns if col != 'geometry'][0]
            
            poi_types = []
            if poi_type_column and len(poi_in_isovist) > 0:
                poi_types = poi_in_isovist[poi_type_column].tolist()
            
            # Calculate dominant POI type
            dominant_poi = get_dominant_poi_type(poi_types)
            
            # Calculate Shannon entropy
            shannon_entropy = calculate_shannon_entropy(poi_types)
            
            # Calculate statistics
            poi_count = len(poi_in_isovist)
            unique_poi_types = len(set(poi_types)) if poi_types else 0
            
            isovists.append({
                'point_id': point_id,
                'dominant_poi': dominant_poi,
                'shannon_entropy': shannon_entropy,
                'poi_count': poi_count,
                'unique_poi_types': unique_poi_types,
                'isovist_area': expanded_isovist.area,
                'geometry': expanded_isovist
            })
    
    print(f"   Successfully computed isovists: {len(isovists)}")
    
    # 5. Create result GeoDataFrame
    print("5. Creating result data...")
    
    isovists_gdf = gpd.GeoDataFrame(isovists, crs='EPSG:3857')
    
    # Add corresponding analysis results to observation points
    observation_points_with_results = observation_points.merge(
        isovists_gdf[['point_id', 'dominant_poi', 'shannon_entropy', 'poi_count', 'unique_poi_types', 'isovist_area']], 
        on='point_id', 
        how='left'
    )
    
    # 6. Output results
    print("6. Saving result files...")
    
    # Output isovist geometries
    isovists_gdf.to_file("isovist_polygons.shp", encoding='utf-8')
    print("   Isovist geometries saved: isovist_polygons.shp")
    
    # Output observation points
    observation_points_with_results.to_file("observation_points.shp", encoding='utf-8')
    print("   Observation points saved: observation_points.shp")
    
    # 7. Output statistical report
    print("\n7. Analysis Result Statistics:")
    print(f"   Total observation points: {len(observation_points)}")
    print(f"   Successfully computed isovists: {len(isovists_gdf)}")
    print(f"   Average POI count per isovist: {isovists_gdf['poi_count'].mean():.2f}")
    print(f"   Average Shannon entropy: {isovists_gdf['shannon_entropy'].mean():.3f}")
    print(f"   Average isovist area: {isovists_gdf['isovist_area'].mean():.2f} square meters")
    
    if 'dominant_poi' in isovists_gdf.columns:
        dominant_counts = isovists_gdf['dominant_poi'].value_counts()
        print(f"   Dominant POI type distribution:")
        for poi_type, count in dominant_counts.head(10).items():
            print(f"     {poi_type}: {count} ({count/len(isovists_gdf)*100:.1f}%)")
    
    print("\nIsovist analysis completed!")

if __name__ == "__main__":
    main() 