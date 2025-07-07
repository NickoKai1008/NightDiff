#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isovist Analysis Script - Optimized Version
Advanced visibility analysis script with performance optimizations

Performance Enhancements:
1. Spatial indexing for accelerated queries
2. Batch processing with progress display
3. Memory optimization and error handling
4. Configurable parameters
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, MultiLineString, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
import math
from collections import Counter
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

class IsovistAnalyzer:
    def __init__(self, config=None):
        """
        Initialize Isovist Analyzer
        
        Parameters:
        config: dict - Configuration parameters
        """
        self.config = config or {
            'input_mode': 2,           # Input mode: 1=direct point file, 2=generate from roads
            'point_interval': 50,      # Observation point interval (meters)
            'max_distance': 200,       # Maximum visibility distance (meters)
            'num_rays': 72,           # Number of rays
            'buffer_distance': 3,      # Isovist expansion distance (meters)
            'batch_size': 100,        # Batch processing size
        }
        
        self.buildings_gdf = None
        self.poi_gdf = None
        self.roads_gdf = None
        self.poi_type_column = None
        self.building_tree = None
        
    def load_data(self):
        """Load shapefile data"""
        print("1. Loading shapefile data...")
        
        try:
            # Load building data
            self.buildings_gdf = gpd.read_file("ad/ad.shp")
            print(f"   Building data: {len(self.buildings_gdf)} records, CRS: {self.buildings_gdf.crs}")
            
            # Load POI data
            self.poi_gdf = gpd.read_file("poi/poi.shp")
            print(f"   POI data: {len(self.poi_gdf)} records, CRS: {self.poi_gdf.crs}")
            
            # Load different data based on input mode
            if self.config['input_mode'] == 1:
                # Mode 1: Direct point file
                print("   Input mode: Using pre-segmented point file")
                self.points_gdf = gpd.read_file("pt/pt.shp")
                print(f"   Observation points: {len(self.points_gdf)} records, CRS: {self.points_gdf.crs}")
                self.roads_gdf = None  # No need for road data
            else:
                # Mode 2: Generate points from roads
                print("   Input mode: Generate observation points from road network")
                self.roads_gdf = gpd.read_file("rd/rd.shp")
                print(f"   Road data: {len(self.roads_gdf)} records, CRS: {self.roads_gdf.crs}")
                self.points_gdf = None  # Will be generated later
            
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def preprocess_data(self):
        """Preprocess data: unify coordinate system and build spatial indices"""
        print("2. Preprocessing data...")
        
        # Unify coordinate system to EPSG:3857
        target_crs = 'EPSG:3857'
        
        if self.poi_gdf.crs != target_crs:
            self.poi_gdf = self.poi_gdf.to_crs(target_crs)
        
        if self.buildings_gdf.crs != target_crs:
            self.buildings_gdf = self.buildings_gdf.to_crs(target_crs)
        
        # Process different data based on input mode
        if self.config['input_mode'] == 1:
            # Mode 1: Process point file
            if self.points_gdf.crs != target_crs:
                self.points_gdf = self.points_gdf.to_crs(target_crs)
        else:
            # Mode 2: Process road file
            if self.roads_gdf.crs != target_crs:
                self.roads_gdf = self.roads_gdf.to_crs(target_crs)
        
        print("   Coordinate system unification completed")
        
        # Detect POI type field
        self.poi_type_column = self._detect_poi_type_column()
        if self.poi_type_column:
            print(f"   Detected POI type field: {self.poi_type_column}")
        else:
            print("   Warning: POI type field not found, will use default values")
        
        # Build spatial index for buildings
        print("   Building spatial indices...")
        building_geometries = list(self.buildings_gdf.geometry)
        self.building_tree = STRtree(building_geometries)
        
        # Build spatial index for POI
        self.poi_gdf['poi_index'] = range(len(self.poi_gdf))
        poi_geometries = list(self.poi_gdf.geometry)
        self.poi_tree = STRtree(poi_geometries)
        
    def _detect_poi_type_column(self):
        """Detect POI type field"""
        # Prioritize fclass field
        if 'fclass' in self.poi_gdf.columns:
            return 'fclass'
        
        # Other possible field names
        possible_names = ['type', 'category', 'class', 'poi_type', 'function', 'land_use']
        
        for col in self.poi_gdf.columns:
            if col.lower() in possible_names:
                return col
        
        # If not found, use the first non-geometry field
        non_geom_columns = [col for col in self.poi_gdf.columns if col != 'geometry']
        return non_geom_columns[0] if non_geom_columns else None
    
    def get_observation_points(self):
        """Get observation points"""
        if self.config['input_mode'] == 1:
            # Mode 1: Use point file directly
            print("3. Using pre-segmented observation points...")
            observation_points = self.points_gdf.copy()
            
            # Ensure point_id field exists
            if 'point_id' not in observation_points.columns:
                observation_points['point_id'] = range(1, len(observation_points) + 1)
            
            print(f"   Loaded observation points: {len(observation_points)}")
            return observation_points
        else:
            # Mode 2: Generate points from roads
            print("3. Generating observation points from road network...")
            return self.generate_points_from_roads()
    
    def generate_points_from_roads(self):
        """Generate observation points from roads"""
        points = []
        point_id = 1
        interval = self.config['point_interval']
        
        print(f"   Segmentation interval: {interval} meters")
        
        for idx, road in self.roads_gdf.iterrows():
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
        
        observation_points = gpd.GeoDataFrame(points, crs=self.roads_gdf.crs)
        print(f"   Generated observation points: {len(observation_points)}")
        return observation_points
    
    def create_isovist_optimized(self, observer_point):
        """
        Optimized isovist calculation
        Uses spatial indexing and efficient geometric computations
        """
        max_distance = self.config['max_distance']
        num_rays = self.config['num_rays']
        
        # Create larger query range to ensure all buildings affecting visibility are found
        search_buffer = max_distance * 1.5  # Expand search range
        search_box = box(
            observer_point.x - search_buffer,
            observer_point.y - search_buffer,
            observer_point.x + search_buffer,
            observer_point.y + search_buffer
        )
        
        # Use spatial index to find nearby buildings
        possible_matches_index = list(self.building_tree.query(search_box))
        nearby_buildings = self.buildings_gdf.iloc[possible_matches_index]
        
        # If no nearby buildings found, query with larger range
        if len(nearby_buildings) == 0:
            # Expand search range to 2x maximum visibility distance
            extended_search_box = box(
                observer_point.x - max_distance * 2,
                observer_point.y - max_distance * 2,
                observer_point.x + max_distance * 2,
                observer_point.y + max_distance * 2
            )
            extended_matches_index = list(self.building_tree.query(extended_search_box))
            buildings_to_check = self.buildings_gdf.iloc[extended_matches_index]
        else:
            buildings_to_check = nearby_buildings
        
        rays_end_points = []
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        
        for angle in angles:
            # Calculate ray endpoint
            end_x = observer_point.x + max_distance * np.cos(angle)
            end_y = observer_point.y + max_distance * np.sin(angle)
            ray = LineString([observer_point, Point(end_x, end_y)])
            
            # Find nearest obstacle intersection
            min_distance = max_distance
            intersection_point = Point(end_x, end_y)
            found_intersection = False
            
            for _, building in buildings_to_check.iterrows():
                try:
                    building_geom = building.geometry
                    
                    # Ensure building geometry is valid
                    if building_geom is None or building_geom.is_empty:
                        continue
                    
                    # Check if ray intersects with building
                    if ray.intersects(building_geom):
                        intersection = ray.intersection(building_geom)
                        
                        # Handle different types of intersections
                        intersection_points = []
                        
                        if intersection.geom_type == 'Point':
                            intersection_points = [intersection]
                        elif intersection.geom_type == 'MultiPoint':
                            intersection_points = list(intersection.geoms)
                        elif intersection.geom_type == 'LineString':
                            # Ray coincides with building edge, find nearest point
                            # Use interpolate method to find closest point on line segment to observer
                            line_length = intersection.length
                            if line_length > 0:
                                # Try multiple points on the line segment
                                sample_points = []
                                for ratio in [0, 0.25, 0.5, 0.75, 1.0]:
                                    sample_point = intersection.interpolate(ratio * line_length)
                                    sample_points.append(sample_point)
                                intersection_points = sample_points
                            else:
                                # Line segment length is 0, take endpoints
                                coords = list(intersection.coords)
                                intersection_points = [Point(coord) for coord in coords]
                        elif intersection.geom_type == 'GeometryCollection':
                            # Handle complex intersections
                            for geom in intersection.geoms:
                                if geom.geom_type == 'Point':
                                    intersection_points.append(geom)
                                elif geom.geom_type == 'LineString':
                                    line_length = geom.length
                                    if line_length > 0:
                                        for ratio in [0, 0.5, 1.0]:
                                            sample_point = geom.interpolate(ratio * line_length)
                                            intersection_points.append(sample_point)
                                    else:
                                        coords = list(geom.coords)
                                        intersection_points.extend([Point(coord) for coord in coords])
                        
                        # Find nearest intersection point
                        for pt in intersection_points:
                            if pt is not None and not pt.is_empty:
                                dist = observer_point.distance(pt)
                                # Exclude observer point itself and points too close
                                if 0.5 < dist < min_distance:
                                    min_distance = dist
                                    intersection_point = pt
                                    found_intersection = True
                                    
                except Exception as e:
                    # Skip problematic building geometries
                    continue
            
            rays_end_points.append(intersection_point)
        
        # Create isovist polygon
        try:
            if len(rays_end_points) >= 3:
                isovist = Polygon([(pt.x, pt.y) for pt in rays_end_points])
                if not isovist.is_valid:
                    isovist = isovist.buffer(0)
                return isovist
        except Exception:
            pass
        
        return None
    
    def analyze_poi_in_isovist(self, isovist_geom):
        """Analyze POI within isovist"""
        if isovist_geom is None:
            return "None", 0, 0, 0
        
        # Use spatial index to find POI
        possible_matches_index = list(self.poi_tree.query(isovist_geom))
        nearby_pois = self.poi_gdf.iloc[possible_matches_index]
        
        # Precisely filter POI within isovist
        poi_in_isovist = nearby_pois[nearby_pois.geometry.within(isovist_geom)]
        
        poi_types = []
        if self.poi_type_column and len(poi_in_isovist) > 0:
            poi_types = poi_in_isovist[self.poi_type_column].dropna().tolist()
        
        # Calculate statistics
        dominant_poi = self._get_dominant_poi_type(poi_types)
        shannon_entropy = self._calculate_shannon_entropy(poi_types)
        poi_count = len(poi_in_isovist)
        unique_poi_types = len(set(poi_types)) if poi_types else 0
        
        return dominant_poi, shannon_entropy, poi_count, unique_poi_types
    
    def _calculate_shannon_entropy(self, poi_types):
        """Calculate Shannon entropy"""
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
    
    def _get_dominant_poi_type(self, poi_types):
        """Get dominant POI type"""
        if not poi_types:
            return "None"
        
        counts = Counter(poi_types)
        return counts.most_common(1)[0][0]
    
    def run_analysis(self):
        """Run complete analysis workflow"""
        start_time = time.time()
        
        # 1. Load data
        self.load_data()
        
        # 2. Preprocess
        self.preprocess_data()
        
        # 3. Get observation points
        observation_points = self.get_observation_points()
        
        # 4. Batch compute isovist
        print("4. Computing isovist geometries...")
        batch_size = self.config['batch_size']
        isovists = []
        
        total_points = len(observation_points)
        
        for i in range(0, total_points, batch_size):
            batch_end = min(i + batch_size, total_points)
            batch_points = observation_points.iloc[i:batch_end]
            
            print(f"   Processing batch {i//batch_size + 1}/{(total_points-1)//batch_size + 1} "
                  f"({i+1}-{batch_end}/{total_points})")
            
            for idx, point_row in batch_points.iterrows():
                point = point_row.geometry
                point_id = point_row['point_id']
                
                # Calculate isovist
                isovist = self.create_isovist_optimized(point)
                
                if isovist is not None:
                    # Expand isovist
                    expanded_isovist = isovist.buffer(self.config['buffer_distance'])
                    
                    # Analyze POI
                    dominant_poi, shannon_entropy, poi_count, unique_poi_types = \
                        self.analyze_poi_in_isovist(expanded_isovist)
                    
                    isovists.append({
                        'point_id': point_id,
                        'dominant_poi': dominant_poi,
                        'shannon_entropy': shannon_entropy,
                        'poi_count': poi_count,
                        'unique_poi_types': unique_poi_types,
                        'isovist_area': expanded_isovist.area,
                        'geometry': expanded_isovist
                    })
        
        print(f"   Successfully computed isovist: {len(isovists)}")
        
        # 5. Create result data
        print("5. Creating result data...")
        isovists_gdf = gpd.GeoDataFrame(isovists, crs='EPSG:3857')
        
        # Merge observation points and analysis results
        observation_points_with_results = observation_points.merge(
            isovists_gdf[['point_id', 'dominant_poi', 'shannon_entropy', 
                         'poi_count', 'unique_poi_types', 'isovist_area']], 
            on='point_id', 
            how='left'
        )
        
        # 6. Output results
        self.save_results(isovists_gdf, observation_points_with_results)
        
        # 7. Output statistics
        self.print_statistics(isovists_gdf, start_time)
        
        return isovists_gdf, observation_points_with_results
    
    def save_results(self, isovists_gdf, observation_points_with_results):
        """Save result files"""
        print("6. Saving result files...")
        
        try:
            # Output isovist geometries
            isovists_gdf.to_file("isovist_polygons.shp", encoding='utf-8')
            print("   Isovist geometries saved: isovist_polygons.shp")
            
            # Output observation points
            observation_points_with_results.to_file("observation_points.shp", encoding='utf-8')
            print("   Observation points saved: observation_points.shp")
            
        except Exception as e:
            print(f"   Error saving files: {e}")
    
    def print_statistics(self, isovists_gdf, start_time):
        """Output statistical information"""
        elapsed_time = time.time() - start_time
        
        print(f"\n7. Analysis Result Statistics:")
        print(f"   Processing time: {elapsed_time:.1f} seconds")
        print(f"   Successfully computed isovists: {len(isovists_gdf)}")
        
        if len(isovists_gdf) > 0:
            print(f"   Average POI count per isovist: {isovists_gdf['poi_count'].mean():.2f}")
            print(f"   Average Shannon entropy: {isovists_gdf['shannon_entropy'].mean():.3f}")
            print(f"   Average isovist area: {isovists_gdf['isovist_area'].mean():.2f} square meters")
            
            if 'dominant_poi' in isovists_gdf.columns:
                dominant_counts = isovists_gdf['dominant_poi'].value_counts()
                print(f"   Dominant POI type distribution:")
                for poi_type, count in dominant_counts.head(10).items():
                    print(f"     {poi_type}: {count} ({count/len(isovists_gdf)*100:.1f}%)")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Isovist Analysis Script')
    parser.add_argument('--mode', type=int, choices=[1, 2], default=2, 
                       help='Input mode: 1=use pre-segmented point file(pt/pt.shp), 2=generate from roads(rd/rd.shp)')
    parser.add_argument('--interval', type=int, default=50, help='Observation point interval in meters (only valid for mode 2)')
    parser.add_argument('--max-distance', type=int, default=200, help='Maximum visibility distance in meters')
    parser.add_argument('--num-rays', type=int, default=72, help='Number of rays')
    parser.add_argument('--buffer', type=int, default=3, help='Isovist expansion distance in meters')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch processing size')
    
    args = parser.parse_args()
    
    config = {
        'input_mode': args.mode,
        'point_interval': args.interval,
        'max_distance': args.max_distance,
        'num_rays': args.num_rays,
        'buffer_distance': args.buffer,
        'batch_size': args.batch_size,
    }
    
    print("Starting Isovist Analysis (Optimized Version)...")
    
    # Display configuration information
    mode_desc = "Pre-segmented point file(pt/pt.shp)" if config['input_mode'] == 1 else "Generate from roads(rd/rd.shp)"
    print(f"Input mode: {config['input_mode']} - {mode_desc}")
    
    if config['input_mode'] == 2:
        print(f"Point interval: {config['point_interval']} meters")
    
    print(f"Maximum visibility distance: {config['max_distance']} meters")
    print(f"Number of rays: {config['num_rays']}")
    print(f"Expansion distance: {config['buffer_distance']} meters")
    print(f"Batch size: {config['batch_size']}")
    print()
    
    analyzer = IsovistAnalyzer(config)
    analyzer.run_analysis()
    
    print("\nIsovist analysis completed!")

if __name__ == "__main__":
    main() 