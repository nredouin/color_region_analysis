import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class ClusterComparator:
    """Class for comparing color regions between different skin tone clusters"""
    
    def __init__(self):
        self.delta_e_threshold = 3.0
    
    def compare_two_clusters(self, cluster_results_1, cluster_results_2, cluster_id_1, cluster_id_2, 
                        delta_e_threshold=3.0, top_n=10, metric_column='combined_criteria_pct', 
                        reflect_delta_e_threshold=None):
        """
        Compare two specific clusters using DeltaE analysis with conditional reflect color checking
        
        Args:
            cluster_results_1: First cluster results dict
            cluster_results_2: Second cluster results dict  
            cluster_id_1: First cluster ID
            cluster_id_2: Second cluster ID
            delta_e_threshold: Threshold for main color similarity (backward compatibility)
            top_n: Number of top regions to compare
            metric_column: Column to use for sorting/scoring
            reflect_delta_e_threshold: Threshold for reflect color similarity (defaults to 2x main threshold)
            
        Logic:
            1. Calculate DeltaE for main colors
            2. If main colors are similar (< delta_e_threshold), then calculate reflect DeltaE
            3. Check if reflect colors are also similar (< reflect_delta_e_threshold)
        """
        # Use delta_e_threshold as main_delta_e_threshold for backward compatibility
        main_delta_e_threshold = delta_e_threshold
        
        # Set default reflect threshold as 2x main threshold
        if reflect_delta_e_threshold is None:
            reflect_delta_e_threshold = main_delta_e_threshold * 2.0
        
        # Store both thresholds for later use
        self.main_delta_e_threshold = main_delta_e_threshold
        self.reflect_delta_e_threshold = reflect_delta_e_threshold
        
        print(f"\n=== DEBUGGING CLUSTER COMPARISON ===")
        print(f"Comparing Cluster {cluster_id_1} vs Cluster {cluster_id_2}")
        print(f"Main DeltaE threshold: {main_delta_e_threshold}")
        print(f"Reflect DeltaE threshold: {reflect_delta_e_threshold}")
        print(f"Metric column: {metric_column}")
        
        # Get data for both clusters
        try:
            results_1 = cluster_results_1[cluster_id_1]
            results_2 = cluster_results_2[cluster_id_2]
            print(f"✅ Successfully loaded results for both clusters")
        except KeyError as e:
            raise ValueError(f"Cluster data not found: {e}")
        
        # Get top regions from both clusters
        top_1 = results_1['top_regions'].head(top_n).copy()
        top_2 = results_2['top_regions'].head(top_n).copy()
        
        centers_1 = results_1['cluster_centers'].copy()
        centers_2 = results_2['cluster_centers'].copy()
        
        print(f"Top_1 shape: {top_1.shape}, Top_2 shape: {top_2.shape}")
        print(f"Centers_1 shape: {centers_1.shape}, Centers_2 shape: {centers_2.shape}")
        
        # Debug data structure
        print(f"\n=== CLUSTER {cluster_id_1} DEBUGGING ===")
        self._debug_cluster_data(top_1, centers_1, cluster_id_1)
        
        print(f"\n=== CLUSTER {cluster_id_2} DEBUGGING ===")
        self._debug_cluster_data(top_2, centers_2, cluster_id_2)
        
        # Ensure data consistency
        top_1, centers_1 = self._fix_data_alignment(top_1, centers_1, cluster_id_1)
        top_2, centers_2 = self._fix_data_alignment(top_2, centers_2, cluster_id_2)
        
        if len(top_1) == 0 or len(top_2) == 0:
            raise ValueError(f"No valid regions found after data alignment. Cluster {cluster_id_1}: {len(top_1)} regions, Cluster {cluster_id_2}: {len(top_2)} regions")
        
        # Perform deltaE comparison analysis
        comparison_results = []
        total_attempts = 0
        successful_comparisons = 0
        main_similar_count = 0
        both_similar_count = 0
        reflect_checked_count = 0
        
        print(f"\n=== PERFORMING COMPARISONS ===")
        
        for i, (_, region_1) in enumerate(top_1.iterrows()):
            region_1_id = region_1['color_regions']
            
            print(f"\nProcessing region {cluster_id_1}-{region_1_id} ({i+1}/{len(top_1)})")
            
            # Skip if region not found in centers or has NaN
            if pd.isna(region_1_id):
                print(f"  ❌ Skipped: region_1_id is NaN")
                continue
                
            if region_1_id not in centers_1.index:
                print(f"  ❌ Skipped: region {region_1_id} not found in centers_1")
                print(f"     Available centers_1 indices: {list(centers_1.index)}")
                continue
                
            center_1 = centers_1.loc[region_1_id]
            
            # Check for required columns and NaN values
            required_cols_1 = ['L_main', 'a_main', 'b_main', 'L_reflect', 'a_reflect', 'b_reflect']
            missing_cols_1 = [col for col in required_cols_1 if col not in center_1.index]
            if missing_cols_1:
                print(f"  ❌ Skipped: missing columns in center_1: {missing_cols_1}")
                continue
            
            lab_values_1 = center_1[['L_main', 'a_main', 'b_main']].values
            if pd.isna(lab_values_1).any():
                print(f"  ❌ Skipped: NaN values in main LAB for region {region_1_id}: {lab_values_1}")
                continue
                
            lab_main_1 = [center_1['L_main'], center_1['a_main'], center_1['b_main']]
            lab_reflect_1 = [center_1['L_reflect'], center_1['a_reflect'], center_1['b_reflect']]
            
            print(f"  ✅ Region {region_1_id} main LAB: {lab_main_1}")
            
            comparisons_for_this_region = 0
            
            for j, (_, region_2) in enumerate(top_2.iterrows()):
                region_2_id = region_2['color_regions']
                total_attempts += 1
                
                # Skip if region not found in centers or has NaN
                if pd.isna(region_2_id):
                    continue
                    
                if region_2_id not in centers_2.index:
                    continue
                    
                center_2 = centers_2.loc[region_2_id]
                
                # Check for required columns and NaN values
                required_cols_2 = ['L_main', 'a_main', 'b_main', 'L_reflect', 'a_reflect', 'b_reflect']
                missing_cols_2 = [col for col in required_cols_2 if col not in center_2.index]
                if missing_cols_2:
                    continue
                
                lab_values_2 = center_2[['L_main', 'a_main', 'b_main']].values
                if pd.isna(lab_values_2).any():
                    continue
                
                lab_main_2 = [center_2['L_main'], center_2['a_main'], center_2['b_main']]
                lab_reflect_2 = [center_2['L_reflect'], center_2['a_reflect'], center_2['b_reflect']]
                
                # STEP 1: Calculate deltaE for main colors (always calculated)
                delta_e_main = self._calculate_delta_e_cie2000_custom(lab_main_1, lab_main_2)
                
                # STEP 2: Check if main colors are similar
                is_main_similar = bool(delta_e_main < main_delta_e_threshold) if not pd.isna(delta_e_main) else False
                
                # STEP 3: Only calculate reflect deltaE if main colors are similar
                delta_e_reflect = np.nan
                is_reflect_similar = False
                
                if is_main_similar:
                    reflect_checked_count += 1
                    delta_e_reflect = self._calculate_delta_e_custom(lab_reflect_1, lab_reflect_2)
                    is_reflect_similar = bool(delta_e_reflect < reflect_delta_e_threshold) if not pd.isna(delta_e_reflect) else False
                
                # STEP 4: Determine if both criteria are met
                meets_both_criteria = is_main_similar and is_reflect_similar
                
                if not pd.isna(delta_e_main):
                    comparison_results.append({
                        f'region_{cluster_id_1}_id': int(region_1_id),
                        f'region_{cluster_id_2}_id': int(region_2_id),
                        f'region_{cluster_id_1}_score': float(region_1[metric_column]),
                        f'region_{cluster_id_2}_score': float(region_2[metric_column]),
                        f'region_{cluster_id_1}_samples': int(region_1['total_samples']),
                        f'region_{cluster_id_2}_samples': int(region_2['total_samples']),
                        'delta_e_main': float(delta_e_main),
                        'delta_e_reflect': float(delta_e_reflect) if not pd.isna(delta_e_reflect) else np.nan,
                        'similar_main': is_main_similar,
                        'similar_reflect': is_reflect_similar,
                        'meets_both_criteria': meets_both_criteria,
                        'reflect_checked': not pd.isna(delta_e_reflect),
                        f'L_main_{cluster_id_1}': float(center_1['L_main']), 
                        f'a_main_{cluster_id_1}': float(center_1['a_main']), 
                        f'b_main_{cluster_id_1}': float(center_1['b_main']),
                        f'L_main_{cluster_id_2}': float(center_2['L_main']), 
                        f'a_main_{cluster_id_2}': float(center_2['a_main']), 
                        f'b_main_{cluster_id_2}': float(center_2['b_main']),
                        f'L_reflect_{cluster_id_1}': float(center_1['L_reflect']), 
                        f'a_reflect_{cluster_id_1}': float(center_1['a_reflect']), 
                        f'b_reflect_{cluster_id_1}': float(center_1['b_reflect']),
                        f'L_reflect_{cluster_id_2}': float(center_2['L_reflect']), 
                        f'a_reflect_{cluster_id_2}': float(center_2['a_reflect']), 
                        f'b_reflect_{cluster_id_2}': float(center_2['b_reflect'])
                    })
                    successful_comparisons += 1
                    comparisons_for_this_region += 1
                    
                    if is_main_similar:
                        main_similar_count += 1
                    if meets_both_criteria:
                        both_similar_count += 1
            
            print(f"    Successful comparisons for this region: {comparisons_for_this_region}")
        
        print(f"\n=== COMPARISON SUMMARY ===")
        print(f"Total attempts: {total_attempts}")
        print(f"Successful comparisons: {successful_comparisons}")
        print(f"Main color similar pairs: {main_similar_count}")
        print(f"Reflect colors checked: {reflect_checked_count}")
        print(f"Both criteria met pairs: {both_similar_count}")
        print(f"Generated {len(comparison_results)} comparison results")
        
        if len(comparison_results) == 0:
            # Provide detailed diagnostic information
            self._provide_diagnostic_info(top_1, centers_1, top_2, centers_2, cluster_id_1, cluster_id_2)
            raise ValueError(f"No valid comparisons could be made between clusters {cluster_id_1} and {cluster_id_2}. See diagnostic information above.")
        
        # Convert to DataFrame and create different filtered views
        comparison_df = pd.DataFrame(comparison_results)
        
        # Filter 1: Regions with similar main colors (reflects may or may not have been checked)
        similar_main_pairs = comparison_df[comparison_df['similar_main'] == True].copy()
        similar_main_pairs = similar_main_pairs.sort_values('delta_e_main') if len(similar_main_pairs) > 0 else similar_main_pairs
        
        # Filter 2: Regions meeting BOTH criteria (main similar AND reflect similar)
        both_criteria_pairs = comparison_df[comparison_df['meets_both_criteria'] == True].copy()
        both_criteria_pairs = both_criteria_pairs.sort_values('delta_e_main') if len(both_criteria_pairs) > 0 else both_criteria_pairs
        
        # Filter 3: Regions where reflects were checked (main was similar)
        reflect_checked_pairs = comparison_df[comparison_df['reflect_checked'] == True].copy()
        reflect_checked_pairs = reflect_checked_pairs.sort_values('delta_e_main') if len(reflect_checked_pairs) > 0 else reflect_checked_pairs
        
        print(f"Similar main color pairs found (ΔE < {main_delta_e_threshold}): {len(similar_main_pairs)}")
        print(f"Both criteria met pairs found: {len(both_criteria_pairs)}")
        print(f"Reflect colors checked pairs: {len(reflect_checked_pairs)}")
        
        return {
            'comparison_df': comparison_df,
            'similar_pairs': similar_main_pairs,  # Keep this for backward compatibility
            'similar_main_pairs': similar_main_pairs,
            'both_criteria_pairs': both_criteria_pairs,
            'reflect_checked_pairs': reflect_checked_pairs,
            'cluster_id_1': cluster_id_1,
            'cluster_id_2': cluster_id_2,
            'delta_e_threshold': main_delta_e_threshold,  # Keep old name for compatibility
            'main_delta_e_threshold': main_delta_e_threshold,
            'reflect_delta_e_threshold': reflect_delta_e_threshold,
            'total_comparisons': len(comparison_df),
            'main_similar_count': len(similar_main_pairs),
            'both_criteria_count': len(both_criteria_pairs),
            'reflect_checked_count': len(reflect_checked_pairs),
            'similarity_rate': (len(similar_main_pairs) / len(comparison_df) * 100) if len(comparison_df) > 0 else 0,  # Keep old name
            'main_similarity_rate': (len(similar_main_pairs) / len(comparison_df) * 100) if len(comparison_df) > 0 else 0,
            'both_criteria_rate': (len(both_criteria_pairs) / len(comparison_df) * 100) if len(comparison_df) > 0 else 0,
            'metric_column': metric_column
        }
    
    def _calculate_delta_e_custom(self, lab1, lab2):
        """
        Custom DeltaE calculation using CIE76 formula (simpler but reliable)
        This avoids the numpy.asscalar error in colormath
        """
        try:
            # Validate input values
            if any(pd.isna(val) for val in lab1 + lab2):
                return np.nan
            
            # Convert to numpy arrays for easier calculation
            lab1 = np.array(lab1, dtype=float)
            lab2 = np.array(lab2, dtype=float)
            
            # Calculate CIE76 Delta E (simpler than CIE2000 but still accurate)
            delta_l = lab1[0] - lab2[0]
            delta_a = lab1[1] - lab2[1]
            delta_b = lab1[2] - lab2[2]
            
            delta_e = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
            
            return float(delta_e)
            
        except Exception as e:
            print(f"    Error calculating custom deltaE for {lab1} vs {lab2}: {e}")
            return np.nan
    
    def _calculate_delta_e_cie2000_custom(self, lab1, lab2):
        """
        Custom implementation of CIE2000 DeltaE (more complex but more accurate)
        This is a simplified version that avoids numpy.asscalar issues
        """
        try:
            # Validate input values
            if any(pd.isna(val) for val in lab1 + lab2):
                return np.nan
            
            # Convert to numpy arrays
            lab1 = np.array(lab1, dtype=float)
            lab2 = np.array(lab2, dtype=float)
            
            # Extract L*, a*, b* values
            L1, a1, b1 = lab1[0], lab1[1], lab1[2]
            L2, a2, b2 = lab2[0], lab2[1], lab2[2]
            
            # Calculate C and h values
            C1 = np.sqrt(a1**2 + b1**2)
            C2 = np.sqrt(a2**2 + b2**2)
            
            # Calculate means
            L_mean = (L1 + L2) / 2
            C_mean = (C1 + C2) / 2
            
            # Calculate G factor
            G = 0.5 * (1 - np.sqrt(C_mean**7 / (C_mean**7 + 25**7)))
            
            # Calculate modified a* values
            a1_prime = a1 * (1 + G)
            a2_prime = a2 * (1 + G)
            
            # Calculate modified C and h values
            C1_prime = np.sqrt(a1_prime**2 + b1**2)
            C2_prime = np.sqrt(a2_prime**2 + b2**2)
            
            # Calculate hue angles (in degrees)
            h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
            h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
            
            # Calculate differences
            delta_L_prime = L2 - L1
            delta_C_prime = C2_prime - C1_prime
            
            # Calculate delta_h_prime
            if C1_prime * C2_prime == 0:
                delta_h_prime = 0
            elif abs(h2_prime - h1_prime) <= 180:
                delta_h_prime = h2_prime - h1_prime
            elif h2_prime - h1_prime > 180:
                delta_h_prime = h2_prime - h1_prime - 360
            else:
                delta_h_prime = h2_prime - h1_prime + 360
            
            delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2))
            
            # Calculate means for weighting functions
            L_prime_mean = (L1 + L2) / 2
            C_prime_mean = (C1_prime + C2_prime) / 2
            
            if C1_prime * C2_prime == 0:
                h_prime_mean = h1_prime + h2_prime
            elif abs(h1_prime - h2_prime) <= 180:
                h_prime_mean = (h1_prime + h2_prime) / 2
            elif abs(h1_prime - h2_prime) > 180 and (h1_prime + h2_prime) < 360:
                h_prime_mean = (h1_prime + h2_prime + 360) / 2
            else:
                h_prime_mean = (h1_prime + h2_prime - 360) / 2
            
            # Calculate T
            T = (1 - 0.17 * np.cos(np.radians(h_prime_mean - 30)) +
                 0.24 * np.cos(np.radians(2 * h_prime_mean)) +
                 0.32 * np.cos(np.radians(3 * h_prime_mean + 6)) -
                 0.20 * np.cos(np.radians(4 * h_prime_mean - 63)))
            
            # Calculate delta_theta
            delta_theta = 30 * np.exp(-((h_prime_mean - 275) / 25)**2)
            
            # Calculate R_C
            R_C = 2 * np.sqrt(C_prime_mean**7 / (C_prime_mean**7 + 25**7))
            
            # Calculate S_L, S_C, S_H
            S_L = 1 + (0.015 * (L_prime_mean - 50)**2) / np.sqrt(20 + (L_prime_mean - 50)**2)
            S_C = 1 + 0.045 * C_prime_mean
            S_H = 1 + 0.015 * C_prime_mean * T
            
            # Calculate R_T
            R_T = -np.sin(2 * np.radians(delta_theta)) * R_C
            
            # Calculate final Delta E 2000
            delta_e_2000 = np.sqrt(
                (delta_L_prime / S_L)**2 +
                (delta_C_prime / S_C)**2 +
                (delta_H_prime / S_H)**2 +
                R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
            )
            
            return float(delta_e_2000)
            
        except Exception as e:
            print(f"    Error calculating CIE2000 deltaE for {lab1} vs {lab2}: {e}")
            # Fall back to simple CIE76 calculation
            return self._calculate_delta_e_custom(lab1, lab2)
    
    def _debug_cluster_data(self, top_regions, centers, cluster_id):
        """Debug cluster data structure"""
        print(f"  Top regions columns: {list(top_regions.columns)}")
        print(f"  Centers columns: {list(centers.columns) if len(centers) > 0 else 'EMPTY'}")
        
        if 'color_regions' in top_regions.columns:
            region_ids = list(top_regions['color_regions'].values)
            print(f"  Region IDs in top_regions: {region_ids}")
            print(f"  Region IDs data type: {top_regions['color_regions'].dtype}")
        else:
            print(f"  ❌ ERROR: 'color_regions' column not found in top_regions")
            return
        
        if len(centers) > 0:
            center_ids = list(centers.index)
            print(f"  Center IDs: {center_ids}")
            print(f"  Center IDs data type: {centers.index.dtype}")
            
            # Check for overlapping IDs
            overlap = set(region_ids) & set(center_ids)
            print(f"  Overlapping region IDs: {list(overlap)} ({len(overlap)} out of {len(region_ids)})")
            
            # Check for NaN values in color data
            if len(overlap) > 0:
                sample_id = list(overlap)[0]
                sample_center = centers.loc[sample_id]
                required_cols = ['L_main', 'a_main', 'b_main', 'L_reflect', 'a_reflect', 'b_reflect']
                available_cols = [col for col in required_cols if col in sample_center.index]
                missing_cols = [col for col in required_cols if col not in sample_center.index]
                
                print(f"  Available color columns: {available_cols}")
                print(f"  Missing color columns: {missing_cols}")
                
                if available_cols:
                    sample_values = sample_center[available_cols]
                    nan_cols = [col for col in available_cols if pd.isna(sample_center[col])]
                    print(f"  Sample values for region {sample_id}: {dict(sample_values)}")
                    print(f"  Columns with NaN values: {nan_cols}")
        else:
            print(f"  ❌ ERROR: Centers dataframe is empty")
    
    def _fix_data_alignment(self, top_regions, centers, cluster_id):
        """Fix data alignment issues"""
        print(f"\n  Fixing data alignment for cluster {cluster_id}...")
        
        if 'color_regions' not in top_regions.columns:
            print(f"    ❌ ERROR: 'color_regions' column not found")
            return pd.DataFrame(), pd.DataFrame()
        
        # Ensure consistent data types
        try:
            top_regions['color_regions'] = pd.to_numeric(top_regions['color_regions'], errors='coerce').astype('Int64')
            centers.index = pd.to_numeric(centers.index, errors='coerce').astype('Int64')
        except Exception as e:
            print(f"    ❌ Error converting data types: {e}")
            return top_regions, centers
        
        # Remove NaN regions
        initial_regions = len(top_regions)
        top_regions = top_regions.dropna(subset=['color_regions'])
        print(f"    Removed {initial_regions - len(top_regions)} regions with NaN IDs")
        
        # Remove centers with NaN indices
        initial_centers = len(centers)
        centers = centers.dropna()
        centers = centers[~pd.isna(centers.index)]
        print(f"    Removed {initial_centers - len(centers)} centers with NaN indices")
        
        # Filter top_regions to only include those with centers
        if len(centers) > 0:
            valid_regions = top_regions[top_regions['color_regions'].isin(centers.index)]
            removed_regions = len(top_regions) - len(valid_regions)
            if removed_regions > 0:
                print(f"    Removed {removed_regions} regions without corresponding centers")
            top_regions = valid_regions
        
        # Ensure required color columns exist
        required_cols = ['L_main', 'a_main', 'b_main', 'L_reflect', 'a_reflect', 'b_reflect']
        missing_cols = [col for col in required_cols if col not in centers.columns]
        
        if missing_cols:
            print(f"    ❌ Missing required color columns: {missing_cols}")
            for col in missing_cols:
                centers[col] = np.nan
        
        # Remove centers with all NaN color values
        if len(centers) > 0:
            main_color_cols = ['L_main', 'a_main', 'b_main']
            valid_centers = centers.dropna(subset=main_color_cols, how='all')
            removed_centers = len(centers) - len(valid_centers)
            if removed_centers > 0:
                print(f"    Removed {removed_centers} centers with all NaN main color values")
            centers = valid_centers
            
            # Update top_regions accordingly
            if len(centers) > 0:
                top_regions = top_regions[top_regions['color_regions'].isin(centers.index)]
        
        print(f"    Final: {len(top_regions)} regions, {len(centers)} centers")
        return top_regions, centers
    
    def _provide_diagnostic_info(self, top_1, centers_1, top_2, centers_2, cluster_id_1, cluster_id_2):
        """Provide detailed diagnostic information when comparison fails"""
        print(f"\n=== DIAGNOSTIC INFORMATION ===")
        print(f"Cluster {cluster_id_1}:")
        print(f"  - Top regions: {len(top_1)}")
        print(f"  - Centers: {len(centers_1)}")
        if len(top_1) > 0 and len(centers_1) > 0:
            overlap_1 = set(top_1['color_regions']) & set(centers_1.index)
            print(f"  - Valid overlapping regions: {len(overlap_1)}")
        
        print(f"Cluster {cluster_id_2}:")
        print(f"  - Top regions: {len(top_2)}")
        print(f"  - Centers: {len(centers_2)}")
        if len(top_2) > 0 and len(centers_2) > 0:
            overlap_2 = set(top_2['color_regions']) & set(centers_2.index)
            print(f"  - Valid overlapping regions: {len(overlap_2)}")
        
        print(f"\nPossible issues:")
        print(f"  1. No regions have valid color data (all NaN LAB values)")
        print(f"  2. Data type mismatches between region IDs and center indices")
        print(f"  3. Missing required color columns in the source data")
        print(f"  4. All color values failed deltaE calculation")
    
    def _lab_to_rgb(self, lab_l, lab_a, lab_b):
        """Convert LAB to RGB for visualization"""
        try:
            lab_color = LabColor(lab_l, lab_a, lab_b)
            rgb_color = convert_color(lab_color, sRGBColor)
            rgb = np.clip([rgb_color.rgb_r, rgb_color.rgb_g, rgb_color.rgb_b], 0, 1)
            return rgb
        except:
            return [0.5, 0.5, 0.5]
    
    def create_color_comparison_plot(self, similar_pairs, cluster_id_1, cluster_id_2, n_pairs=8):
        """Create color comparison visualization"""
        if len(similar_pairs) == 0:
            return None
            
        n_pairs = min(n_pairs, len(similar_pairs))
        fig, axes = plt.subplots(n_pairs, 4, figsize=(16, n_pairs * 3))
        
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for i, (_, pair) in enumerate(similar_pairs.head(n_pairs).iterrows()):
            try:
                # Cluster 1 - Main color
                rgb_main_1 = self._lab_to_rgb(pair[f'L_main_{cluster_id_1}'], pair[f'a_main_{cluster_id_1}'], pair[f'b_main_{cluster_id_1}'])
                axes[i, 0].add_patch(Rectangle((0, 0), 1, 1, color=rgb_main_1))
                axes[i, 0].set_title(f"ST{cluster_id_1}-R{pair[f'region_{cluster_id_1}_id']} Main\nL={pair[f'L_main_{cluster_id_1}']:.1f}, a={pair[f'a_main_{cluster_id_1}']:.1f}, b={pair[f'b_main_{cluster_id_1}']:.1f}", 
                                   fontsize=10, fontweight='bold')
                axes[i, 0].axis('off')
                
                # Cluster 2 - Main color
                rgb_main_2 = self._lab_to_rgb(pair[f'L_main_{cluster_id_2}'], pair[f'a_main_{cluster_id_2}'], pair[f'b_main_{cluster_id_2}'])
                axes[i, 1].add_patch(Rectangle((0, 0), 1, 1, color=rgb_main_2))
                axes[i, 1].set_title(f"ST{cluster_id_2}-R{pair[f'region_{cluster_id_2}_id']} Main\nL={pair[f'L_main_{cluster_id_2}']:.1f}, a={pair[f'a_main_{cluster_id_2}']:.1f}, b={pair[f'b_main_{cluster_id_2}']:.1f}\nΔE={pair['delta_e_main']:.2f}", 
                                   fontsize=10, fontweight='bold')
                axes[i, 1].axis('off')
                
                # Cluster 1 - Reflect color
                rgb_reflect_1 = self._lab_to_rgb(pair[f'L_reflect_{cluster_id_1}'], pair[f'a_reflect_{cluster_id_1}'], pair[f'b_reflect_{cluster_id_1}'])
                axes[i, 2].add_patch(Rectangle((0, 0), 1, 1, color=rgb_reflect_1))
                axes[i, 2].set_title(f"ST{cluster_id_1}-R{pair[f'region_{cluster_id_1}_id']} Reflect\nL={pair[f'L_reflect_{cluster_id_1}']:.1f}, a={pair[f'a_reflect_{cluster_id_1}']:.1f}, b={pair[f'b_reflect_{cluster_id_1}']:.1f}", 
                                   fontsize=10, fontweight='bold')
                axes[i, 2].axis('off')
                
                # Cluster 2 - Reflect color
                rgb_reflect_2 = self._lab_to_rgb(pair[f'L_reflect_{cluster_id_2}'], pair[f'a_reflect_{cluster_id_2}'], pair[f'b_reflect_{cluster_id_2}'])
                axes[i, 3].add_patch(Rectangle((0, 0), 1, 1, color=rgb_reflect_2))
                axes[i, 3].set_title(f"ST{cluster_id_2}-R{pair[f'region_{cluster_id_2}_id']} Reflect\nL={pair[f'L_reflect_{cluster_id_2}']:.1f}, a={pair[f'a_reflect_{cluster_id_2}']:.1f}, b={pair[f'b_reflect_{cluster_id_2}']:.1f}\nΔE={pair['delta_e_reflect']:.2f}", 
                                   fontsize=10, fontweight='bold')
                axes[i, 3].axis('off')
                
            except Exception as e:
                print(f"Error creating visualization for pair {i}: {e}")
        
        plt.suptitle(f'Similar Color Regions Comparison (ΔE < {self.delta_e_threshold})\nSkin Tone {cluster_id_1} vs Skin Tone {cluster_id_2}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_delta_e_scatter_plotly(self, comparison_df, similar_pairs, cluster_id_1, cluster_id_2):
        """Create interactive DeltaE scatter plot using Plotly"""
        # Create color coding
        colors = ['green' if similar else 'red' for similar in comparison_df['similar_main']]
        sizes = [(row[f'region_{cluster_id_1}_samples'] + row[f'region_{cluster_id_2}_samples']) / 20 
                 for _, row in comparison_df.iterrows()]
        
        # Create hover text
        hover_text = []
        for _, row in comparison_df.iterrows():
            text = f"ST{cluster_id_1}-R{row[f'region_{cluster_id_1}_id']} vs ST{cluster_id_2}-R{row[f'region_{cluster_id_2}_id']}<br>" + \
                   f"ΔE Main: {row['delta_e_main']:.2f}<br>" + \
                   f"ΔE Reflect: {row['delta_e_reflect']:.2f}<br>" + \
                   f"ST{cluster_id_1} Score: {row[f'region_{cluster_id_1}_score']:.1f}%<br>" + \
                   f"ST{cluster_id_2} Score: {row[f'region_{cluster_id_2}_score']:.1f}%"
            hover_text.append(text)
        
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=comparison_df['delta_e_main'],
            y=comparison_df['delta_e_reflect'],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Region Pairs"
        ))
        
        # Add threshold line
        fig.add_vline(x=self.delta_e_threshold, line_dash="dash", line_color="orange", 
                     annotation_text=f"ΔE threshold ({self.delta_e_threshold})")
        
        # Annotate top 5 similar pairs
        if len(similar_pairs) > 0:
            for _, row in similar_pairs.head(5).iterrows():
                fig.add_annotation(
                    x=row['delta_e_main'],
                    y=row['delta_e_reflect'],
                    text=f"R{row[f'region_{cluster_id_1}_id']}-R{row[f'region_{cluster_id_2}_id']}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="blue",
                    font=dict(size=10)
                )
        
        fig.update_layout(
            title=f'Color Similarity Analysis Between Skin Tones {cluster_id_1} and {cluster_id_2}<br>' +
                  f'<span style="color:green">Green = Similar Main Colors (ΔE < {self.delta_e_threshold})</span>, ' +
                  f'<span style="color:red">Red = Different Main Colors</span>',
            xaxis_title='ΔE Main Colors',
            yaxis_title='ΔE Reflect Colors',
            width=800,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_comparison_excel_data(self, comparison_results, cluster_id_1, cluster_id_2):
        """Create Excel data for comparison analysis"""
        comparison_df = comparison_results['comparison_df']
        similar_pairs = comparison_results['similar_pairs']
        metric_column = comparison_results.get('metric_column', 'combined_criteria_pct')
        
        # Main comparison data
        excel_data = []
        for _, row in comparison_df.iterrows():
            excel_data.append({
                f'Skin_Tone_{cluster_id_1}_Region': int(row[f'region_{cluster_id_1}_id']),
                f'Skin_Tone_{cluster_id_2}_Region': int(row[f'region_{cluster_id_2}_id']),
                f'ST{cluster_id_1}_Score': row[f'region_{cluster_id_1}_score'],
                f'ST{cluster_id_2}_Score': row[f'region_{cluster_id_2}_score'],
                f'ST{cluster_id_1}_Samples': int(row[f'region_{cluster_id_1}_samples']),
                f'ST{cluster_id_2}_Samples': int(row[f'region_{cluster_id_2}_samples']),
                'DeltaE_Main_Colors': row['delta_e_main'],
                'DeltaE_Reflect_Colors': row['delta_e_reflect'],
                'Similar_Main_Colors': row['similar_main'],
                f'ST{cluster_id_1}_L_main': row[f'L_main_{cluster_id_1}'], 
                f'ST{cluster_id_1}_a_main': row[f'a_main_{cluster_id_1}'], 
                f'ST{cluster_id_1}_b_main': row[f'b_main_{cluster_id_1}'],
                f'ST{cluster_id_2}_L_main': row[f'L_main_{cluster_id_2}'], 
                f'ST{cluster_id_2}_a_main': row[f'a_main_{cluster_id_2}'], 
                f'ST{cluster_id_2}_b_main': row[f'b_main_{cluster_id_2}'],
                f'ST{cluster_id_1}_L_reflect': row[f'L_reflect_{cluster_id_1}'], 
                f'ST{cluster_id_1}_a_reflect': row[f'a_reflect_{cluster_id_1}'], 
                f'ST{cluster_id_1}_b_reflect': row[f'b_reflect_{cluster_id_1}'],
                f'ST{cluster_id_2}_L_reflect': row[f'L_reflect_{cluster_id_2}'], 
                f'ST{cluster_id_2}_a_reflect': row[f'a_reflect_{cluster_id_2}'], 
                f'ST{cluster_id_2}_b_reflect': row[f'b_reflect_{cluster_id_2}']
            })
        
        # Similar pairs data
        similar_excel_data = []
        if len(similar_pairs) > 0:
            for _, row in similar_pairs.iterrows():
                similar_excel_data.append({
                    f'Skin_Tone_{cluster_id_1}_Region': int(row[f'region_{cluster_id_1}_id']),
                    f'Skin_Tone_{cluster_id_2}_Region': int(row[f'region_{cluster_id_2}_id']),
                    'DeltaE_Main': row['delta_e_main'],
                    'DeltaE_Reflect': row['delta_e_reflect'],
                    f'ST{cluster_id_1}_Score': row[f'region_{cluster_id_1}_score'],
                    f'ST{cluster_id_2}_Score': row[f'region_{cluster_id_2}_score'],
                    f'ST{cluster_id_1}_Samples': int(row[f'region_{cluster_id_1}_samples']),
                    f'ST{cluster_id_2}_Samples': int(row[f'region_{cluster_id_2}_samples']),
                    f'ST{cluster_id_1}_Main_LAB': f"L={row[f'L_main_{cluster_id_1}']:.1f}, a={row[f'a_main_{cluster_id_1}']:.1f}, b={row[f'b_main_{cluster_id_1}']:.1f}",
                    f'ST{cluster_id_2}_Main_LAB': f"L={row[f'L_main_{cluster_id_2}']:.1f}, a={row[f'a_main_{cluster_id_2}']:.1f}, b={row[f'b_main_{cluster_id_2}']:.1f}",
                    f'ST{cluster_id_1}_Reflect_LAB': f"L={row[f'L_reflect_{cluster_id_1}']:.1f}, a={row[f'a_reflect_{cluster_id_1}']:.1f}, b={row[f'b_reflect_{cluster_id_1}']:.1f}",
                    f'ST{cluster_id_2}_Reflect_LAB': f"L={row[f'L_reflect_{cluster_id_2}']:.1f}, a={row[f'a_reflect_{cluster_id_2}']:.1f}, b={row[f'b_reflect_{cluster_id_2}']:.1f}"
                })
        
        # Summary data
        summary_data = {
            'Metric': [
                'Metric Column Used',
                'DeltaE Threshold Used',
                'Total Comparisons Made',
                'Similar Main Color Pairs Found',
                'Similarity Rate (%)',
                'Best Matching Pair (Lowest ΔE)',
                'Best Match DeltaE Main',
                'Best Match DeltaE Reflect',
                'Average ΔE Main Colors',
                'Average ΔE Reflect Colors',
                'Analysis Date'
            ],
            'Value': [
                metric_column,
                comparison_results['delta_e_threshold'],
                comparison_results['total_comparisons'],
                comparison_results['similar_count'],
                f"{comparison_results['similarity_rate']:.1f}%",
                f"ST{cluster_id_1}-R{similar_pairs.iloc[0][f'region_{cluster_id_1}_id']} vs ST{cluster_id_2}-R{similar_pairs.iloc[0][f'region_{cluster_id_2}_id']}" if len(similar_pairs) > 0 else "None",
                f"{similar_pairs.iloc[0]['delta_e_main']:.2f}" if len(similar_pairs) > 0 else "N/A",
                f"{similar_pairs.iloc[0]['delta_e_reflect']:.2f}" if len(similar_pairs) > 0 else "N/A",
                f"{comparison_df['delta_e_main'].mean():.2f}",
                f"{comparison_df['delta_e_reflect'].mean():.2f}",
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        return {
            'all_comparisons': pd.DataFrame(excel_data),
            'similar_pairs': pd.DataFrame(similar_excel_data) if similar_excel_data else pd.DataFrame(),
            'summary': pd.DataFrame(summary_data)
        }

    def create_similar_pairs_recap(self, comparison_results, cluster_id_1, cluster_id_2):
        """
        Create a recap of similar pairs, grouping regions that match with multiple regions
        and calculating centroids for grouped pairs
        """
        similar_pairs = comparison_results['similar_pairs']
        
        if len(similar_pairs) == 0:
            return {
                'grouped_pairs': pd.DataFrame(),
                'centroid_data': pd.DataFrame(),
                'summary_stats': {}
            }
        
        print(f"\n=== CREATING SIMILAR PAIRS RECAP ===")
        print(f"Processing {len(similar_pairs)} similar pairs...")
        
        # Group by cluster 1 regions that have multiple matches
        cluster_1_groups = self._group_regions_by_matches(similar_pairs, cluster_id_1, cluster_id_2, 'cluster_1')
        
        # Group by cluster 2 regions that have multiple matches  
        cluster_2_groups = self._group_regions_by_matches(similar_pairs, cluster_id_2, cluster_id_1, 'cluster_2')
        
        # Combine all groups
        all_groups = cluster_1_groups + cluster_2_groups
        
        # Create grouped pairs dataframe
        grouped_pairs_data = []
        centroid_data = []
        
        for group in all_groups:
            group_info = self._process_group_for_recap(group, cluster_id_1, cluster_id_2, similar_pairs)
            if group_info:
                grouped_pairs_data.append(group_info['group_data'])
                centroid_data.append(group_info['centroid_data'])
        
        # Create summary statistics
        summary_stats = self._calculate_recap_summary(all_groups, similar_pairs, cluster_id_1, cluster_id_2)
        
        return {
            'grouped_pairs': pd.DataFrame(grouped_pairs_data) if grouped_pairs_data else pd.DataFrame(),
            'centroid_data': pd.DataFrame(centroid_data) if centroid_data else pd.DataFrame(),
            'summary_stats': summary_stats,
            'individual_groups': all_groups
        }

    def _group_regions_by_matches(self, similar_pairs, primary_cluster, secondary_cluster, group_type):
        """Group regions that have multiple matches"""
        groups = []
        
        # Count matches for each region in primary cluster
        primary_col = f'region_{primary_cluster}_id'
        region_matches = similar_pairs.groupby(primary_col).agg({
            f'region_{secondary_cluster}_id': list,
            'delta_e_main': list,
            'delta_e_reflect': list
        }).reset_index()
        
        # Filter regions with 2+ matches
        multi_match_regions = region_matches[region_matches[f'region_{secondary_cluster}_id'].apply(len) >= 2]
        
        for _, region_data in multi_match_regions.iterrows():
            primary_region = region_data[primary_col]
            matched_regions = region_data[f'region_{secondary_cluster}_id']
            delta_e_main_list = region_data['delta_e_main']
            delta_e_reflect_list = region_data['delta_e_reflect']
            
            # Create group name
            if group_type == 'cluster_1':
                group_name = f"ST{primary_cluster}R{primary_region} + " + " + ".join([f"ST{secondary_cluster}R{r}" for r in matched_regions])
            else:
                group_name = f"ST{primary_cluster}R{primary_region} + " + " + ".join([f"ST{secondary_cluster}R{r}" for r in matched_regions])
            
            groups.append({
                'group_name': group_name,
                'primary_cluster': primary_cluster,
                'primary_region': primary_region,
                'secondary_cluster': secondary_cluster,
                'matched_regions': matched_regions,
                'group_type': group_type,
                'match_count': len(matched_regions),
                'avg_delta_e_main': np.mean(delta_e_main_list),
                'avg_delta_e_reflect': np.mean(delta_e_reflect_list),
                'min_delta_e_main': np.min(delta_e_main_list),
                'max_delta_e_main': np.max(delta_e_main_list)
            })
        
        return groups

    def _process_group_for_recap(self, group, cluster_id_1, cluster_id_2, similar_pairs):
        """Process a group to calculate centroids and create display data"""
        try:
            primary_cluster = group['primary_cluster']
            primary_region = group['primary_region']
            secondary_cluster = group['secondary_cluster']
            matched_regions = group['matched_regions']
            
            # Get all pairs for this group
            group_pairs = similar_pairs[
                (similar_pairs[f'region_{primary_cluster}_id'] == primary_region) &
                (similar_pairs[f'region_{secondary_cluster}_id'].isin(matched_regions))
            ]
            
            if len(group_pairs) == 0:
                return None
            
            # Calculate centroids for main colors
            main_centroid_primary = self._calculate_color_centroid(group_pairs, primary_cluster, 'main')
            main_centroid_secondary = self._calculate_color_centroid(group_pairs, secondary_cluster, 'main')
            
            # Calculate centroids for reflect colors
            reflect_centroid_primary = self._calculate_color_centroid(group_pairs, primary_cluster, 'reflect')
            reflect_centroid_secondary = self._calculate_color_centroid(group_pairs, secondary_cluster, 'reflect')
            
            # Calculate centroid of centroids (overall group centroid)
            overall_main_centroid = {
                'L': (main_centroid_primary['L'] + main_centroid_secondary['L']) / 2,
                'a': (main_centroid_primary['a'] + main_centroid_secondary['a']) / 2,
                'b': (main_centroid_primary['b'] + main_centroid_secondary['b']) / 2
            }
            
            overall_reflect_centroid = {
                'L': (reflect_centroid_primary['L'] + reflect_centroid_secondary['L']) / 2,
                'a': (reflect_centroid_primary['a'] + reflect_centroid_secondary['a']) / 2,
                'b': (reflect_centroid_primary['b'] + reflect_centroid_secondary['b']) / 2
            }
            
            # Calculate average performance scores
            avg_score_primary = group_pairs[f'region_{primary_cluster}_score'].mean()
            avg_score_secondary = group_pairs[f'region_{secondary_cluster}_score'].mean()
            
            # Calculate total samples
            total_samples_primary = group_pairs[f'region_{primary_cluster}_samples'].sum()
            total_samples_secondary = group_pairs[f'region_{secondary_cluster}_samples'].sum()
            
            group_data = {
                'group_name': group['group_name'],
                'primary_cluster': primary_cluster,
                'primary_region': primary_region,
                'secondary_cluster': secondary_cluster,
                'matched_regions_count': len(matched_regions),
                'matched_regions_list': ", ".join([str(r) for r in matched_regions]),
                'avg_delta_e_main': group['avg_delta_e_main'],
                'avg_delta_e_reflect': group['avg_delta_e_reflect'],
                'min_delta_e_main': group['min_delta_e_main'],
                'max_delta_e_main': group['max_delta_e_main'],
                'avg_score_primary': avg_score_primary,
                'avg_score_secondary': avg_score_secondary,
                'total_samples_primary': total_samples_primary,
                'total_samples_secondary': total_samples_secondary,
                'overall_avg_score': (avg_score_primary + avg_score_secondary) / 2
            }
            
            centroid_data = {
                'group_name': group['group_name'],
                'overall_main_L': overall_main_centroid['L'],
                'overall_main_a': overall_main_centroid['a'],
                'overall_main_b': overall_main_centroid['b'],
                'overall_reflect_L': overall_reflect_centroid['L'],
                'overall_reflect_a': overall_reflect_centroid['a'],
                'overall_reflect_b': overall_reflect_centroid['b'],
                f'primary_main_L': main_centroid_primary['L'],
                f'primary_main_a': main_centroid_primary['a'],
                f'primary_main_b': main_centroid_primary['b'],
                f'primary_reflect_L': reflect_centroid_primary['L'],
                f'primary_reflect_a': reflect_centroid_primary['a'],
                f'primary_reflect_b': reflect_centroid_primary['b'],
                f'secondary_main_L': main_centroid_secondary['L'],
                f'secondary_main_a': main_centroid_secondary['a'],
                f'secondary_main_b': main_centroid_secondary['b'],
                f'secondary_reflect_L': reflect_centroid_secondary['L'],
                f'secondary_reflect_a': reflect_centroid_secondary['a'],
                f'secondary_reflect_b': reflect_centroid_secondary['b']
            }
            
            return {
                'group_data': group_data,
                'centroid_data': centroid_data
            }
            
        except Exception as e:
            print(f"Error processing group {group.get('group_name', 'Unknown')}: {e}")
            return None

    def _calculate_color_centroid(self, group_pairs, cluster_id, color_type):
        """Calculate centroid of colors for a specific cluster and color type"""
        L_col = f'L_{color_type}_{cluster_id}'
        a_col = f'a_{color_type}_{cluster_id}'
        b_col = f'b_{color_type}_{cluster_id}'
        
        return {
            'L': group_pairs[L_col].mean(),
            'a': group_pairs[a_col].mean(),
            'b': group_pairs[b_col].mean()
        }

    def _calculate_recap_summary(self, all_groups, similar_pairs, cluster_id_1, cluster_id_2):
        """Calculate summary statistics for the recap"""
        total_groups = len(all_groups)
        total_similar_pairs = len(similar_pairs)
        
        # Count how many regions have multiple matches
        cluster_1_multi_match = len([g for g in all_groups if g['group_type'] == 'cluster_1'])
        cluster_2_multi_match = len([g for g in all_groups if g['group_type'] == 'cluster_2'])
        
        # Calculate average matches per multi-match region
        avg_matches_per_group = np.mean([g['match_count'] for g in all_groups]) if all_groups else 0
        
        # Find the group with most matches
        max_matches_group = max(all_groups, key=lambda x: x['match_count']) if all_groups else None
        
        return {
            'total_groups': total_groups,
            'total_similar_pairs': total_similar_pairs,
            'cluster_1_multi_match_regions': cluster_1_multi_match,
            'cluster_2_multi_match_regions': cluster_2_multi_match,
            'avg_matches_per_group': avg_matches_per_group,
            'max_matches_group': max_matches_group,
            'multi_match_rate_cluster_1': (cluster_1_multi_match / len(similar_pairs[f'region_{cluster_id_1}_id'].unique())) * 100 if len(similar_pairs) > 0 else 0,
            'multi_match_rate_cluster_2': (cluster_2_multi_match / len(similar_pairs[f'region_{cluster_id_2}_id'].unique())) * 100 if len(similar_pairs) > 0 else 0
        }

    def create_recap_visualization(self, recap_results, cluster_id_1, cluster_id_2):
        """Create visualization for the recap"""
        grouped_pairs = recap_results['grouped_pairs']
        centroid_data = recap_results['centroid_data']
        
        if len(grouped_pairs) == 0:
            return None, None
        
        # Create grouped pairs visualization
        fig_groups = self._create_grouped_pairs_chart(grouped_pairs, cluster_id_1, cluster_id_2)
        
        # Create centroid color swatches
        fig_centroids = self._create_centroid_swatches(centroid_data)
        
        return fig_groups, fig_centroids

    def _create_grouped_pairs_chart(self, grouped_pairs, cluster_id_1, cluster_id_2):
        """Create chart showing grouped pairs performance"""
        fig = go.Figure()
        
        # Add bars for average performance
        fig.add_trace(go.Bar(
            x=grouped_pairs['group_name'],
            y=grouped_pairs['overall_avg_score'],
            text=[f"{x:.1f}%" for x in grouped_pairs['overall_avg_score']],
            textposition='outside',
            marker_color='#4A74F3',
            name='Overall Average Score',
            hovertemplate="<b>%{x}</b><br>" +
                        "Average Performance: %{y:.1f}%<br>" +
                        "Matches: %{customdata}<br>" +
                        "<extra></extra>",
            customdata=grouped_pairs['matched_regions_count']
        ))
        
        fig.update_layout(
            title=f'Multi-Match Region Groups Performance<br>ST{cluster_id_1} vs ST{cluster_id_2}',
            xaxis_title='Region Groups',
            yaxis_title='Average Performance Score (%)',
            xaxis={'tickangle': 45},
            height=500,
            showlegend=False
        )
        
        return fig

    def _create_centroid_swatches(self, centroid_data):
        """Create color swatches for centroids"""
        if len(centroid_data) == 0:
            return None
        
        n_groups = len(centroid_data)
        fig, axes = plt.subplots(n_groups, 3, figsize=(12, n_groups * 2))
        
        if n_groups == 1:
            axes = axes.reshape(1, -1)
        
        for i, (_, row) in enumerate(centroid_data.iterrows()):
            try:
                # Overall centroid - main
                rgb_overall_main = self._lab_to_rgb(row['overall_main_L'], row['overall_main_a'], row['overall_main_b'])
                axes[i, 0].add_patch(Rectangle((0, 0), 1, 1, color=rgb_overall_main))
                axes[i, 0].set_title(f"Overall Main Centroid\nL={row['overall_main_L']:.1f}, a={row['overall_main_a']:.1f}, b={row['overall_main_b']:.1f}", fontsize=10)
                axes[i, 0].axis('off')
                
                # Overall centroid - reflect
                rgb_overall_reflect = self._lab_to_rgb(row['overall_reflect_L'], row['overall_reflect_a'], row['overall_reflect_b'])
                axes[i, 1].add_patch(Rectangle((0, 0), 1, 1, color=rgb_overall_reflect))
                axes[i, 1].set_title(f"Overall Reflect Centroid\nL={row['overall_reflect_L']:.1f}, a={row['overall_reflect_a']:.1f}, b={row['overall_reflect_b']:.1f}", fontsize=10)
                axes[i, 1].axis('off')
                
                # Group name
                axes[i, 2].text(0.5, 0.5, row['group_name'], ha='center', va='center', fontsize=10, fontweight='bold', wrap=True)
                axes[i, 2].set_xlim(0, 1)
                axes[i, 2].set_ylim(0, 1)
                axes[i, 2].set_title("Group Name", fontsize=10)
                axes[i, 2].axis('off')
                
            except Exception as e:
                print(f"Error creating swatch for group {i}: {e}")
        
        plt.suptitle('Centroid Colors for Multi-Match Groups', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig

    def create_recap_excel_data(self, recap_results, cluster_id_1, cluster_id_2):
        """Create Excel data for recap analysis"""
        grouped_pairs = recap_results['grouped_pairs']
        centroid_data = recap_results['centroid_data']
        summary_stats = recap_results['summary_stats']
        
        # Grouped pairs data
        excel_grouped = grouped_pairs.copy() if len(grouped_pairs) > 0 else pd.DataFrame()
        
        # Centroid data
        excel_centroids = centroid_data.copy() if len(centroid_data) > 0 else pd.DataFrame()
        
        # Summary data
        summary_data = {
            'Metric': [
                'Total Multi-Match Groups',
                f'ST{cluster_id_1} Multi-Match Regions',
                f'ST{cluster_id_2} Multi-Match Regions',
                f'ST{cluster_id_1} Multi-Match Rate (%)',
                f'ST{cluster_id_2} Multi-Match Rate (%)',
                'Average Matches per Group',
                'Max Matches in Single Group',
                'Best Performing Group',
                'Analysis Date'
            ],
            'Value': [
                summary_stats.get('total_groups', 0),
                summary_stats.get('cluster_1_multi_match_regions', 0),
                summary_stats.get('cluster_2_multi_match_regions', 0),
                f"{summary_stats.get('multi_match_rate_cluster_1', 0):.1f}%",
                f"{summary_stats.get('multi_match_rate_cluster_2', 0):.1f}%",
                f"{summary_stats.get('avg_matches_per_group', 0):.1f}",
                summary_stats.get('max_matches_group', {}).get('match_count', 'N/A') if summary_stats.get('max_matches_group') else 'N/A',
                summary_stats.get('max_matches_group', {}).get('group_name', 'N/A') if summary_stats.get('max_matches_group') else 'N/A',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        return {
            'grouped_pairs': excel_grouped,
            'centroid_data': excel_centroids,
            'summary': pd.DataFrame(summary_data)
        }
    
    def find_universal_regions(self, cluster_results, delta_e_threshold=4.0, min_clusters_required=3, min_performance_threshold=40.0, top_n=10, metric_column='combined_criteria_pct'):
        """
        Find color regions that appear across multiple clusters with similar colors
        """
        print(f"\n=== UNIVERSAL REGIONS ANALYSIS ===")
        print(f"DeltaE threshold: {delta_e_threshold}")
        print(f"Min clusters required: {min_clusters_required}")
        print(f"Min performance threshold: {min_performance_threshold}%")
        print(f"Top N regions per cluster: {top_n}")
        print(f"Metric column: {metric_column}")
        
        # Prepare data from all clusters
        cluster_data = {}
        analyzed_clusters = []
        
        for cluster_id, results in cluster_results.items():
            top_regions = results['top_regions'].head(top_n).copy()
            centers = results['cluster_centers'].copy()
            
            # Filter by performance threshold
            top_regions = top_regions[top_regions[metric_column] >= min_performance_threshold]
            
            if len(top_regions) == 0:
                print(f"No regions meet performance threshold in cluster {cluster_id}")
                continue
            
            # Fix data alignment
            top_regions, centers = self._fix_data_alignment(top_regions, centers, cluster_id)
            
            if len(top_regions) > 0:
                cluster_data[cluster_id] = {
                    'top_regions': top_regions,
                    'centers': centers
                }
                analyzed_clusters.append(cluster_id)
        
        if len(analyzed_clusters) < min_clusters_required:
            raise ValueError(f"Only {len(analyzed_clusters)} clusters have sufficient data. Need at least {min_clusters_required}.")
        
        print(f"Analyzing {len(analyzed_clusters)} clusters: {analyzed_clusters}")
        
        # Find universal groups by comparing all regions across all clusters
        universal_groups = self._find_universal_groups(
            cluster_data, delta_e_threshold, min_clusters_required, analyzed_clusters, metric_column
        )
        
        # Calculate centroids for universal groups
        centroid_colors = self._calculate_universal_centroids(universal_groups, cluster_data)
        
        # Calculate performance summaries
        performance_summary = self._calculate_universal_performance(universal_groups, cluster_data)
        
        # Create cluster coverage matrix
        cluster_coverage = self._create_cluster_coverage_matrix(universal_groups, analyzed_clusters)
        
        return {
            'universal_regions': universal_groups,
            'centroid_colors': centroid_colors,
            'performance_summary': performance_summary,
            'cluster_coverage': cluster_coverage,
            'analyzed_clusters': analyzed_clusters,
            'parameters': {
                'delta_e_threshold': delta_e_threshold,
                'min_clusters_required': min_clusters_required,
                'min_performance_threshold': min_performance_threshold,
                'top_n': top_n,
                'metric_column': metric_column
            }
        }

    def _find_universal_groups(self, cluster_data, delta_e_threshold, min_clusters_required, analyzed_clusters, metric_column):
        """Find groups of similar regions across clusters"""
        print(f"\n=== FINDING UNIVERSAL GROUPS ===")
        
        # Create all possible region pairs across clusters
        all_regions = []
        for cluster_id, data in cluster_data.items():
            for _, region in data['top_regions'].iterrows():
                region_id = region['color_regions']
                if region_id in data['centers'].index:
                    center = data['centers'].loc[region_id]
                    all_regions.append({
                        'cluster_id': cluster_id,
                        'region_id': region_id,
                        'performance': region[metric_column],
                        'samples': region['total_samples'],
                        'L_main': center['L_main'],
                        'a_main': center['a_main'],
                        'b_main': center['b_main'],
                        'L_reflect': center['L_reflect'],
                        'a_reflect': center['a_reflect'],
                        'b_reflect': center['b_reflect']
                    })
        
        print(f"Total regions to analyze: {len(all_regions)}")
        
        # Find groups of similar regions
        universal_groups = []
        processed_regions = set()
        
        for i, region_1 in enumerate(all_regions):
            if f"{region_1['cluster_id']}_{region_1['region_id']}" in processed_regions:
                continue
            
            # Find all regions similar to this one
            similar_group = [region_1]
            group_clusters = {region_1['cluster_id']}
            
            for j, region_2 in enumerate(all_regions):
                if i == j or region_2['cluster_id'] == region_1['cluster_id']:
                    continue
                
                if f"{region_2['cluster_id']}_{region_2['region_id']}" in processed_regions:
                    continue
                
                # Calculate DeltaE for main colors
                delta_e_main = self._calculate_delta_e_cie2000_custom(
                    [region_1['L_main'], region_1['a_main'], region_1['b_main']],
                    [region_2['L_main'], region_2['a_main'], region_2['b_main']]
                )
                
                if not pd.isna(delta_e_main) and delta_e_main < delta_e_threshold:
                    similar_group.append(region_2)
                    group_clusters.add(region_2['cluster_id'])
            
            # Check if this group meets minimum cluster requirements
            if len(group_clusters) >= min_clusters_required:
                # Mark all regions in this group as processed
                for region in similar_group:
                    processed_regions.add(f"{region['cluster_id']}_{region['region_id']}")
                
                # Calculate group statistics
                group_delta_es = []
                for k in range(len(similar_group)):
                    for l in range(k+1, len(similar_group)):
                        delta_e = self._calculate_delta_e_cie2000_custom(
                            [similar_group[k]['L_main'], similar_group[k]['a_main'], similar_group[k]['b_main']],
                            [similar_group[l]['L_main'], similar_group[l]['a_main'], similar_group[l]['b_main']]
                        )
                        if not pd.isna(delta_e):
                            group_delta_es.append(delta_e)
                
                # Create group info
                cluster_regions = {}
                for region in similar_group:
                    cluster_regions[region['cluster_id']] = {
                        'region_id': region['region_id'],
                        'performance': region['performance'],
                        'samples': region['samples']
                    }
                
                group_name = "Universal_" + "_".join([f"ST{c}R{cluster_regions[c]['region_id']}" for c in sorted(group_clusters)])
                
                universal_groups.append({
                    'group_name': group_name,
                    'cluster_regions': cluster_regions,
                    'avg_performance': np.mean([r['performance'] for r in similar_group]),
                    'min_performance': np.min([r['performance'] for r in similar_group]),
                    'max_performance': np.max([r['performance'] for r in similar_group]),
                    'total_samples': sum([r['samples'] for r in similar_group]),
                    'cluster_count': len(group_clusters),
                    'avg_delta_e': np.mean(group_delta_es) if group_delta_es else 0,
                    'max_delta_e': np.max(group_delta_es) if group_delta_es else 0,
                    'regions_data': similar_group
                })
        
        # Sort by average performance and cluster count
        universal_groups.sort(key=lambda x: (x['cluster_count'], x['avg_performance']), reverse=True)
        
        print(f"Found {len(universal_groups)} universal groups")
        return universal_groups

    def _calculate_universal_centroids(self, universal_groups, cluster_data):
        """Calculate centroid colors for universal groups"""
        centroid_colors = []
        
        for group in universal_groups:
            regions_data = group['regions_data']
            
            # Calculate centroids for main and reflect colors
            main_centroid = {
                'L': np.mean([r['L_main'] for r in regions_data]),
                'a': np.mean([r['a_main'] for r in regions_data]),
                'b': np.mean([r['b_main'] for r in regions_data])
            }
            
            reflect_centroid = {
                'L': np.mean([r['L_reflect'] for r in regions_data]),
                'a': np.mean([r['a_reflect'] for r in regions_data]),
                'b': np.mean([r['b_reflect'] for r in regions_data])
            }
            
            centroid_colors.append({
                'group_name': group['group_name'],
                'cluster_count': group['cluster_count'],
                'centroid_main_L': main_centroid['L'],
                'centroid_main_a': main_centroid['a'],
                'centroid_main_b': main_centroid['b'],
                'centroid_reflect_L': reflect_centroid['L'],
                'centroid_reflect_a': reflect_centroid['a'],
                'centroid_reflect_b': reflect_centroid['b']
            })
        
        return centroid_colors

    def _calculate_universal_performance(self, universal_groups, cluster_data):
        """Calculate performance summaries for universal groups"""
        performance_summary = {}
        
        for group in universal_groups:
            group_name = group['group_name']
            cluster_regions = group['cluster_regions']
            
            performance_by_cluster = {}
            for cluster_id, region_info in cluster_regions.items():
                performance_by_cluster[cluster_id] = region_info['performance']
            
            performance_summary[group_name] = {
                'cluster_performances': performance_by_cluster,
                'avg_performance': group['avg_performance'],
                'performance_std': np.std(list(performance_by_cluster.values())),
                'consistency_score': 100 - np.std(list(performance_by_cluster.values()))  # Higher = more consistent
            }
        
        return performance_summary

    def _create_cluster_coverage_matrix(self, universal_groups, analyzed_clusters):
        """Create cluster coverage matrix for visualization"""
        coverage_data = []
        
        for group in universal_groups:
            for cluster_id in analyzed_clusters:
                if cluster_id in group['cluster_regions']:
                    region_info = group['cluster_regions'][cluster_id]
                    coverage_data.append({
                        'group_name': group['group_name'],
                        'cluster_id': cluster_id,
                        'region_id': region_info['region_id'],
                        'performance': region_info['performance'],
                        'samples': region_info['samples'],
                        'covered': True
                    })
                else:
                    coverage_data.append({
                        'group_name': group['group_name'],
                        'cluster_id': cluster_id,
                        'region_id': None,
                        'performance': 0,
                        'samples': 0,
                        'covered': False
                    })
        
        return pd.DataFrame(coverage_data)

    def create_universal_performance_chart(self, universal_regions, analyzed_clusters):
        """Create performance chart for universal regions"""
        if len(universal_regions) == 0:
            return None
        
        group_names = [group['group_name'] for group in universal_regions]
        avg_performances = [group['avg_performance'] for group in universal_regions]
        cluster_counts = [group['cluster_count'] for group in universal_regions]
        
        # Debug information
        print(f"DEBUG: Creating chart for {len(universal_regions)} groups")
        print(f"DEBUG: Cluster counts: {cluster_counts}")
        print(f"DEBUG: Min count: {min(cluster_counts) if cluster_counts else 'N/A'}")
        print(f"DEBUG: Max count: {max(cluster_counts) if cluster_counts else 'N/A'}")
        print(f"DEBUG: Unique counts: {list(set(cluster_counts))}")
        
        fig = go.Figure()
        
        # Simple, safe color assignment
        colors_to_use = []
        color_palette = ['#2649B2', '#4A74F3', '#8E7DE3', '#9D5CE6', '#D4D9F0', '#6C8BE0', '#B55CE6']
        
        for i, count in enumerate(cluster_counts):
            color_index = (count - min(cluster_counts)) % len(color_palette) if cluster_counts else 0
            colors_to_use.append(color_palette[color_index])
        
        fig.add_trace(go.Bar(
            x=group_names,
            y=avg_performances,
            text=[f"{x:.1f}%" for x in avg_performances],
            textposition='outside',
            marker_color=colors_to_use,
            hovertemplate="<b>%{x}</b><br>" +
                        "Avg Performance: %{y:.1f}%<br>" +
                        "Cluster Count: %{customdata}<br>" +
                        "<extra></extra>",
            customdata=cluster_counts,
            name="Average Performance"
        ))
        
        fig.update_layout(
            title=f'Universal Region Groups Performance<br>Across {len(analyzed_clusters)} Skin Tone Clusters',
            xaxis_title='Universal Groups',
            yaxis_title='Average Performance Score (%)',
            xaxis={'tickangle': 45},
            height=500,
            showlegend=False
        )
        
        return fig


    def create_universal_color_swatches(self, centroid_colors, universal_regions):
        """Create color swatches for universal groups"""
        if len(centroid_colors) == 0:
            return None
        
        n_groups = len(centroid_colors)
        fig, axes = plt.subplots(n_groups, 3, figsize=(15, n_groups * 2.5))
        
        if n_groups == 1:
            axes = axes.reshape(1, -1)
        
        for i, centroid in enumerate(centroid_colors):
            try:
                # Main color centroid
                rgb_main = self._lab_to_rgb(centroid['centroid_main_L'], centroid['centroid_main_a'], centroid['centroid_main_b'])
                axes[i, 0].add_patch(Rectangle((0, 0), 1, 1, color=rgb_main))
                axes[i, 0].set_title(f"Main Color Centroid\nL={centroid['centroid_main_L']:.1f}, a={centroid['centroid_main_a']:.1f}, b={centroid['centroid_main_b']:.1f}", fontsize=10)
                axes[i, 0].axis('off')
                
                # Reflect color centroid
                rgb_reflect = self._lab_to_rgb(centroid['centroid_reflect_L'], centroid['centroid_reflect_a'], centroid['centroid_reflect_b'])
                axes[i, 1].add_patch(Rectangle((0, 0), 1, 1, color=rgb_reflect))
                axes[i, 1].set_title(f"Reflect Color Centroid\nL={centroid['centroid_reflect_L']:.1f}, a={centroid['centroid_reflect_a']:.1f}, b={centroid['centroid_reflect_b']:.1f}", fontsize=10)
                axes[i, 1].axis('off')
                
                # Group info
                group = next(g for g in universal_regions if g['group_name'] == centroid['group_name'])
                info_text = f"{centroid['group_name']}\n\n"
                info_text += f"Clusters: {centroid['cluster_count']}\n"
                info_text += f"Avg Performance: {group['avg_performance']:.1f}%\n"
                info_text += f"Total Samples: {group['total_samples']}\n"
                info_text += f"Avg ΔE: {group['avg_delta_e']:.2f}"
                
                axes[i, 2].text(0.05, 0.95, info_text, ha='left', va='top', fontsize=10, 
                            transform=axes[i, 2].transAxes, fontweight='bold')
                axes[i, 2].set_xlim(0, 1)
                axes[i, 2].set_ylim(0, 1)
                axes[i, 2].set_title("Group Information", fontsize=10)
                axes[i, 2].axis('off')
                
            except Exception as e:
                print(f"Error creating swatch for universal group {i}: {e}")
        
        plt.suptitle('Universal Color Groups - Centroid Colors', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig

    def create_universal_excel_data(self, universal_results, params):
        """Create Excel data for universal analysis"""
        universal_regions = universal_results['universal_regions']
        centroid_colors = universal_results['centroid_colors']
        cluster_coverage = universal_results['cluster_coverage']
        analyzed_clusters = universal_results['analyzed_clusters']
        
        # Universal regions data
        universal_excel_data = []
        for group in universal_regions:
            cluster_info = []
            region_info = []
            performance_info = []
            
            for cluster_id in sorted(group['cluster_regions'].keys()):
                region_data = group['cluster_regions'][cluster_id]
                cluster_info.append(f"ST{cluster_id}")
                region_info.append(f"R{region_data['region_id']}")
                performance_info.append(f"{region_data['performance']:.1f}%")
            
            universal_excel_data.append({
                'Group_Name': group['group_name'],
                'Cluster_Count': group['cluster_count'],
                'Clusters_Covered': ", ".join(cluster_info),
                'Region_IDs': ", ".join(region_info),
                'Performance_Scores': ", ".join(performance_info),
                'Avg_Performance': group['avg_performance'],
                'Min_Performance': group['min_performance'],
                'Max_Performance': group['max_performance'],
                'Total_Samples': group['total_samples'],
                'Avg_DeltaE': group['avg_delta_e'],
                'Max_DeltaE': group['max_delta_e']
            })
        
        # Centroid colors data
        centroid_excel_data = []
        for centroid in centroid_colors:
            centroid_excel_data.append({
                'Group_Name': centroid['group_name'],
                'Cluster_Count': centroid['cluster_count'],
                'Main_L': centroid['centroid_main_L'],
                'Main_a': centroid['centroid_main_a'],
                'Main_b': centroid['centroid_main_b'],
                'Reflect_L': centroid['centroid_reflect_L'],
                'Reflect_a': centroid['centroid_reflect_a'],
                'Reflect_b': centroid['centroid_reflect_b']
            })
        
        # Summary data
        summary_data = {
            'Metric': [
                'Analysis Type',
                'Metric Column Used',
                'Clusters Analyzed',
                'DeltaE Threshold',
                'Min Clusters Required',
                'Performance Threshold (%)',
                'Universal Groups Found',
                'Best Group',
                'Best Group Performance (%)',
                'Best Group Cluster Count',
                'Analysis Date'
            ],
            'Value': [
                'Universal Regions Analysis',
                params['metric_column'],
                ", ".join([f"ST{c}" for c in analyzed_clusters]),
                params['delta_e_threshold'],
                params['min_clusters_required'],
                params['min_performance_threshold'],
                len(universal_regions),
                universal_regions[0]['group_name'] if universal_regions else 'None',
                f"{universal_regions[0]['avg_performance']:.1f}" if universal_regions else 'N/A',
                universal_regions[0]['cluster_count'] if universal_regions else 'N/A',
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        
        return {
            'universal_regions': pd.DataFrame(universal_excel_data),
            'centroid_colors': pd.DataFrame(centroid_excel_data),
            'cluster_coverage': cluster_coverage,
            'summary': pd.DataFrame(summary_data)
        }