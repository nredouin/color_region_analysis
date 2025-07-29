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
    
    def compare_two_clusters(self, cluster_results_1, cluster_results_2, cluster_id_1, cluster_id_2, delta_e_threshold=3.0, top_n=10):
        """
        Compare two specific clusters using DeltaE analysis
        """
        self.delta_e_threshold = delta_e_threshold
        
        print(f"\n=== DEBUGGING CLUSTER COMPARISON ===")
        print(f"Comparing Cluster {cluster_id_1} vs Cluster {cluster_id_2}")
        print(f"DeltaE threshold: {delta_e_threshold}")
        
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
                
                # Calculate deltaE for main colors using our custom implementation
                delta_e_main = self._calculate_delta_e_cie2000_custom(lab_main_1, lab_main_2)
                
                # Calculate deltaE for reflect colors
                delta_e_reflect = self._calculate_delta_e_custom(lab_reflect_1, lab_reflect_2)
                
                if not pd.isna(delta_e_main):
                    comparison_results.append({
                        f'region_{cluster_id_1}_id': int(region_1_id),
                        f'region_{cluster_id_2}_id': int(region_2_id),
                        f'region_{cluster_id_1}_score': float(region_1['combined_criteria_pct']),
                        f'region_{cluster_id_2}_score': float(region_2['combined_criteria_pct']),
                        f'region_{cluster_id_1}_samples': int(region_1['total_samples']),
                        f'region_{cluster_id_2}_samples': int(region_2['total_samples']),
                        'delta_e_main': float(delta_e_main),
                        'delta_e_reflect': float(delta_e_reflect) if not pd.isna(delta_e_reflect) else np.nan,
                        'similar_main': bool(delta_e_main < delta_e_threshold),
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
            
            print(f"    Successful comparisons for this region: {comparisons_for_this_region}")
        
        print(f"\n=== COMPARISON SUMMARY ===")
        print(f"Total attempts: {total_attempts}")
        print(f"Successful comparisons: {successful_comparisons}")
        print(f"Generated {len(comparison_results)} comparison results")
        
        if len(comparison_results) == 0:
            # Provide detailed diagnostic information
            self._provide_diagnostic_info(top_1, centers_1, top_2, centers_2, cluster_id_1, cluster_id_2)
            raise ValueError(f"No valid comparisons could be made between clusters {cluster_id_1} and {cluster_id_2}. See diagnostic information above.")
        
        # Convert to DataFrame and filter similar colors
        comparison_df = pd.DataFrame(comparison_results)
        similar_pairs = comparison_df[comparison_df['similar_main'] == True].copy()
        similar_pairs = similar_pairs.sort_values('delta_e_main') if len(similar_pairs) > 0 else similar_pairs
        
        print(f"Similar main color pairs found (ΔE < {delta_e_threshold}): {len(similar_pairs)}")
        
        return {
            'comparison_df': comparison_df,
            'similar_pairs': similar_pairs,
            'cluster_id_1': cluster_id_1,
            'cluster_id_2': cluster_id_2,
            'delta_e_threshold': delta_e_threshold,
            'total_comparisons': len(comparison_df),
            'similar_count': len(similar_pairs),
            'similarity_rate': (len(similar_pairs) / len(comparison_df) * 100) if len(comparison_df) > 0 else 0
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
        
        # Main comparison data
        excel_data = []
        for _, row in comparison_df.iterrows():
            excel_data.append({
                f'Skin_Tone_{cluster_id_1}_Region': int(row[f'region_{cluster_id_1}_id']),
                f'Skin_Tone_{cluster_id_2}_Region': int(row[f'region_{cluster_id_2}_id']),
                f'ST{cluster_id_1}_Combined_Score': row[f'region_{cluster_id_1}_score'],
                f'ST{cluster_id_2}_Combined_Score': row[f'region_{cluster_id_2}_score'],
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