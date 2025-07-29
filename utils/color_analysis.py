import pandas as pd
import numpy as np
import os
import re
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import warnings
warnings.filterwarnings('ignore')

class ColorAnalyzer:
    """Main class for color region analysis"""
    
    def __init__(self):
        self.classification_path = 'data/classification_data.csv'
        
    def analyze_color_regions(self, file_path, threshold=45, top_n=10):
        """Main analysis function that returns comprehensive results"""
        try:
            # Load and process data
            df = self._load_and_process_data(file_path)
            if df is None:
                return None
            
            # Calculate statistics
            cluster_stats = self._calculate_cluster_statistics(df)
            
            # Get top regions
            top_regions = self._get_top_regions(cluster_stats, top_n)
            
            # Calculate cluster centers for colors
            cluster_centers = self._calculate_cluster_centers(df)
            
            # Ensure data type consistency between top_regions and cluster_centers
            top_regions, cluster_centers = self._ensure_data_consistency(top_regions, cluster_centers)
            
            # Exposure analysis
            exposure_data = self._calculate_exposure_analysis(df)
            
            # Family composition
            family_data = self._calculate_family_composition(df, top_regions)
            
            return {
                'original_data': df,
                'cluster_stats': cluster_stats,
                'top_regions': top_regions,
                'cluster_centers': cluster_centers,
                'exposure_data': exposure_data,
                'family_data': family_data,
                'threshold': threshold
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def _ensure_data_consistency(self, top_regions, cluster_centers):
        """Ensure data type consistency between top_regions and cluster_centers"""
        try:
            # Ensure color_regions column exists and is properly typed
            if 'color_regions' in top_regions.columns:
                # Convert both to consistent integer type
                top_regions['color_regions'] = pd.to_numeric(top_regions['color_regions'], errors='coerce').astype('Int64')
                cluster_centers.index = pd.to_numeric(cluster_centers.index, errors='coerce').astype('Int64')
                
                # Remove any NaN regions
                top_regions = top_regions.dropna(subset=['color_regions'])
                cluster_centers = cluster_centers.dropna()
                
                print(f"Data consistency check:")
                print(f"Top regions: {len(top_regions)} regions")
                print(f"Cluster centers: {len(cluster_centers)} regions")
                print(f"Overlapping regions: {len(set(top_regions['color_regions']) & set(cluster_centers.index))}")
                
            return top_regions, cluster_centers
            
        except Exception as e:
            print(f"Error ensuring data consistency: {e}")
            return top_regions, cluster_centers
    
    def _load_and_process_data(self, file_path):
        """Load and preprocess the main data file"""
        try:
            # Load main data
            df = pd.read_csv(file_path, sep=";")
            print(f"Data loaded successfully! Shape: {df.shape}")
            
            # Check required columns
            required_columns = ['color_regions', 'S1R']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Error: Missing required columns: {missing_columns}")
                print(f"Available columns: {list(df.columns)}")
                return None
            
            # Handle S2R column
            if 'S2R' not in df.columns:
                print("Warning: S2R column not found. Creating with NaN values.")
                df['S2R'] = np.nan
            
            # Ensure color_regions is numeric
            df['color_regions'] = pd.to_numeric(df['color_regions'], errors='coerce')
            df = df.dropna(subset=['color_regions'])
            
            # Load classification data if available
            try:
                classification_df = pd.read_csv(self.classification_path, sep=';', encoding='latin1')
                print(f"Classification data loaded successfully! Shape: {classification_df.shape}")
                df = self._merge_classification_data(df, classification_df)
                print(f"Data after classification merge: {df.shape}")
            except Exception as e:
                print(f"Warning: Could not load classification data: {e}")
                df['Family_MCB'] = 'Unknown'
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def _merge_classification_data(self, df, classification_df):
        """Merge with classification data"""
        # Extract numeric values from XSHADE_S column
        if 'XSHADE_S' in df.columns:
            df['XSHADE_S_numeric'] = df['XSHADE_S'].apply(self._extract_numeric_from_xshade)
            
            # Merge with classification data
            df = df.merge(
                classification_df[['id', 'FAMILY', 'Family_MCB']], 
                left_on='XSHADE_S_numeric', 
                right_on='id', 
                how='left'
            )
        else:
            print("Warning: XSHADE_S column not found. Skipping classification merge.")
            df['Family_MCB'] = 'Unknown'
        
        return df
    
    def _extract_numeric_from_xshade(self, xshade_value):
        """Extract numeric values from XSHADE_S column"""
        if pd.isna(xshade_value):
            return np.nan
        match = re.match(r'^(\d+)', str(xshade_value).strip())
        if match:
            return int(match.group(1))
        return np.nan
    
    def _calculate_cluster_statistics(self, df):
        """Calculate comprehensive statistics for each color region"""
        def calculate_corrected_stats(group):
            total_samples = len(group['S1R'].dropna())
            
            # S1R analysis
            s1r_top2box_count = len(group[group['S1R'] >= 4]['S1R'].dropna())
            s1r_top2box_pct = (s1r_top2box_count / total_samples * 100) if total_samples > 0 else 0
            s1r_good_count = len(group[group['S1R'] >= 3]['S1R'].dropna())
            s1r_good_pct = (s1r_good_count / total_samples * 100) if total_samples > 0 else 0
            
            # S2R analysis
            s2r_values = group['S2R'].dropna()
            s2r_total = len(s2r_values)
            
            if s2r_total > 0:
                s2r_top2box_count = len(s2r_values[s2r_values >= 2])
                s2r_top2box_pct = (s2r_top2box_count / s2r_total * 100)
                s2r_good_count = len(s2r_values[s2r_values >= 2])
                s2r_mean = s2r_values.mean()
            else:
                s2r_top2box_count = 0
                s2r_top2box_pct = np.nan
                s2r_good_count = 0
                s2r_mean = np.nan
            
            # Combined criteria
            if total_samples > 0:
                both_criteria_samples = group[(group['S1R'] >= 3) & (group['S2R'] >= 2)]
                combined_criteria_count = len(both_criteria_samples)
                combined_criteria_pct = (combined_criteria_count / total_samples * 100)
                s1r_3plus_pct = (s1r_good_count / total_samples * 100)
                s2r_2plus_of_all_pct = (s2r_good_count / total_samples * 100)
            else:
                combined_criteria_count = 0
                combined_criteria_pct = np.nan
                s1r_3plus_pct = np.nan
                s2r_2plus_of_all_pct = np.nan
            
            return pd.Series({
                'total_samples': total_samples,
                'S1R_mean': group['S1R'].mean(),
                'S1R_top2box_pct': s1r_top2box_pct,
                'S1R_top2box_count': s1r_top2box_count,
                'S1R_3plus_pct': s1r_3plus_pct,
                'S1R_3plus_count': s1r_good_count,
                'S2R_mean': s2r_mean,
                'S2R_top2box_pct': s2r_top2box_pct,
                'S2R_top2box_count': s2r_top2box_count,
                'S2R_total': s2r_total,
                'S2R_data_pct': (s2r_total / total_samples * 100) if total_samples > 0 else 0,
                'S2R_2plus_of_all_pct': s2r_2plus_of_all_pct,
                'combined_criteria_pct': combined_criteria_pct,
                'combined_criteria_count': combined_criteria_count,
            })
        
        cluster_stats = df.groupby('color_regions').apply(calculate_corrected_stats).reset_index()
        return cluster_stats
    
    def _get_top_regions(self, cluster_stats, top_n):
        """Get top N performing regions"""
        clusters_with_data = cluster_stats[cluster_stats['total_samples'] > 0].copy()
        top_regions = clusters_with_data.dropna(subset=['combined_criteria_pct']).nlargest(top_n, 'combined_criteria_pct')
        return top_regions
    
    def _calculate_cluster_centers(self, df):
        """Calculate cluster centers for color visualization"""
        color_columns = ['L_main', 'a_main', 'b_main', 'L_reflect', 'a_reflect', 'b_reflect']
        
        # Check which color columns exist
        available_color_columns = [col for col in color_columns if col in df.columns]
        
        if not available_color_columns:
            print("Warning: No color columns found in data")
            return pd.DataFrame()
        
        print(f"Using color columns: {available_color_columns}")
        
        cluster_centers = df.groupby('color_regions')[available_color_columns].mean()
        
        # Fill missing columns with NaN if some are missing
        for col in color_columns:
            if col not in cluster_centers.columns:
                cluster_centers[col] = np.nan
                
        return cluster_centers
    
    def _calculate_exposure_analysis(self, df):
        """Calculate exposure analysis"""
        exposure_data = {}
        
        if 'RESP_FINAL' in df.columns:
            df_clean_exposure = df.dropna(subset=['RESP_FINAL', 'color_regions']).copy()
            total_unique_women = df_clean_exposure['RESP_FINAL'].nunique()
            
            cluster_exposure = df_clean_exposure.groupby('color_regions')['RESP_FINAL'].nunique().reset_index()
            cluster_exposure.columns = ['Color_Region', 'Unique_Women_Count']
            cluster_exposure['Percentage_of_Women'] = (cluster_exposure['Unique_Women_Count'] / total_unique_women * 100).round(2)
            
            # Convert to dictionary for easy lookup
            for _, row in cluster_exposure.iterrows():
                exposure_data[row['Color_Region']] = {
                    'unique_women': row['Unique_Women_Count'],
                    'percentage_women': row['Percentage_of_Women']
                }
        
        return exposure_data
    
    def _calculate_family_composition(self, df, top_regions):
        """Calculate family composition for top regions"""
        family_data = {}
        
        if 'Family_MCB' in df.columns and 'color_regions' in top_regions.columns:
            for color_region in top_regions['color_regions']:
                region_data = df[df['color_regions'] == color_region]
                if len(region_data) > 0:
                    family_counts = region_data['Family_MCB'].value_counts()
                    total_in_region = len(region_data)
                    
                    family_composition = []
                    for family_mcb, count in family_counts.head(5).items():  # Top 5 families
                        if not pd.isna(family_mcb):
                            proportion = (count / total_in_region * 100)
                            family_composition.append({
                                'family': family_mcb,
                                'count': count,
                                'percentage': round(proportion, 2)
                            })
                    
                    family_data[color_region] = family_composition
        
        return family_data