import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from matplotlib.colors import LinearSegmentedColormap
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from matplotlib.patches import Rectangle
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# Import our custom modules
from utils.color_analysis import ColorAnalyzer
from utils.hair_rendering import HairRenderer
from utils.streamlit_helpers import StreamlitHelpers
from utils.cluster_comparison import ClusterComparator

# Page configuration
st.set_page_config(
    page_title="L'Oréal Multi-Cluster Hair Color Analysis",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2649B2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .region-card {
        border: 2px solid #D4D9F0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .color-swatch {
        width: 50px;
        height: 50px;
        border-radius: 5px;
        border: 2px solid #333;
        display: inline-block;
        margin-right: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4A74F3;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 L\'Oréal Multi-Cluster Hair Color Analysis & Comparison</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📊 Analysis Configuration")
        
        # Find available clusters
        available_clusters = []
        results_folder = "results"
        if os.path.exists(results_folder):
            for file in os.listdir(results_folder):
                if file.startswith("df_100_eval_") and file.endswith(".csv"):
                    match = re.search(r'df_100_eval_(\d+)_min_100_minwomen_50\.0_s1r_0\.0\.csv', file)
                    if match:
                        available_clusters.append(int(match.group(1)))
        
        available_clusters.sort()
        
        if not available_clusters:
            st.error("No data files found in results folder!")
            st.stop()
        
        # Analysis mode selection
        st.subheader("🔬 Analysis Mode")
        analysis_mode = st.radio(
            "Choose Analysis Type",
            ["Single Cluster", "Compare Clusters (DeltaE)", "Universal Regions"],
            help="Single: Analyze one cluster | Compare: DeltaE similarity analysis | Universal: Find regions across all clusters"
        )
        
        # Metric selection (NEW FEATURE)
        st.subheader("📈 Performance Metric")
        metric_type = st.radio(
            "Select Performance Metric:",
            ["Combined Criteria", "S1R Top2Box Only", "S2R Top2Box Only"],
            help="Combined: Uses both S1R≥3 AND S2R≥2 criteria | S1R: Only S1R top2box percentage | S2R: Only S2R top2box percentage"
        )
        
        # Get metric column name and description
        if metric_type == "Combined Criteria":
            metric_column = "combined_criteria_pct"
            metric_description = "Combined S1R≥3 AND S2R≥2"
        elif metric_type == "S1R Top2Box Only":
            metric_column = "S1R_top2box_pct"
            metric_description = "S1R Top2Box percentage"
        else:  # S2R Top2Box Only
            metric_column = "S2R_top2box_pct"
            metric_description = "S2R Top2Box percentage"
        
        # Display selected metric info
        st.info(f"**Selected Metric**: {metric_description}")
        
        # Cluster selection based on mode
        if analysis_mode == "Single Cluster":
            selected_clusters = [st.selectbox(
                "🎯 Select Skin Tone Cluster",
                available_clusters,
                help="Choose which skin tone cluster to analyze"
            )]
            
        elif analysis_mode == "Compare Clusters (DeltaE)":
            st.info("You'll select clusters to compare on the main page")
            selected_clusters = available_clusters  # Load all for comparison
            
        else:  # Universal Regions
            selected_clusters = st.multiselect(
                "🌍 Clusters for Universal Analysis",
                available_clusters,
                default=available_clusters,
                help="Select clusters to find universal color regions"
            )
            
            if len(selected_clusters) < 2:
                st.warning("Please select at least 2 clusters for universal analysis!")
                st.stop()
        
        # Analysis parameters
        st.subheader("⚙️ Analysis Parameters")
        threshold = st.slider(
            f"{metric_description} Threshold (%)",
            min_value=0,
            max_value=100,
            value=45,
            help=f"Minimum percentage for {metric_description}"
        )
        
        top_n_regions = st.slider(
            "Number of Top Regions per Cluster",
            min_value=5,
            max_value=20,
            value=10
        )
        
        # Universal region parameters (only for universal mode)
        if analysis_mode == "Universal Regions":
            st.subheader("🌍 Universal Region Settings")
            min_clusters_required = st.slider(
                "Minimum Clusters Required",
                min_value=2,
                max_value=len(selected_clusters),
                value=min(3, len(selected_clusters)),
                help="Minimum number of clusters where a region must perform well"
            )
            
            universal_threshold = st.slider(
                "Universal Performance Threshold (%)",
                min_value=30,
                max_value=80,
                value=40,
                help="Minimum performance required in each cluster"
            )
        
        # Hair rendering options
        st.subheader("🎨 Hair Rendering")
        auto_render = st.checkbox(
            "Auto-render top performing regions",
            value=True,
            help="Automatically generate hair renderings for top regions"
        )
        
        swatch_type = st.selectbox(
            "Hair Swatch Template",
            ["Dark", "Medium", "Light"],
            index=1
        )

    # Initialize components
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ColorAnalyzer()
        st.session_state.renderer = HairRenderer()
        st.session_state.helpers = StreamlitHelpers()
        st.session_state.comparator = ClusterComparator()

    # Load and analyze data for selected clusters
    cluster_results = {}
    
    if analysis_mode != "Compare Clusters (DeltaE)":
        with st.spinner(f"Loading data for {len(selected_clusters)} cluster(s)..."):
            for cluster_id in selected_clusters:
                file_path = f"results/df_100_eval_{cluster_id}_min_100_minwomen_50.0_s1r_0.0.csv"
                
                try:
                    # Fixed: Now properly passing metric_column as keyword argument
                    analysis_results = st.session_state.analyzer.analyze_color_regions(
                        file_path, 
                        threshold=threshold, 
                        top_n=top_n_regions, 
                        metric_column=metric_column
                    )
                    
                    if analysis_results is not None:
                        cluster_results[cluster_id] = analysis_results
                        
                except Exception as e:
                    st.error(f"Error loading data for cluster {cluster_id}: {str(e)}")
        if not cluster_results and analysis_mode != "Compare Clusters (DeltaE)":
            st.error("Failed to load any cluster data!")
            st.stop()

    # Main content based on analysis mode
    if analysis_mode == "Single Cluster":
        show_single_cluster_analysis(cluster_results, selected_clusters[0], threshold, swatch_type, auto_render, metric_column, metric_description)
        
    elif analysis_mode == "Compare Clusters (DeltaE)":
        show_cluster_comparison_analysis(available_clusters, threshold, metric_column, metric_description)
        
    else:  # Universal Regions
        show_universal_regions_analysis(
            cluster_results, threshold, min_clusters_required, 
            universal_threshold, swatch_type, metric_column, metric_description
        )

    # Export section (only for non-comparison modes)
    if analysis_mode != "Compare Clusters (DeltaE)":
        st.markdown("---")
        create_export_section(cluster_results, selected_clusters, analysis_mode, metric_type)

def show_single_cluster_analysis(cluster_results, cluster_id, threshold, swatch_type, auto_render, metric_column, metric_description):
    """Show single cluster analysis (original functionality)"""
    analysis_results = cluster_results[cluster_id]
    
    # Initialize selected region state
    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = None
    
    # Main dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📈 Top Regions - Cluster {cluster_id}")
        st.info(f"**Performance Metric**: {metric_description}")
        
        # Performance overview chart
        fig = create_performance_overview(analysis_results, threshold, metric_column, metric_description)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive region selection
        st.subheader("🎯 Select a Region for Detailed Analysis")
        selected_region = create_region_selector(analysis_results, metric_column)
        
    with col2:
        st.header("📊 Cluster Overview")
        
        # Summary metrics
        total_regions = len(analysis_results['top_regions'])
        meeting_criteria = len(analysis_results['top_regions'][
            analysis_results['top_regions'][metric_column] >= threshold
        ])
        
        st.metric("Total Regions Analyzed", total_regions)
        st.metric("Meeting Criteria", meeting_criteria)
        st.metric("Best Performance", 
                 f"{analysis_results['top_regions'].iloc[0][metric_column]:.1f}%")

    # Detailed analysis section
    if selected_region is not None:
        st.markdown("---")
        show_detailed_analysis(selected_region, analysis_results, cluster_id, swatch_type, auto_render, metric_column)

def show_detailed_analysis(selected_region, analysis_results, cluster_id, swatch_type, auto_render, metric_column):
    """Show detailed analysis for a selected region"""
    try:
        st.header(f"🔍 Detailed Analysis - Region {selected_region}")
        
        # Get region data
        top_regions = analysis_results['top_regions']
        cluster_centers = analysis_results['cluster_centers']
        original_data = analysis_results['original_data']
        exposure_data = analysis_results.get('exposure_data', {})
        family_data = analysis_results.get('family_data', {})
        
        # Find region in top_regions
        region_data = top_regions[top_regions['color_regions'] == selected_region]
        if len(region_data) == 0:
            st.error(f"Region {selected_region} not found in analysis results!")
            return
        
        region_info = region_data.iloc[0]
        
        # Check if region exists in cluster_centers
        if selected_region not in cluster_centers.index:
            st.error(f"Color data for region {selected_region} not found!")
            return
        
        center_data = cluster_centers.loc[selected_region]
        
        # Layout for detailed analysis
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("📊 Performance Metrics")
            
            # Performance metrics - show the selected metric prominently
            st.metric("Selected Metric Score", f"{region_info[metric_column]:.1f}%")
            st.metric("Total Samples", int(region_info['total_samples']))
            
            # Show all available metrics for comparison
            st.write("**All Available Metrics:**")
            st.metric("S1R Mean Rating", f"{region_info['S1R_mean']:.2f}")
            st.metric("S1R TOP2BOX %", f"{region_info['S1R_top2box_pct']:.1f}%")
            
            if not pd.isna(region_info.get('S2R_mean', np.nan)):
                st.metric("S2R Mean Rating", f"{region_info['S2R_mean']:.2f}")
                st.metric("S2R TOP2BOX %", f"{region_info['S2R_top2box_pct']:.1f}%")
                st.metric("S2R Data Coverage", f"{region_info['S2R_data_pct']:.1f}%")
            
            if 'combined_criteria_pct' in region_info.index:
                st.metric("Combined Criteria", f"{region_info['combined_criteria_pct']:.1f}%")
            
            # Exposure data
            if selected_region in exposure_data:
                exp_data = exposure_data[selected_region]
                st.metric("Unique Women Exposed", int(exp_data['unique_women']))
                st.metric("% of All Women", f"{exp_data['percentage_women']:.1f}%")
        
        with col2:
            st.subheader("🎨 Color Information")
            
            # Color swatches
            try:
                # Main color
                rgb_main = st.session_state.helpers.lab_to_rgb(
                    center_data['L_main'], center_data['a_main'], center_data['b_main']
                )
                
                # Reflect color
                rgb_reflect = st.session_state.helpers.lab_to_rgb(
                    center_data['L_reflect'], center_data['a_reflect'], center_data['b_reflect']
                )
                
                # Create visual representation
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
                
                ax1.add_patch(Rectangle((0, 0), 1, 1, color=rgb_main))
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                ax1.set_title("Main Color", fontweight='bold')
                ax1.axis('off')
                
                ax2.add_patch(Rectangle((0, 0), 1, 1, color=rgb_reflect))
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.set_title("Reflect Color", fontweight='bold')
                ax2.axis('off')
                
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
                
            except Exception as e:
                st.error(f"Error creating color swatches: {e}")
            
            # LAB values table
            st.subheader("LAB Color Values")
            lab_data = {
                'Component': ['L* (Lightness)', 'a* (Green-Red)', 'b* (Blue-Yellow)'],
                'Main Color': [
                    f"{center_data['L_main']:.2f}",
                    f"{center_data['a_main']:.2f}",
                    f"{center_data['b_main']:.2f}"
                ],
                'Reflect Color': [
                    f"{center_data['L_reflect']:.2f}",
                    f"{center_data['a_reflect']:.2f}",
                    f"{center_data['b_reflect']:.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(lab_data), use_container_width=True)
            
            # Chroma and Hue calculations
            chroma_main, hue_main = st.session_state.helpers.calculate_chroma_hue(
                center_data['a_main'], center_data['b_main']
            )
            chroma_reflect, hue_reflect = st.session_state.helpers.calculate_chroma_hue(
                center_data['a_reflect'], center_data['b_reflect']
            )
            
            if not pd.isna(chroma_main):
                st.write(f"**Main Color**: Chroma = {chroma_main:.2f}, Hue = {hue_main:.1f}°")
            if not pd.isna(chroma_reflect):
                st.write(f"**Reflect Color**: Chroma = {chroma_reflect:.2f}, Hue = {hue_reflect:.1f}°")
        
        with col3:
            st.subheader("👥 Family Composition")
            
            if selected_region in family_data and family_data[selected_region]:
                family_composition = family_data[selected_region]
                
                for family_info in family_composition[:5]:  # Top 5 families
                    st.write(f"**{family_info['family']}**: {family_info['count']} samples ({family_info['percentage']:.1f}%)")
            else:
                st.info("No family composition data available")
        
        # Performance radar chart
        st.subheader("📈 Performance Radar Chart")
        try:
            radar_fig = st.session_state.helpers.create_performance_radar(region_info)
            st.plotly_chart(radar_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create radar chart: {e}")
        
        # Hair rendering section
        st.subheader("💇‍♀️ Hair Color Rendering")
        
        render_button_col1, render_button_col2 = st.columns(2)
        
        with render_button_col1:
            if st.button(f"🎨 Render Hair Swatch - Region {selected_region}", type="primary"):
                render_hair_for_region(selected_region, center_data, swatch_type)
        
        with render_button_col2:
            if auto_render:
                st.info("Auto-rendering enabled for top regions")
        
        # Sample data from this region
        st.subheader("📋 Sample Data")
        region_samples = original_data[original_data['color_regions'] == selected_region]
        
        if len(region_samples) > 0:
            # Show basic stats
            st.write(f"**Total samples in this region**: {len(region_samples)}")
            
            # Show sample of the data
            display_columns = ['S1R', 'S2R', 'L_main', 'a_main', 'b_main', 'L_reflect', 'a_reflect', 'b_reflect']
            available_columns = [col for col in display_columns if col in region_samples.columns]
            
            if available_columns:
                sample_data = region_samples[available_columns].head(10)
                st.dataframe(sample_data, use_container_width=True)
        else:
            st.warning("No sample data found for this region")
        
    except Exception as e:
        st.error(f"Error in detailed analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def render_hair_for_region(region_id, center_data, swatch_type):
    """Render hair for a specific region"""
    try:
        with st.spinner("Rendering hair swatch..."):
            # Prepare data for rendering
            render_data = pd.DataFrame({
                'model': [f"region_{region_id}"],
                'cluster': [region_id],
                'L_1': [center_data['L_main']],
                'a_1': [center_data['a_main']],
                'b_1': [center_data['b_main']],
                'L_2': [center_data['L_reflect']],
                'a_2': [center_data['a_reflect']],
                'b_2': [center_data['b_reflect']]
            })
            
            # Save temp data
            os.makedirs("assets", exist_ok=True)
            temp_path = "assets/temp_region.xlsx"
            render_data.to_excel(temp_path, index=False)
            
            # Render hair
            rendered_path = st.session_state.renderer.render_hair_swatch(
                temp_path, swatch_type, f"region_{region_id}"
            )
            
            if rendered_path and os.path.exists(rendered_path):
                st.success("✅ Hair rendering completed!")
                st.image(rendered_path, caption=f"Region {region_id} Hair Rendering")
                
                # Download button
                with open(rendered_path, "rb") as file:
                    st.download_button(
                        "📥 Download Rendered Image",
                        file.read(),
                        file_name=f"region_{region_id}_hair_rendering.jpg",
                        mime="image/jpeg"
                    )
            else:
                st.error("❌ Failed to generate hair rendering")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def show_cluster_comparison_analysis(available_clusters, threshold, metric_column, metric_description):
    """Show comparison analysis between two clusters using DeltaE"""
    st.header("🔍 Cluster Comparison Analysis (DeltaE Method)")
    st.info(f"**Performance Metric**: {metric_description}")
    
    # Cluster selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cluster_1 = st.selectbox(
            "Select First Cluster",
            available_clusters,
            index=0,
            key="cluster_1_select"
        )
    
    with col2:
        cluster_2 = st.selectbox(
            "Select Second Cluster", 
            available_clusters,
            index=1 if len(available_clusters) > 1 else 0,
            key="cluster_2_select"
        )
    
    with col3:
        delta_e_threshold = st.slider(
            "Main Color DeltaE Threshold",  # Updated label
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Lower values = more similar main colors required"
        )
    
    if cluster_1 == cluster_2:
        st.warning("Please select two different clusters for comparison!")
        return
    
    # Comparison parameters
    st.subheader("🔧 Comparison Parameters")
    col1, col2, col3 = st.columns(3)  # Changed to 3 columns
    
    with col1:
        top_n_compare = st.slider(
            "Top N regions per cluster to compare",
            min_value=5,
            max_value=20,
            value=10
        )
    
    with col2:
        max_pairs_display = st.slider(
            "Max similar pairs to display",
            min_value=3,
            max_value=15,
            value=8
        )
    
    with col3:  # NEW: Reflect threshold multiplier
        reflect_multiplier = st.slider(
            "Reflect threshold multiplier",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Reflect threshold = Main threshold × this multiplier"
        )
    
    # Calculate reflect threshold
    reflect_threshold = delta_e_threshold * reflect_multiplier
    st.info(f"🎯 **Thresholds**: Main Colors ≤ {delta_e_threshold:.1f}, Reflect Colors ≤ {reflect_threshold:.1f}")
    
    # Rest of your code remains the same until the comparison call...
    
    # Perform comparison
    if st.button("🔍 Perform DeltaE Comparison", type="primary"):
        with st.spinner(f"Comparing clusters {cluster_1} and {cluster_2}..."):
            
            try:
                # Load data for both clusters (unchanged)
                cluster_results = {}
                for cluster_id in [cluster_1, cluster_2]:
                    file_path = f"results/df_100_eval_{cluster_id}_min_100_minwomen_50.0_s1r_0.0.csv"
                    
                    analysis_results = st.session_state.analyzer.analyze_color_regions(
                                            file_path, 
                                            threshold=threshold, 
                                            top_n=top_n_compare, 
                                            metric_column=metric_column
                                        )
                    
                    if analysis_results is None:
                        st.error(f"Failed to analyze data for cluster {cluster_id}")
                        return
                        
                    cluster_results[cluster_id] = analysis_results
                
                if len(cluster_results) != 2:
                    st.error("Failed to load data for both clusters!")
                    return
                
                # Perform comparison with reflect threshold
                comparison_results = st.session_state.comparator.compare_two_clusters(
                    {cluster_1: cluster_results[cluster_1]},
                    {cluster_2: cluster_results[cluster_2]}, 
                    cluster_1, 
                    cluster_2, 
                    delta_e_threshold, 
                    top_n_compare,
                    metric_column,
                    reflect_threshold  # NEW: Add reflect threshold
                )
                
                # Store results in session state (add new values)
                st.session_state.comparison_results = comparison_results
                st.session_state.cluster_1 = cluster_1
                st.session_state.cluster_2 = cluster_2
                st.session_state.cluster_results_comparison = cluster_results
                st.session_state.max_pairs_display = max_pairs_display
                st.session_state.metric_column = metric_column
                st.session_state.metric_description = metric_description
                st.session_state.reflect_multiplier = reflect_multiplier  # NEW
                st.session_state.reflect_threshold = reflect_threshold  # NEW
                
                st.success("✅ Comparison analysis completed!")
                
            except Exception as e:
                st.error(f"Error during comparison: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Display results if available (unchanged)
    if 'comparison_results' in st.session_state:
        show_comparison_results()
        show_similar_pairs_recap()

def show_comparison_results():
    """Display comparison results"""
    comparison_results = st.session_state.comparison_results
    similar_pairs = comparison_results['similar_pairs']
    both_criteria_pairs = comparison_results['both_criteria_pairs']  # NEW
    comparison_df = comparison_results['comparison_df']
    cluster_1 = st.session_state.cluster_1
    cluster_2 = st.session_state.cluster_2
    max_pairs_display = st.session_state.get('max_pairs_display', 8)
    metric_column = st.session_state.get('metric_column', 'combined_criteria_pct')
    metric_description = st.session_state.get('metric_description', 'Combined Criteria')
    
    # Results overview
    st.markdown("---")
    st.subheader("📊 Comparison Results")
    st.info(f"**Performance Metric Used**: {metric_description}")
    
    # Updated metrics display
    col1, col2, col3, col4, col5 = st.columns(5)  # Added one more column
    
    with col1:
        st.metric("Total Comparisons", comparison_results['total_comparisons'])
    
    with col2:
        st.metric("Main Similar", comparison_results['main_similar_count'])
    
    with col3:
        st.metric("Both Criteria Met", comparison_results['both_criteria_count'])  # NEW
    
    with col4:
        st.metric("Reflect Checked", comparison_results['reflect_checked_count'])  # NEW
    
    with col5:
        if len(both_criteria_pairs) > 0:
            best_match = both_criteria_pairs.iloc[0]
            st.metric("Best Match ΔE", f"{best_match['delta_e_main']:.2f}")
        elif len(similar_pairs) > 0:
            best_match = similar_pairs.iloc[0]
            st.metric("Best Match ΔE", f"{best_match['delta_e_main']:.2f}")
        else:
            st.metric("Best Match ΔE", "N/A")
    
    # Display threshold information
    main_threshold = comparison_results.get('main_delta_e_threshold', comparison_results.get('delta_e_threshold', 3.0))
    reflect_threshold = comparison_results.get('reflect_delta_e_threshold', main_threshold * 2)
    
    st.info(f"🎯 **Applied Thresholds**: Main Colors ≤ {main_threshold:.1f}, Reflect Colors ≤ {reflect_threshold:.1f}")
    
    # Success rates
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Main Similarity Rate", f"{comparison_results.get('main_similarity_rate', 0):.1f}%")
    with col2:
        st.metric("Both Criteria Rate", f"{comparison_results.get('both_criteria_rate', 0):.1f}%")
    with col3:
        reflect_check_rate = (comparison_results['reflect_checked_count'] / comparison_results['total_comparisons'] * 100) if comparison_results['total_comparisons'] > 0 else 0
        st.metric("Reflect Check Rate", f"{reflect_check_rate:.1f}%")
    
    # Display options for different pair types
    st.subheader("📋 Analysis Options")
    analysis_type = st.radio(
        "Choose pairs to analyze:",
        ["Both Criteria Met", "Main Color Similar", "All Comparisons"],
        index=0 if len(both_criteria_pairs) > 0 else 1
    )
    
    # Select which pairs to display based on choice
    if analysis_type == "Both Criteria Met" and len(both_criteria_pairs) > 0:
        display_pairs = both_criteria_pairs
        pairs_description = f"pairs meeting BOTH main (≤{main_threshold:.1f}) AND reflect (≤{reflect_threshold:.1f}) criteria"
    elif analysis_type == "Main Color Similar" and len(similar_pairs) > 0:
        display_pairs = similar_pairs
        pairs_description = f"pairs with similar main colors (≤{main_threshold:.1f})"
    else:
        display_pairs = comparison_df
        pairs_description = "all region pairs compared"
    
    st.info(f"Showing {len(display_pairs)} {pairs_description}")
    
    # DeltaE scatter plot (this should work with your existing function)
    st.subheader("📈 DeltaE Analysis")
    fig_scatter = st.session_state.comparator.create_delta_e_scatter_plotly(
        comparison_df, display_pairs, cluster_1, cluster_2
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Similar pairs visualization
    if len(display_pairs) > 0:
        st.subheader("🎨 Similar Color Pairs")
        
        # Color comparison plot
        fig_colors = st.session_state.comparator.create_color_comparison_plot(
            display_pairs, cluster_1, cluster_2, max_pairs_display
        )
        
        if fig_colors:
            st.pyplot(fig_colors, clear_figure=True)
        
        # Similar pairs table with new columns
        st.subheader("📋 Detailed Pairs Table")
        
        # Format pairs for display with new columns
        display_pairs_copy = display_pairs.copy()
        display_cols = [
            f'region_{cluster_1}_id', f'region_{cluster_2}_id', 
            'delta_e_main', 'delta_e_reflect',
            'similar_main', 'similar_reflect', 'meets_both_criteria',  # NEW columns
            f'region_{cluster_1}_score', f'region_{cluster_2}_score',
            f'region_{cluster_1}_samples', f'region_{cluster_2}_samples'
        ]
        
        # Only include columns that exist
        display_cols = [col for col in display_cols if col in display_pairs_copy.columns]
        
        display_df = display_pairs_copy[display_cols].round(2)
        
        # Update column names
        new_col_names = []
        for col in display_cols:
            if col == f'region_{cluster_1}_id':
                new_col_names.append(f'ST{cluster_1} Region')
            elif col == f'region_{cluster_2}_id':
                new_col_names.append(f'ST{cluster_2} Region')
            elif col == 'delta_e_main':
                new_col_names.append('ΔE Main')
            elif col == 'delta_e_reflect':
                new_col_names.append('ΔE Reflect')
            elif col == 'similar_main':
                new_col_names.append('Main Similar')
            elif col == 'similar_reflect':
                new_col_names.append('Reflect Similar')
            elif col == 'meets_both_criteria':
                new_col_names.append('Both Criteria')
            elif col == f'region_{cluster_1}_score':
                new_col_names.append(f'ST{cluster_1} Score %')
            elif col == f'region_{cluster_2}_score':
                new_col_names.append(f'ST{cluster_2} Score %')
            elif col == f'region_{cluster_1}_samples':
                new_col_names.append(f'ST{cluster_1} Samples')
            elif col == f'region_{cluster_2}_samples':
                new_col_names.append(f'ST{cluster_2} Samples')
            else:
                new_col_names.append(col)
        
        display_df.columns = new_col_names
        st.dataframe(display_df, use_container_width=True)
        
        # Hair rendering for best match
        st.subheader("💇‍♀️ Hair Rendering - Best Match")
        
        if st.button("🎨 Render Best Matching Pair"):
            render_best_matching_pair(display_pairs, st.session_state.cluster_results_comparison, cluster_1, cluster_2)
    
    else:
        threshold_used = comparison_results.get('main_delta_e_threshold', comparison_results.get('delta_e_threshold', 3.0))
        if analysis_type == "Both Criteria Met":
            st.warning(f"No pairs found meeting BOTH criteria (Main ≤{main_threshold:.1f} AND Reflect ≤{reflect_threshold:.1f})")
            st.info("Try increasing the thresholds or check 'Main Color Similar' pairs.")
        else:
            st.warning(f"No similar color pairs found with Main ΔE threshold < {threshold_used}")
            st.info("Try increasing the DeltaE threshold to find more matches.")

def show_similar_pairs_recap():
    """Display the similar pairs recap analysis"""
    if 'comparison_results' not in st.session_state:
        return
    
    comparison_results = st.session_state.comparison_results
    cluster_1 = st.session_state.cluster_1
    cluster_2 = st.session_state.cluster_2
    
    # Generate recap analysis
    with st.spinner("Creating similar pairs recap..."):
        recap_results = st.session_state.comparator.create_similar_pairs_recap(
            comparison_results, cluster_1, cluster_2
        )
        
        # Store in session state
        st.session_state.recap_results = recap_results
    
    st.markdown("---")
    st.subheader("🔄 Similar Pairs Recap & Centroid Analysis")
    st.info("This section shows regions that match with multiple regions from the other cluster and their centroid colors.")
    
    grouped_pairs = recap_results['grouped_pairs']
    centroid_data = recap_results['centroid_data']
    summary_stats = recap_results['summary_stats']
    
    if len(grouped_pairs) == 0:
        st.warning("No multi-match groups found. Each region only matches with one region from the other cluster.")
        return
    
    # Summary metrics
    st.subheader("📊 Multi-Match Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Multi-Match Groups", summary_stats['total_groups'])
    
    with col2:
        st.metric(f"ST{cluster_1} Multi-Match", summary_stats['cluster_1_multi_match_regions'])
    
    with col3:
        st.metric(f"ST{cluster_2} Multi-Match", summary_stats['cluster_2_multi_match_regions'])
    
    with col4:
        st.metric("Avg Matches/Group", f"{summary_stats['avg_matches_per_group']:.1f}")
    
    # Grouped pairs performance chart
    st.subheader("📈 Multi-Match Groups Performance")
    fig_groups, fig_centroids = st.session_state.comparator.create_recap_visualization(
        recap_results, cluster_1, cluster_2
    )
    
    if fig_groups:
        st.plotly_chart(fig_groups, use_container_width=True)
    
    # Centroid color visualization
    st.subheader("🎨 Centroid Colors for Multi-Match Groups")
    if fig_centroids:
        st.pyplot(fig_centroids, clear_figure=True)
    
    # Detailed groups table
    st.subheader("📋 Detailed Multi-Match Groups")
    
    # Format the grouped pairs data for display
    if len(grouped_pairs) > 0:
        display_df = grouped_pairs[[
            'group_name', 'matched_regions_count', 'matched_regions_list',
            'avg_delta_e_main', 'avg_delta_e_reflect', 'overall_avg_score',
            'total_samples_primary', 'total_samples_secondary'
        ]].copy()
        
        display_df.columns = [
            'Group Name', 'Match Count', 'Matched Regions',
            'Avg ΔE Main', 'Avg ΔE Reflect', 'Overall Score %',
            f'ST{cluster_1} Samples', f'ST{cluster_2} Samples'
        ]
        
        # Round numeric columns
        numeric_cols = ['Avg ΔE Main', 'Avg ΔE Reflect', 'Overall Score %']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        st.dataframe(display_df, use_container_width=True)
    
    # Hair rendering for best multi-match group
    st.subheader("💇‍♀️ Hair Rendering - Best Multi-Match Group")
    
    if len(centroid_data) > 0:
        best_group = grouped_pairs.loc[grouped_pairs['overall_avg_score'].idxmax()]
        best_centroid = centroid_data[centroid_data['group_name'] == best_group['group_name']].iloc[0]
        
        st.write(f"**Best Performing Group**: {best_group['group_name']}")
        st.write(f"**Overall Performance**: {best_group['overall_avg_score']:.1f}%")
        st.write(f"**Number of Matches**: {best_group['matched_regions_count']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎨 Render Overall Main Centroid", key="render_main_centroid"):
                render_centroid_hair(best_centroid, "main", best_group['group_name'])
        
        with col2:
            if st.button("🎨 Render Overall Reflect Centroid", key="render_reflect_centroid"):
                render_centroid_hair(best_centroid, "reflect", best_group['group_name'])
    
    # Export section
    st.subheader("📥 Export Recap Results")
    if st.button("📊 Export Recap to Excel"):
        try:
            excel_data = st.session_state.comparator.create_recap_excel_data(
                recap_results, cluster_1, cluster_2
            )
            
            # Create Excel file
            from io import BytesIO
            buffer = BytesIO()
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                if len(excel_data['grouped_pairs']) > 0:
                    excel_data['grouped_pairs'].to_excel(writer, sheet_name='Grouped_Pairs', index=False)
                if len(excel_data['centroid_data']) > 0:
                    excel_data['centroid_data'].to_excel(writer, sheet_name='Centroid_Data', index=False)
                excel_data['summary'].to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                "📥 Download Recap Excel File",
                buffer.getvalue(),
                file_name=f"similar_pairs_recap_ST{cluster_1}_vs_ST{cluster_2}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success("✅ Recap Excel export ready!")
        except Exception as e:
            st.error(f"Export error: {e}")

def render_centroid_hair(centroid_data, color_type, group_name):
    """Render hair for centroid colors"""
    try:
        with st.spinner(f"Rendering {color_type} centroid hair..."):
            # Prepare LAB data
            lab_data = {
                'L_main': centroid_data[f'overall_{color_type}_L'],
                'a_main': centroid_data[f'overall_{color_type}_a'],
                'b_main': centroid_data[f'overall_{color_type}_b'],
                'L_reflect': centroid_data[f'overall_{color_type}_L'],  # Use same for both for centroid
                'a_reflect': centroid_data[f'overall_{color_type}_a'],
                'b_reflect': centroid_data[f'overall_{color_type}_b']
            }
            
            region_name = f"{group_name}_{color_type}_centroid"
            render_single_region_from_lab(lab_data, region_name)
            
    except Exception as e:
        st.error(f"❌ Error rendering centroid: {str(e)}")

def render_best_matching_pair(similar_pairs, cluster_results, cluster_1, cluster_2):
    """Render hair for the best matching color pair"""
    if len(similar_pairs) == 0:
        return
    
    best_pair = similar_pairs.iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Cluster {cluster_1} - Region {best_pair[f'region_{cluster_1}_id']}**")
        
        # Get LAB values for cluster 1
        lab_data_1 = {
            'L_main': best_pair[f'L_main_{cluster_1}'],
            'a_main': best_pair[f'a_main_{cluster_1}'],
            'b_main': best_pair[f'b_main_{cluster_1}'],
            'L_reflect': best_pair[f'L_reflect_{cluster_1}'],
            'a_reflect': best_pair[f'a_reflect_{cluster_1}'],
            'b_reflect': best_pair[f'b_reflect_{cluster_1}']
        }
        
        if st.button(f"🎨 Render Cluster {cluster_1}", key=f"render_c1"):
            render_single_region_from_lab(lab_data_1, f"cluster_{cluster_1}_region_{best_pair[f'region_{cluster_1}_id']}")
    
    with col2:
        st.write(f"**Cluster {cluster_2} - Region {best_pair[f'region_{cluster_2}_id']}**")
        
        # Get LAB values for cluster 2
        lab_data_2 = {
            'L_main': best_pair[f'L_main_{cluster_2}'],
            'a_main': best_pair[f'a_main_{cluster_2}'],
            'b_main': best_pair[f'b_main_{cluster_2}'],
            'L_reflect': best_pair[f'L_reflect_{cluster_2}'],
            'a_reflect': best_pair[f'a_reflect_{cluster_2}'],
            'b_reflect': best_pair[f'b_reflect_{cluster_2}']
        }
        
        if st.button(f"🎨 Render Cluster {cluster_2}", key=f"render_c2"):
            render_single_region_from_lab(lab_data_2, f"cluster_{cluster_2}_region_{best_pair[f'region_{cluster_2}_id']}")

def render_single_region_from_lab(lab_data, region_name):
    """Render a single hair swatch from LAB data"""
    try:
        # Prepare data for rendering
        render_data = pd.DataFrame({
            'model': [region_name],
            'cluster': [region_name],
            'L_1': [lab_data['L_main']],
            'a_1': [lab_data['a_main']],
            'b_1': [lab_data['b_main']],
            'L_2': [lab_data['L_reflect']],
            'a_2': [lab_data['a_reflect']],
            'b_2': [lab_data['b_reflect']]
        })
        
        # Save temp data
        os.makedirs("assets", exist_ok=True)
        temp_path = "assets/temp.xlsx"
        render_data.to_excel(temp_path, index=False)
        
        # Render hair
        rendered_path = st.session_state.renderer.render_hair_swatch(
            temp_path, "Medium", region_name
        )
        
        if rendered_path and os.path.exists(rendered_path):
            st.success("✅ Hair rendering completed!")
            st.image(rendered_path, caption=f"{region_name} Rendering")
            
            # Download button
            with open(rendered_path, "rb") as file:
                st.download_button(
                    "📥 Download Rendered Image",
                    file.read(),
                    file_name=f"{region_name}_hair_rendering.jpg",
                    mime="image/jpeg",
                    key=f"download_{region_name}"
                )
        else:
            st.error("❌ Failed to generate hair rendering")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def create_performance_overview(analysis_results, threshold, metric_column, metric_description):
    """Create interactive performance overview chart"""
    df = analysis_results['top_regions']
    
    # Create color scale based on threshold
    colors = ['#D32F2F' if x < threshold else '#4CAF50' for x in df[metric_column]]
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=df['color_regions'].astype(str),
        y=df[metric_column],
        marker_color=colors,
        text=[f"{x:.1f}%" for x in df[metric_column]],
        textposition='outside',
        hovertemplate="<b>Region %{x}</b><br>" +
                     f"{metric_description}: %{{y:.1f}}%<br>" +
                     "Samples: %{customdata}<br>" +
                     "<extra></extra>",
        customdata=df['total_samples'],
        name="Performance"
    ))
    
    # Add threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="#FF9800", 
                  annotation_text=f"Threshold: {threshold}%")
    
    fig.update_layout(
        title=f"{metric_description} Performance by Color Region",
        xaxis_title="Color Region",
        yaxis_title=f"{metric_description} Score (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_region_selector(analysis_results, metric_column):
    """Create interactive region selector with color swatches"""
    df = analysis_results['top_regions']
    cluster_centers = analysis_results['cluster_centers']
    
    selected_region = None
    
    # Create a grid of region cards
    cols = st.columns(5)
    
    for i, (_, row) in enumerate(df.iterrows()):
        col_idx = i % 5
        region_id = int(row['color_regions'])
        
        with cols[col_idx]:
            # Create color swatch
            if region_id in cluster_centers.index:
                center = cluster_centers.loc[region_id]
                rgb_main = st.session_state.helpers.lab_to_rgb(
                    center['L_main'], center['a_main'], center['b_main']
                )
                rgb_reflect = st.session_state.helpers.lab_to_rgb(
                    center['L_reflect'], center['a_reflect'], center['b_reflect']
                )
                
                # Create visual representation
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 1.5))
                ax1.add_patch(Rectangle((0, 0), 1, 1, color=rgb_main))
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                ax1.set_title("Main", fontsize=8)
                ax1.axis('off')
                
                ax2.add_patch(Rectangle((0, 0), 1, 1, color=rgb_reflect))
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.set_title("Reflect", fontsize=8)
                ax2.axis('off')
                
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
            else:
                st.warning(f"Color data missing for region {region_id}")
            
            # Region info
            st.write(f"**Region {region_id}**")
            st.write(f"Score: {row[metric_column]:.1f}%")
            st.write(f"Samples: {int(row['total_samples'])}")
            
            # Selection button - use session state
            button_key = f"btn_{region_id}"
            is_selected = st.session_state.get('selected_region') == region_id
            
            if st.button(
                f"{'✅ Selected' if is_selected else 'Analyze'} Region {region_id}", 
                key=button_key,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_region = region_id
                st.rerun()  # Force rerun to update UI
    
    return st.session_state.get('selected_region')

def show_universal_regions_analysis(cluster_results, threshold, min_clusters_required, universal_threshold, swatch_type, metric_column, metric_description):
    """Universal regions analysis - finds similar regions across multiple clusters"""
    st.header("🌍 Universal Regions Analysis")
    st.info(f"This analysis finds color regions that perform well across multiple skin tone clusters using DeltaE similarity.\n\n**Performance Metric**: {metric_description}")
    
    if len(cluster_results) < 2:
        st.error("Universal analysis requires at least 2 clusters!")
        return
    
    # Analysis parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delta_e_threshold = st.slider(
            "DeltaE Threshold",
            min_value=1.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            help="Lower values = more similar colors required",
            key="universal_delta_e"
        )
    
    with col2:
        top_n_universal = st.slider(
            "Top N regions per cluster",
            min_value=5,
            max_value=20,
            value=10,
            key="universal_top_n"
        )
    
    with col3:
        min_performance_threshold = st.slider(
            "Min Performance Threshold (%)",
            min_value=30,
            max_value=80,
            value=universal_threshold,
            help="Minimum performance required in each cluster",
            key="universal_perf_threshold"
        )
    
    # Perform universal analysis
    if st.button("🔍 Find Universal Regions", type="primary"):
        with st.spinner("Analyzing universal regions across clusters..."):
            
            try:
                # Perform universal analysis
                universal_results = st.session_state.comparator.find_universal_regions(
                    cluster_results, 
                    delta_e_threshold, 
                    min_clusters_required,
                    min_performance_threshold,
                    top_n_universal,
                    metric_column
                )
                
                # Store results in session state
                st.session_state.universal_results = universal_results
                st.session_state.universal_params = {
                    'delta_e_threshold': delta_e_threshold,
                    'min_clusters_required': min_clusters_required,
                    'min_performance_threshold': min_performance_threshold,
                    'swatch_type': swatch_type,
                    'metric_column': metric_column,
                    'metric_description': metric_description
                }
                
                st.success("✅ Universal regions analysis completed!")
                
            except Exception as e:
                st.error(f"Error during universal analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Display results if available
    if 'universal_results' in st.session_state:
        show_universal_results()

def show_universal_results():
    """Display universal regions analysis results"""
    universal_results = st.session_state.universal_results
    params = st.session_state.universal_params
    
    universal_regions = universal_results['universal_regions']
    cluster_coverage = universal_results['cluster_coverage']
    centroid_colors = universal_results['centroid_colors']
    performance_summary = universal_results['performance_summary']
    
    # Results overview
    st.markdown("---")
    st.subheader("📊 Universal Regions Results")
    st.info(f"**Performance Metric Used**: {params['metric_description']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Universal Groups Found", len(universal_regions))
    
    with col2:
        total_clusters = len(universal_results['analyzed_clusters'])
        st.metric("Clusters Analyzed", total_clusters)
    
    with col3:
        if len(universal_regions) > 0:
            max_coverage = max([len(group['cluster_regions']) for group in universal_regions])
            st.metric("Max Cluster Coverage", f"{max_coverage}/{total_clusters}")
        else:
            st.metric("Max Cluster Coverage", "0")
    
    with col4:
        if len(universal_regions) > 0:
            best_group = max(universal_regions, key=lambda x: x['avg_performance'])
            st.metric("Best Avg Performance", f"{best_group['avg_performance']:.1f}%")
        else:
            st.metric("Best Avg Performance", "N/A")
    
    if len(universal_regions) == 0:
        st.warning(f"No universal color groups found with the current parameters:")
        st.write(f"- DeltaE threshold: {params['delta_e_threshold']}")
        st.write(f"- Minimum clusters required: {params['min_clusters_required']}")
        st.write(f"- Performance threshold: {params['min_performance_threshold']}%")
        st.write(f"- Metric used: {params['metric_description']}")
        st.info("Try increasing the DeltaE threshold or reducing the minimum clusters required.")
        return
    
    # Debug information (optional)
    if st.checkbox("Show Debug Information", key="debug_universal"):
        st.subheader("🔍 Debug Information")
        
        with st.expander("Universal Groups Details"):
            for i, group in enumerate(universal_regions):
                st.write(f"**Group {i+1}: {group['group_name']}**")
                st.write(f"- Clusters: {list(group['cluster_regions'].keys())}")
                st.write(f"- Performance: {group['avg_performance']:.1f}%")
                st.write(f"- Total samples: {group['total_samples']}")
                st.write(f"- Avg DeltaE: {group['avg_delta_e']:.2f}")
                st.write("---")
        
        with st.expander("Centroid Colors Details"):
            if centroid_colors:
                st.dataframe(pd.DataFrame(centroid_colors))
            else:
                st.write("No centroid colors data")
        
        with st.expander("Analysis Parameters"):
            st.json(params)
    
    # Universal regions visualization
    st.subheader("📈 Universal Region Groups Performance")
    try:
        fig_performance = st.session_state.comparator.create_universal_performance_chart(
            universal_regions, universal_results['analyzed_clusters']
        )
        if fig_performance:
            st.plotly_chart(fig_performance, use_container_width=True)
        else:
            st.warning("Could not create performance chart")
    except Exception as e:
        st.error(f"Error creating performance chart: {e}")
        if st.checkbox("Show performance chart error details", key="debug_perf_chart"):
            import traceback
            st.code(traceback.format_exc())
    
    # Universal color swatches
    st.subheader("🎨 Universal Color Groups")
    try:
        fig_colors = st.session_state.comparator.create_universal_color_swatches(
            centroid_colors, universal_regions
        )
        if fig_colors:
            st.pyplot(fig_colors, clear_figure=True)
        else:
            st.warning("Could not create color swatches")
    except Exception as e:
        st.error(f"Error creating color swatches: {e}")
        if st.checkbox("Show color swatches error details", key="debug_colors"):
            import traceback
            st.code(traceback.format_exc())
    
    # Detailed universal regions table
    st.subheader("📋 Universal Regions Details")
    
    try:
        # Format universal regions for display
        display_data = []
        for i, group in enumerate(universal_regions):
            # Create cluster info string
            cluster_info = []
            region_info = []
            performance_info = []
            
            for cluster_id in sorted(group['cluster_regions'].keys()):
                region_data = group['cluster_regions'][cluster_id]
                cluster_info.append(f"ST{cluster_id}")
                region_info.append(f"R{region_data['region_id']}")
                performance_info.append(f"{region_data['performance']:.1f}%")
            
            display_data.append({
                'Group ID': f"UG{i+1}",
                'Group Name': group['group_name'],
                'Cluster Count': len(group['cluster_regions']),
                'Clusters': ", ".join(cluster_info),
                'Regions': ", ".join(region_info),
                'Performances': ", ".join(performance_info),
                'Avg Performance': f"{group['avg_performance']:.1f}%",
                'Min Performance': f"{group['min_performance']:.1f}%",
                'Max Performance': f"{group['max_performance']:.1f}%",
                'Total Samples': group['total_samples'],
                'Avg DeltaE': f"{group['avg_delta_e']:.2f}",
                'Max DeltaE': f"{group['max_delta_e']:.2f}"
            })
        
        if display_data:
            display_df = pd.DataFrame(display_data)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No display data available")
            
    except Exception as e:
        st.error(f"Error creating details table: {e}")
        if st.checkbox("Show table error details", key="debug_table"):
            import traceback
            st.code(traceback.format_exc())
    
    # Hair rendering section
    st.subheader("💇‍♀️ Hair Rendering - Universal Groups")

    try:
        if len(universal_regions) > 0 and len(centroid_colors) > 0:
            
            # Create selection options
            group_options = []
            group_lookup = {}
            
            for i, group in enumerate(universal_regions):
                # Create descriptive option text
                option_text = f"UG{i+1}: {len(group['cluster_regions'])} clusters, {group['avg_performance']:.1f}% avg performance"
                group_options.append(option_text)
                group_lookup[option_text] = group
            
            # Group selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_option = st.selectbox(
                    "🎯 Select Universal Group to Render",
                    group_options,
                    help="Choose which universal group you want to render as hair color"
                )
            
            with col2:
                st.metric("Selected Group Clusters", len(group_lookup[selected_option]['cluster_regions']))
            
            # Get selected group and its centroid
            selected_group = group_lookup[selected_option]
            selected_centroid = None
            
            # Find corresponding centroid
            for centroid in centroid_colors:
                if centroid['group_name'] == selected_group['group_name']:
                    selected_centroid = centroid
                    break
            
            if selected_centroid:
                # Display selected group info
                st.write("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Group**: {selected_group['group_name']}")
                    st.write(f"**Clusters**: {len(selected_group['cluster_regions'])}")
                    cluster_list = ", ".join([f"ST{c}" for c in sorted(selected_group['cluster_regions'].keys())])
                    st.write(f"**Coverage**: {cluster_list}")
                
                with col2:
                    st.write(f"**Avg Performance**: {selected_group['avg_performance']:.1f}%")
                    st.write(f"**Performance Range**: {selected_group['min_performance']:.1f}% - {selected_group['max_performance']:.1f}%")
                    st.write(f"**Total Samples**: {selected_group['total_samples']}")
                
                with col3:
                    st.write(f"**Avg DeltaE**: {selected_group['avg_delta_e']:.2f}")
                    st.write(f"**Max DeltaE**: {selected_group['max_delta_e']:.2f}")
                    
                    # Show regions in each cluster
                    region_details = []
                    for cluster_id in sorted(selected_group['cluster_regions'].keys()):
                        region_info = selected_group['cluster_regions'][cluster_id]
                        region_details.append(f"ST{cluster_id}-R{region_info['region_id']}")
                    st.write(f"**Regions**: {', '.join(region_details)}")
                
                # Color preview
                st.write("---")
                st.subheader("🎨 Color Preview")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Show LAB values for main color
                    st.write("**Main Color LAB:**")
                    st.write(f"L*: {selected_centroid['centroid_main_L']:.2f}")
                    st.write(f"a*: {selected_centroid['centroid_main_a']:.2f}")
                    st.write(f"b*: {selected_centroid['centroid_main_b']:.2f}")
                
                with col2:
                    # Show LAB values for reflect color
                    st.write("**Reflect Color LAB:**")
                    st.write(f"L*: {selected_centroid['centroid_reflect_L']:.2f}")
                    st.write(f"a*: {selected_centroid['centroid_reflect_a']:.2f}")
                    st.write(f"b*: {selected_centroid['centroid_reflect_b']:.2f}")
                
                with col3:
                    # Rendering options
                    st.write("**Hair Rendering Options:**")
                    
                    swatch_template = st.selectbox(
                        "Hair Template",
                        ["Dark", "Medium", "Light"],
                        index=1,
                        key="universal_swatch_template"
                    )
                    
                    # Main rendering button
                    if st.button("🎨 Render Complete Hair Color", key="render_universal_complete", type="primary"):
                        render_universal_hair_complete(selected_centroid, selected_group['group_name'], swatch_template)
                
                # Additional rendering options
                st.write("---")
                st.subheader("🔬 Advanced Rendering Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🎨 Render Main Color Only", key="render_universal_main_only"):
                        render_universal_centroid(selected_centroid, "main", selected_group['group_name'], swatch_template)
                
                with col2:
                    if st.button("🎨 Render Reflect Color Only", key="render_universal_reflect_only"):
                        render_universal_centroid(selected_centroid, "reflect", selected_group['group_name'], swatch_template)
                
                with col3:
                    st.info("Use 'Main Only' or 'Reflect Only' to see individual color components separately.")
            
            else:
                st.warning("Could not find centroid data for selected group")
        
        else:
            st.warning("No universal groups or centroid data available for rendering")
            
    except Exception as e:
        st.error(f"Error in hair rendering section: {e}")
        if st.checkbox("Show rendering error details", key="debug_rendering"):
            import traceback
            st.code(traceback.format_exc())
    
    # Export section
    st.subheader("📥 Export Universal Results")
    try:
        if st.button("📊 Export Universal Analysis to Excel"):
            try:
                excel_data = st.session_state.comparator.create_universal_excel_data(
                    universal_results, params
                )
                
                # Create Excel file
                from io import BytesIO
                buffer = BytesIO()
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    if len(excel_data['universal_regions']) > 0:
                        excel_data['universal_regions'].to_excel(writer, sheet_name='Universal_Regions', index=False)
                    if len(excel_data['centroid_colors']) > 0:
                        excel_data['centroid_colors'].to_excel(writer, sheet_name='Centroid_Colors', index=False)
                    if len(excel_data['cluster_coverage']) > 0:
                        excel_data['cluster_coverage'].to_excel(writer, sheet_name='Cluster_Coverage', index=False)
                    excel_data['summary'].to_excel(writer, sheet_name='Summary', index=False)
                
                clusters_str = "_".join([str(c) for c in universal_results['analyzed_clusters']])
                st.download_button(
                    "📥 Download Universal Analysis Excel",
                    buffer.getvalue(),
                    file_name=f"universal_regions_analysis_ST{clusters_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Universal analysis Excel export ready!")
            except Exception as e:
                st.error(f"Export error: {e}")
                if st.checkbox("Show export error details", key="debug_export"):
                    import traceback
                    st.code(traceback.format_exc())
                    
    except Exception as e:
        st.error(f"Error in export section: {e}")

def create_short_universal_name(group_name, suffix=""):
    """Create a short name for universal groups to avoid filesystem limits"""
    import re
    
    # Extract cluster numbers from the group name
    clusters = re.findall(r'ST(\d+)', group_name)
    
    if clusters:
        # Create short name like "UG_1_2_3_4_5_6" + suffix
        short_name = f"UG_{'_'.join(clusters)}{suffix}"
        
        # If still too long, use hash
        if len(short_name) > 50:
            short_name = f"UG_{hash(group_name) % 10000}{suffix}"
    else:
        # Fallback to hash
        short_name = f"UG_{hash(group_name) % 10000}{suffix}"
    
    return short_name

def render_universal_hair_complete(centroid_data, group_name, swatch_type="Medium"):
    """Render complete hair swatch using both main and reflect colors from universal centroid"""
    try:
        with st.spinner("Rendering complete universal hair swatch..."):
            # Prepare LAB data using BOTH main and reflect colors properly
            lab_data = {
                'L_main': centroid_data['centroid_main_L'],
                'a_main': centroid_data['centroid_main_a'], 
                'b_main': centroid_data['centroid_main_b'],
                'L_reflect': centroid_data['centroid_reflect_L'],
                'a_reflect': centroid_data['centroid_reflect_a'],
                'b_reflect': centroid_data['centroid_reflect_b']
            }
            
            # Create shorter name for file system
            short_name = create_short_universal_name(group_name, f"_complete_{swatch_type}")
            render_single_region_from_lab(lab_data, short_name)
            
    except Exception as e:
        st.error(f"❌ Error rendering universal hair: {str(e)}")

def render_universal_centroid(centroid_data, color_type, group_name, swatch_type="Medium"):
    """Render hair for universal centroid colors (individual components)"""
    try:
        with st.spinner(f"Rendering {color_type} universal centroid hair..."):
            # Prepare LAB data - use the same color for both main and reflect when rendering individual components
            lab_data = {
                'L_main': centroid_data[f'centroid_{color_type}_L'],
                'a_main': centroid_data[f'centroid_{color_type}_a'],
                'b_main': centroid_data[f'centroid_{color_type}_b'],
                'L_reflect': centroid_data[f'centroid_{color_type}_L'],  # Use same for both when showing individual component
                'a_reflect': centroid_data[f'centroid_{color_type}_a'],
                'b_reflect': centroid_data[f'centroid_{color_type}_b']
            }
            
            # Create shorter name for file system
            short_name = create_short_universal_name(group_name, f"_{color_type}_{swatch_type}")
            render_single_region_from_lab(lab_data, short_name)
            
    except Exception as e:
        st.error(f"❌ Error rendering universal {color_type} centroid: {str(e)}")

def create_export_section(cluster_results, selected_clusters, analysis_mode, metric_type):
    """Export results section"""
    st.header("📥 Export Results")
    
    if len(cluster_results) > 0:
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📊 Export Analysis to Excel"):
                try:
                    # Create comprehensive export data
                    all_data = []
                    for cluster_id, results in cluster_results.items():
                        cluster_data = results['top_regions'].copy()
                        cluster_data['cluster_id'] = cluster_id
                        cluster_data['metric_type'] = metric_type
                        all_data.append(cluster_data)
                    
                    if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        
                        # Create Excel file
                        excel_data = st.session_state.helpers.create_export_excel(
                            combined_df, f"Analysis_{analysis_mode.replace(' ', '_')}_{metric_type.replace(' ', '_')}"
                        )
                        
                        st.download_button(
                            "📥 Download Excel File",
                            excel_data,
                            file_name=f"hair_color_analysis_{analysis_mode.replace(' ', '_').lower()}_{metric_type.replace(' ', '_').lower()}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("✅ Excel export ready!")
                except Exception as e:
                    st.error(f"Export error: {e}")
        
        with export_col2:
            st.info(f"Analysis includes {len(selected_clusters)} cluster(s) using {metric_type}")
    else:
        st.warning("No data available for export")

if __name__ == "__main__":
    main()