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
            "Combined Criteria Threshold (%)",
            min_value=0,
            max_value=100,
            value=45,
            help="Minimum percentage for S1R≥3 AND S2R≥2"
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
                    analysis_results = st.session_state.analyzer.analyze_color_regions(
                        file_path, threshold, top_n_regions
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
        show_single_cluster_analysis(cluster_results, selected_clusters[0], threshold, swatch_type, auto_render)
        
    elif analysis_mode == "Compare Clusters (DeltaE)":
        show_cluster_comparison_analysis(available_clusters, threshold)
        
    else:  # Universal Regions
        show_universal_regions_analysis(
            cluster_results, threshold, min_clusters_required, 
            universal_threshold, swatch_type
        )

    # Export section (only for non-comparison modes)
    if analysis_mode != "Compare Clusters (DeltaE)":
        st.markdown("---")
        create_export_section(cluster_results, selected_clusters, analysis_mode)

def show_single_cluster_analysis(cluster_results, cluster_id, threshold, swatch_type, auto_render):
    """Show single cluster analysis (original functionality)"""
    analysis_results = cluster_results[cluster_id]
    
    # Initialize selected region state
    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = None
    
    # Main dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📈 Top Regions - Cluster {cluster_id}")
        
        # Performance overview chart
        fig = create_performance_overview(analysis_results, threshold)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive region selection
        st.subheader("🎯 Select a Region for Detailed Analysis")
        selected_region = create_region_selector(analysis_results)
        
    with col2:
        st.header("📊 Cluster Overview")
        
        # Summary metrics
        total_regions = len(analysis_results['top_regions'])
        meeting_criteria = len(analysis_results['top_regions'][
            analysis_results['top_regions']['combined_criteria_pct'] >= threshold
        ])
        
        st.metric("Total Regions Analyzed", total_regions)
        st.metric("Meeting Criteria", meeting_criteria)
        st.metric("Best Performance", 
                 f"{analysis_results['top_regions'].iloc[0]['combined_criteria_pct']:.1f}%")

    # Detailed analysis section
    if selected_region is not None:
        st.markdown("---")
        show_detailed_analysis(selected_region, analysis_results, cluster_id, swatch_type, auto_render)

def show_detailed_analysis(selected_region, analysis_results, cluster_id, swatch_type, auto_render):
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
            
            # Performance metrics
            st.metric("Combined Criteria Score", f"{region_info['combined_criteria_pct']:.1f}%")
            st.metric("Total Samples", int(region_info['total_samples']))
            st.metric("S1R Mean Rating", f"{region_info['S1R_mean']:.2f}")
            st.metric("S1R TOP2BOX %", f"{region_info['S1R_top2box_pct']:.1f}%")
            
            if not pd.isna(region_info['S2R_mean']):
                st.metric("S2R Mean Rating", f"{region_info['S2R_mean']:.2f}")
                st.metric("S2R TOP2BOX %", f"{region_info['S2R_top2box_pct']:.1f}%")
                st.metric("S2R Data Coverage", f"{region_info['S2R_data_pct']:.1f}%")
            
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

def show_cluster_comparison_analysis(available_clusters, threshold):
    """Show comparison analysis between two clusters using DeltaE"""
    st.header("🔍 Cluster Comparison Analysis (DeltaE Method)")
    
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
            "DeltaE Threshold",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Lower values = more similar colors required"
        )
    
    if cluster_1 == cluster_2:
        st.warning("Please select two different clusters for comparison!")
        return
    
    # Comparison parameters
    st.subheader("🔧 Comparison Parameters")
    col1, col2 = st.columns(2)
    
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
    
    # Perform comparison
    if st.button("🔍 Perform DeltaE Comparison", type="primary"):
        with st.spinner(f"Comparing clusters {cluster_1} and {cluster_2}..."):
            
            try:
                # Load data for both clusters
                cluster_results = {}
                for cluster_id in [cluster_1, cluster_2]:
                    file_path = f"results/df_100_eval_{cluster_id}_min_100_minwomen_50.0_s1r_0.0.csv"
                    
                    analysis_results = st.session_state.analyzer.analyze_color_regions(
                        file_path, threshold, top_n_compare
                    )
                    
                    if analysis_results is None:
                        st.error(f"Failed to analyze data for cluster {cluster_id}")
                        return
                        
                    cluster_results[cluster_id] = analysis_results
                
                if len(cluster_results) != 2:
                    st.error("Failed to load data for both clusters!")
                    return
                
                # Perform comparison
                comparison_results = st.session_state.comparator.compare_two_clusters(
                    {cluster_1: cluster_results[cluster_1]},
                    {cluster_2: cluster_results[cluster_2]}, 
                    cluster_1, 
                    cluster_2, 
                    delta_e_threshold, 
                    top_n_compare
                )
                
                # Store results in session state
                st.session_state.comparison_results = comparison_results
                st.session_state.cluster_1 = cluster_1
                st.session_state.cluster_2 = cluster_2
                st.session_state.cluster_results_comparison = cluster_results
                st.session_state.max_pairs_display = max_pairs_display
                
                st.success("✅ Comparison analysis completed!")
                
            except Exception as e:
                st.error(f"Error during comparison: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Display results if available
    if 'comparison_results' in st.session_state:
        show_comparison_results()

def show_comparison_results():
    """Display comparison results"""
    comparison_results = st.session_state.comparison_results
    similar_pairs = comparison_results['similar_pairs']
    comparison_df = comparison_results['comparison_df']
    cluster_1 = st.session_state.cluster_1
    cluster_2 = st.session_state.cluster_2
    max_pairs_display = st.session_state.get('max_pairs_display', 8)
    
    # Results overview
    st.markdown("---")
    st.subheader("📊 Comparison Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Comparisons", comparison_results['total_comparisons'])
    
    with col2:
        st.metric("Similar Pairs Found", comparison_results['similar_count'])
    
    with col3:
        st.metric("Similarity Rate", f"{comparison_results['similarity_rate']:.1f}%")
    
    with col4:
        if len(similar_pairs) > 0:
            best_match = similar_pairs.iloc[0]
            st.metric("Best Match ΔE", f"{best_match['delta_e_main']:.2f}")
        else:
            st.metric("Best Match ΔE", "N/A")
    
    # DeltaE scatter plot
    st.subheader("📈 DeltaE Analysis")
    fig_scatter = st.session_state.comparator.create_delta_e_scatter_plotly(
        comparison_df, similar_pairs, cluster_1, cluster_2
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Similar pairs visualization
    if len(similar_pairs) > 0:
        st.subheader("🎨 Similar Color Pairs")
        
        # Color comparison plot
        fig_colors = st.session_state.comparator.create_color_comparison_plot(
            similar_pairs, cluster_1, cluster_2, max_pairs_display
        )
        
        if fig_colors:
            st.pyplot(fig_colors, clear_figure=True)
        
        # Similar pairs table
        st.subheader("📋 Similar Pairs Details")
        
        # Format similar pairs for display
        display_pairs = similar_pairs.copy()
        display_cols = [
            f'region_{cluster_1}_id', f'region_{cluster_2}_id', 
            'delta_e_main', 'delta_e_reflect',
            f'region_{cluster_1}_score', f'region_{cluster_2}_score',
            f'region_{cluster_1}_samples', f'region_{cluster_2}_samples'
        ]
        
        display_df = display_pairs[display_cols].round(2)
        display_df.columns = [
            f'ST{cluster_1} Region', f'ST{cluster_2} Region',
            'ΔE Main', 'ΔE Reflect', 
            f'ST{cluster_1} Score %', f'ST{cluster_2} Score %',
            f'ST{cluster_1} Samples', f'ST{cluster_2} Samples'
        ]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Hair rendering for best match
        st.subheader("💇‍♀️ Hair Rendering - Best Match")
        
        if st.button("🎨 Render Best Matching Pair"):
            render_best_matching_pair(similar_pairs, st.session_state.cluster_results_comparison, cluster_1, cluster_2)
    
    else:
        st.warning(f"No similar color pairs found with ΔE threshold < {comparison_results['delta_e_threshold']}")
        st.info("Try increasing the DeltaE threshold to find more matches.")

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

def create_performance_overview(analysis_results, threshold):
    """Create interactive performance overview chart"""
    df = analysis_results['top_regions']
    
    # Create color scale based on threshold
    colors = ['#D32F2F' if x < threshold else '#4CAF50' for x in df['combined_criteria_pct']]
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=df['color_regions'].astype(str),
        y=df['combined_criteria_pct'],
        marker_color=colors,
        text=[f"{x:.1f}%" for x in df['combined_criteria_pct']],
        textposition='outside',
        hovertemplate="<b>Region %{x}</b><br>" +
                     "Performance: %{y:.1f}%<br>" +
                     "Samples: %{customdata}<br>" +
                     "<extra></extra>",
        customdata=df['total_samples'],
        name="Performance"
    ))
    
    # Add threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="#FF9800", 
                  annotation_text=f"Threshold: {threshold}%")
    
    fig.update_layout(
        title="Combined Criteria Performance by Color Region",
        xaxis_title="Color Region",
        yaxis_title="Combined Criteria Score (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_region_selector(analysis_results):
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
            st.write(f"Score: {row['combined_criteria_pct']:.1f}%")
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

def show_universal_regions_analysis(cluster_results, threshold, min_clusters_required, universal_threshold, swatch_type):
    """Universal regions analysis"""
    st.header("🌍 Universal Regions Analysis")
    st.info("This feature finds color regions that perform well across multiple skin tone clusters.")
    
    # Implementation placeholder - can be expanded based on requirements
    st.write("Coming soon: Analysis of regions that work well across multiple clusters!")

def create_export_section(cluster_results, selected_clusters, analysis_mode):
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
                        all_data.append(cluster_data)
                    
                    if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        
                        # Create Excel file
                        excel_data = st.session_state.helpers.create_export_excel(
                            combined_df, f"Analysis_{analysis_mode.replace(' ', '_')}"
                        )
                        
                        st.download_button(
                            "📥 Download Excel File",
                            excel_data,
                            file_name=f"hair_color_analysis_{analysis_mode.replace(' ', '_').lower()}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("✅ Excel export ready!")
                except Exception as e:
                    st.error(f"Export error: {e}")
        
        with export_col2:
            st.info(f"Analysis includes {len(selected_clusters)} cluster(s)")
    else:
        st.warning("No data available for export")

if __name__ == "__main__":
    main()