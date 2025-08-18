import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from front.ui_components import create_performance_overview, create_region_selector
from front.hair_rendering import render_hair_for_region

def show_single_cluster_analysis(cluster_results, config):
    """Show single cluster analysis (original functionality)"""
    cluster_id = config['selected_clusters'][0]
    analysis_results = cluster_results[cluster_id]
    
    # Initialize selected region state
    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = None
    
    # Main dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📈 Top Regions - Cluster {cluster_id}")
        st.info(f"**Performance Metric**: {config['metric_description']}")
        
        # Performance overview chart
        fig = create_performance_overview(
            analysis_results, 
            config['threshold'], 
            config['metric_column'], 
            config['metric_description']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive region selection
        st.subheader("🎯 Select a Region for Detailed Analysis")
        selected_region = create_region_selector(analysis_results, config['metric_column'])
        
    with col2:
        st.header("📊 Cluster Overview")
        
        # Summary metrics
        total_regions = len(analysis_results['top_regions'])
        meeting_criteria = len(analysis_results['top_regions'][
            analysis_results['top_regions'][config['metric_column']] >= config['threshold']
        ])
        
        st.metric("Total Regions Analyzed", total_regions)
        st.metric("Meeting Criteria", meeting_criteria)
        st.metric("Best Performance", 
                 f"{analysis_results['top_regions'].iloc[0][config['metric_column']]:.1f}%")

    # Detailed analysis section
    if selected_region is not None:
        st.markdown("---")
        show_detailed_analysis(selected_region, analysis_results, cluster_id, config)

def show_detailed_analysis(selected_region, analysis_results, cluster_id, config):
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
            show_performance_metrics(region_info, config['metric_column'], exposure_data, selected_region)
        
        with col2:
            show_color_information(center_data)
        
        with col3:
            show_family_composition(family_data, selected_region)
        
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
                render_hair_for_region(selected_region, center_data, config['rendering_options']['swatch_type'])
        
        with render_button_col2:
            if config['rendering_options']['auto_render']:
                st.info("Auto-rendering enabled for top regions")
        
        # Sample data from this region
        show_sample_data(original_data, selected_region)
        
    except Exception as e:
        st.error(f"Error in detailed analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def show_performance_metrics(region_info, metric_column, exposure_data, selected_region):
    """Show performance metrics section"""
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

def show_color_information(center_data):
    """Show color information section"""
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

def show_family_composition(family_data, selected_region):
    """Show family composition section"""
    st.subheader("👥 Family Composition")
    
    if selected_region in family_data and family_data[selected_region]:
        family_composition = family_data[selected_region]
        
        for family_info in family_composition[:5]:  # Top 5 families
            st.write(f"**{family_info['family']}**: {family_info['count']} samples ({family_info['percentage']:.1f}%)")
    else:
        st.info("No family composition data available")

def show_sample_data(original_data, selected_region):
    """Show sample data section"""
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