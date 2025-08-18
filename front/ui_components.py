import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Rectangle
import os
import re

def create_main_header():
    """Create main header and custom CSS styling"""
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
    
    st.markdown('<h1 class="main-header">🎨 L\'Oréal Multi-Cluster Hair Color Analysis & Comparison</h1>', unsafe_allow_html=True)

def create_sidebar():
    """Create sidebar configuration and return settings"""
    with st.sidebar:
        st.header("📊 Analysis Configuration")
        
        # Find available clusters
        available_clusters = find_available_clusters()
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
        
        # Metric selection
        st.subheader("📈 Performance Metric")
        metric_type = st.radio(
            "Select Performance Metric:",
            ["Combined Criteria", "S1R Top2Box Only", "S2R Top2Box Only"],
            help="Combined: Uses both S1R≥3 AND S2R≥2 criteria | S1R: Only S1R top2box percentage | S2R: Only S2R top2box percentage"
        )
        
        # Get metric details
        metric_column, metric_description = get_metric_details(metric_type)
        st.info(f"**Selected Metric**: {metric_description}")
        
        # Cluster selection based on mode
        selected_clusters = get_cluster_selection(analysis_mode, available_clusters)
        
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
        
        # Universal region parameters
        universal_params = {}
        if analysis_mode == "Universal Regions":
            universal_params = get_universal_parameters(selected_clusters)
        
        # Hair rendering options
        rendering_options = get_rendering_options()
        
        # Return configuration dictionary
        return {
            'analysis_mode': analysis_mode,
            'metric_type': metric_type,
            'metric_column': metric_column,
            'metric_description': metric_description,
            'selected_clusters': selected_clusters,
            'available_clusters': available_clusters,
            'threshold': threshold,
            'top_n_regions': top_n_regions,
            'universal_params': universal_params,
            'rendering_options': rendering_options
        }

def find_available_clusters():
    """Find available cluster data files"""
    available_clusters = []
    results_folder = "results"
    if os.path.exists(results_folder):
        for file in os.listdir(results_folder):
            if file.startswith("df_100_eval_") and file.endswith(".csv"):
                match = re.search(r'df_100_eval_(\d+)_min_100_minwomen_50\.0_s1r_0\.0\.csv', file)
                if match:
                    available_clusters.append(int(match.group(1)))
    
    available_clusters.sort()
    return available_clusters

def get_metric_details(metric_type):
    """Get metric column name and description"""
    if metric_type == "Combined Criteria":
        return "combined_criteria_pct", "Combined S1R≥3 AND S2R≥2"
    elif metric_type == "S1R Top2Box Only":
        return "S1R_top2box_pct", "S1R Top2Box percentage"
    else:  # S2R Top2Box Only
        return "S2R_top2box_pct", "S2R Top2Box percentage"

def get_cluster_selection(analysis_mode, available_clusters):
    """Get cluster selection based on analysis mode"""
    if analysis_mode == "Single Cluster":
        return [st.selectbox(
            "🎯 Select Skin Tone Cluster",
            available_clusters,
            help="Choose which skin tone cluster to analyze"
        )]
    elif analysis_mode == "Compare Clusters (DeltaE)":
        st.info("You'll select clusters to compare on the main page")
        return available_clusters  # Load all for comparison
    else:  # Universal Regions
        selected = st.multiselect(
            "🌍 Clusters for Universal Analysis",
            available_clusters,
            default=available_clusters,
            help="Select clusters to find universal color regions"
        )
        
        if len(selected) < 2:
            st.warning("Please select at least 2 clusters for universal analysis!")
            st.stop()
        
        return selected

def get_universal_parameters(selected_clusters):
    """Get universal region parameters"""
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
    
    return {
        'min_clusters_required': min_clusters_required,
        'universal_threshold': universal_threshold
    }

def get_rendering_options():
    """Get hair rendering options"""
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
    
    return {
        'auto_render': auto_render,
        'swatch_type': swatch_type
    }

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
                st.rerun()
    
    return st.session_state.get('selected_region')