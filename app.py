import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import warnings

# Import our custom modules
from utils.color_analysis import ColorAnalyzer
from utils.hair_rendering import HairRenderer
from utils.streamlit_helpers import StreamlitHelpers
from utils.cluster_comparison import ClusterComparator

# Import our new modular components
from front.ui_components import create_sidebar, create_main_header
from front.single_cluster_analysis import show_single_cluster_analysis
from front.cluster_comparison import show_cluster_comparison_analysis
from front.universal_regions import show_universal_regions_analysis
from front.export_utils import create_export_section

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="L'Oréal Multi-Cluster Hair Color Analysis",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Apply custom styling
    create_main_header()
    
    # Create sidebar and get configuration
    config = create_sidebar()
    
    if not config:
        return
    
    # Initialize components in session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ColorAnalyzer()
        st.session_state.renderer = HairRenderer()
        st.session_state.helpers = StreamlitHelpers()
        st.session_state.comparator = ClusterComparator()
    
    # Load and analyze data for selected clusters
    cluster_results = {}
    
    if config['analysis_mode'] != "Compare Clusters (DeltaE)":
        with st.spinner(f"Loading data for {len(config['selected_clusters'])} cluster(s)..."):
            for cluster_id in config['selected_clusters']:
                file_path = f"results/df_100_eval_{cluster_id}_min_100_minwomen_50.0_s1r_0.0.csv"
                
                try:
                    analysis_results = st.session_state.analyzer.analyze_color_regions(
                        file_path, 
                        threshold=config['threshold'], 
                        top_n=config['top_n_regions'], 
                        metric_column=config['metric_column']
                    )
                    
                    if analysis_results is not None:
                        cluster_results[cluster_id] = analysis_results
                        
                except Exception as e:
                    st.error(f"Error loading data for cluster {cluster_id}: {str(e)}")
        
        if not cluster_results and config['analysis_mode'] != "Compare Clusters (DeltaE)":
            st.error("Failed to load any cluster data!")
            st.stop()

    # Route to appropriate analysis based on mode
    if config['analysis_mode'] == "Single Cluster":
        show_single_cluster_analysis(cluster_results, config)
        
    elif config['analysis_mode'] == "Compare Clusters (DeltaE)":
        show_cluster_comparison_analysis(config)
        
    else:  # Universal Regions
        show_universal_regions_analysis(cluster_results, config)

    # Export section (only for non-comparison modes)
    if config['analysis_mode'] != "Compare Clusters (DeltaE)":
        st.markdown("---")
        create_export_section(cluster_results, config)

if __name__ == "__main__":
    main()