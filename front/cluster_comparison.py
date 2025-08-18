import streamlit as st
import pandas as pd
from front.hair_rendering import render_best_matching_pair, render_centroid_hair

def show_cluster_comparison_analysis(config):
    """Show comparison analysis between two clusters using DeltaE"""
    st.header("🔍 Cluster Comparison Analysis (DeltaE Method)")
    st.info(f"**Performance Metric**: {config['metric_description']}")
    
    available_clusters = config['available_clusters']
    
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
            "Main Color DeltaE Threshold",
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
    comparison_params = get_comparison_parameters(delta_e_threshold)
    
    # Perform comparison
    if st.button("🔍 Perform DeltaE Comparison", type="primary"):
        perform_comparison(cluster_1, cluster_2, comparison_params, config)
    
    # Display results if available
    if 'comparison_results' in st.session_state:
        show_comparison_results()
        show_similar_pairs_recap()

def get_comparison_parameters(delta_e_threshold):
    """Get comparison parameters"""
    st.subheader("🔧 Comparison Parameters")
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
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
    
    return {
        'delta_e_threshold': delta_e_threshold,
        'reflect_threshold': reflect_threshold,
        'top_n_compare': top_n_compare,
        'max_pairs_display': max_pairs_display,
        'reflect_multiplier': reflect_multiplier
    }

def perform_comparison(cluster_1, cluster_2, params, config):
    """Perform the actual cluster comparison"""
    with st.spinner(f"Comparing clusters {cluster_1} and {cluster_2}..."):
        try:
            # Load data for both clusters
            cluster_results = {}
            for cluster_id in [cluster_1, cluster_2]:
                file_path = f"results/df_100_eval_{cluster_id}_min_100_minwomen_50.0_s1r_0.0.csv"
                
                analysis_results = st.session_state.analyzer.analyze_color_regions(
                    file_path, 
                    threshold=config['threshold'], 
                    top_n=params['top_n_compare'], 
                    metric_column=config['metric_column']
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
                params['delta_e_threshold'], 
                params['top_n_compare'],
                config['metric_column'],
                params['reflect_threshold']
            )
            
            # Store results in session state
            store_comparison_results(comparison_results, cluster_1, cluster_2, cluster_results, params, config)
            
            st.success("✅ Comparison analysis completed!")
            
        except Exception as e:
            st.error(f"Error during comparison: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def store_comparison_results(comparison_results, cluster_1, cluster_2, cluster_results, params, config):
    """Store comparison results in session state"""
    st.session_state.comparison_results = comparison_results
    st.session_state.cluster_1 = cluster_1
    st.session_state.cluster_2 = cluster_2
    st.session_state.cluster_results_comparison = cluster_results
    st.session_state.max_pairs_display = params['max_pairs_display']
    st.session_state.metric_column = config['metric_column']
    st.session_state.metric_description = config['metric_description']
    st.session_state.reflect_multiplier = params['reflect_multiplier']
    st.session_state.reflect_threshold = params['reflect_threshold']

def show_comparison_results():
    """Display comparison results"""
    comparison_results = st.session_state.comparison_results
    similar_pairs = comparison_results['similar_pairs']
    both_criteria_pairs = comparison_results['both_criteria_pairs']
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
    
    # Display metrics
    show_comparison_metrics(comparison_results, both_criteria_pairs, similar_pairs)
    
    # Display options for different pair types
    display_pairs = get_display_pairs_selection(both_criteria_pairs, similar_pairs, comparison_df, cluster_1, cluster_2)
    
    # DeltaE scatter plot
    show_deltae_analysis(comparison_df, display_pairs, cluster_1, cluster_2)
    
    # Similar pairs visualization and table
    if len(display_pairs) > 0:
        show_similar_pairs_visualization(display_pairs, cluster_1, cluster_2, max_pairs_display)
        show_detailed_pairs_table(display_pairs, cluster_1, cluster_2)
        show_hair_rendering_section(display_pairs)
    else:
        show_no_pairs_found_message(comparison_results)

def show_comparison_metrics(comparison_results, both_criteria_pairs, similar_pairs):
    """Show comparison metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Comparisons", comparison_results['total_comparisons'])
    
    with col2:
        st.metric("Main Similar", comparison_results['main_similar_count'])
    
    with col3:
        st.metric("Both Criteria Met", comparison_results['both_criteria_count'])
    
    with col4:
        st.metric("Reflect Checked", comparison_results['reflect_checked_count'])
    
    with col5:
        if len(both_criteria_pairs) > 0:
            best_match = both_criteria_pairs.iloc[0]
            st.metric("Best Match ΔE", f"{best_match['delta_e_main']:.2f}")
        elif len(similar_pairs) > 0:
            best_match = similar_pairs.iloc[0]
            st.metric("Best Match ΔE", f"{best_match['delta_e_main']:.2f}")
        else:
            st.metric("Best Match ΔE", "N/A")
    
    # Success rates
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Main Similarity Rate", f"{comparison_results.get('main_similarity_rate', 0):.1f}%")
    with col2:
        st.metric("Both Criteria Rate", f"{comparison_results.get('both_criteria_rate', 0):.1f}%")
    with col3:
        reflect_check_rate = (comparison_results['reflect_checked_count'] / comparison_results['total_comparisons'] * 100) if comparison_results['total_comparisons'] > 0 else 0
        st.metric("Reflect Check Rate", f"{reflect_check_rate:.1f}%")

def get_display_pairs_selection(both_criteria_pairs, similar_pairs, comparison_df, cluster_1, cluster_2):
    """Get display pairs based on user selection"""
    main_threshold = st.session_state.comparison_results.get('main_delta_e_threshold', 3.0)
    reflect_threshold = st.session_state.comparison_results.get('reflect_delta_e_threshold', main_threshold * 2)
    
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
    
    return display_pairs

def show_deltae_analysis(comparison_df, display_pairs, cluster_1, cluster_2):
    """Show DeltaE analysis chart"""
    st.subheader("📈 DeltaE Analysis")
    fig_scatter = st.session_state.comparator.create_delta_e_scatter_plotly(
        comparison_df, display_pairs, cluster_1, cluster_2
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

def show_similar_pairs_visualization(display_pairs, cluster_1, cluster_2, max_pairs_display):
    """Show similar pairs color visualization"""
    st.subheader("🎨 Similar Color Pairs")
    
    # Color comparison plot
    fig_colors = st.session_state.comparator.create_color_comparison_plot(
        display_pairs, cluster_1, cluster_2, max_pairs_display
    )
    
    if fig_colors:
        st.pyplot(fig_colors, clear_figure=True)

def show_detailed_pairs_table(display_pairs, cluster_1, cluster_2):
    """Show detailed pairs table"""
    st.subheader("📋 Detailed Pairs Table")
    
    # Format pairs for display with new columns
    display_pairs_copy = display_pairs.copy()
    display_cols = [
        f'region_{cluster_1}_id', f'region_{cluster_2}_id', 
        'delta_e_main', 'delta_e_reflect',
        'similar_main', 'similar_reflect', 'meets_both_criteria',
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

def show_hair_rendering_section(display_pairs):
    """Show hair rendering section for best match"""
    st.subheader("💇‍♀️ Hair Rendering - Best Match")
    
    if st.button("🎨 Render Best Matching Pair"):
        render_best_matching_pair(
            display_pairs, 
            st.session_state.cluster_results_comparison, 
            st.session_state.cluster_1, 
            st.session_state.cluster_2
        )

def show_no_pairs_found_message(comparison_results):
    """Show message when no pairs are found"""
    main_threshold = comparison_results.get('main_delta_e_threshold', 3.0)
    reflect_threshold = comparison_results.get('reflect_delta_e_threshold', main_threshold * 2)
    
    st.warning(f"No pairs found meeting the selected criteria")
    st.info("Try increasing the thresholds or check different pair types.")

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
        st.session_state.recap_results = recap_results
    
    st.markdown("---")
    st.subheader("🔄 Similar Pairs Recap & Centroid Analysis")
    st.info("This section shows regions that match with multiple regions from the other cluster and their centroid colors.")
    
    show_recap_content(recap_results, cluster_1, cluster_2)

def show_recap_content(recap_results, cluster_1, cluster_2):
    """Show recap content"""
    grouped_pairs = recap_results['grouped_pairs']
    centroid_data = recap_results['centroid_data']
    summary_stats = recap_results['summary_stats']
    
    if len(grouped_pairs) == 0:
        st.warning("No multi-match groups found. Each region only matches with one region from the other cluster.")
        return
    
    # Summary metrics
    show_recap_summary_metrics(summary_stats, cluster_1, cluster_2)
    
    # Grouped pairs performance chart and centroid visualization
    show_recap_visualizations(recap_results, cluster_1, cluster_2)
    
    # Detailed groups table
    show_recap_detailed_table(grouped_pairs, cluster_1, cluster_2)
    
    # Hair rendering for best multi-match group
    show_recap_hair_rendering(grouped_pairs, centroid_data)
    
    # Export section
    show_recap_export_section(recap_results, cluster_1, cluster_2)

def show_recap_summary_metrics(summary_stats, cluster_1, cluster_2):
    """Show recap summary metrics"""
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

def show_recap_visualizations(recap_results, cluster_1, cluster_2):
    """Show recap visualizations"""
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

def show_recap_detailed_table(grouped_pairs, cluster_1, cluster_2):
    """Show recap detailed table"""
    st.subheader("📋 Detailed Multi-Match Groups")
    
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

def show_recap_hair_rendering(grouped_pairs, centroid_data):
    """Show recap hair rendering section"""
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

def show_recap_export_section(recap_results, cluster_1, cluster_2):
    """Show recap export section"""
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