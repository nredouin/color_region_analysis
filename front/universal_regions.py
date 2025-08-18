import streamlit as st
import pandas as pd
from front.hair_rendering import render_universal_hair_complete, render_universal_centroid

def show_universal_regions_analysis(cluster_results, config):
    """Universal regions analysis - finds similar regions across multiple clusters"""
    st.header("🌍 Universal Regions Analysis")
    st.info(f"This analysis finds color regions that perform well across multiple skin tone clusters using DeltaE similarity.\n\n**Performance Metric**: {config['metric_description']}")
    
    if len(cluster_results) < 2:
        st.error("Universal analysis requires at least 2 clusters!")
        return
    
    # Get universal analysis parameters
    universal_params = get_universal_analysis_parameters(config)
    
    # Perform universal analysis
    if st.button("🔍 Find Universal Regions", type="primary"):
        perform_universal_analysis(cluster_results, universal_params, config)
    
    # Display results if available
    if 'universal_results' in st.session_state:
        show_universal_results()

def get_universal_analysis_parameters(config):
    """Get universal analysis parameters"""
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
            value=config['universal_params']['universal_threshold'],
            help="Minimum performance required in each cluster",
            key="universal_perf_threshold"
        )
    
    return {
        'delta_e_threshold': delta_e_threshold,
        'top_n_universal': top_n_universal,
        'min_performance_threshold': min_performance_threshold,
        'min_clusters_required': config['universal_params']['min_clusters_required']
    }

def perform_universal_analysis(cluster_results, universal_params, config):
    """Perform the universal analysis"""
    with st.spinner("Analyzing universal regions across clusters..."):
        try:
            # Perform universal analysis
            universal_results = st.session_state.comparator.find_universal_regions(
                cluster_results, 
                universal_params['delta_e_threshold'], 
                universal_params['min_clusters_required'],
                universal_params['min_performance_threshold'],
                universal_params['top_n_universal'],
                config['metric_column']
            )
            
            # Store results in session state
            st.session_state.universal_results = universal_results
            st.session_state.universal_params = {
                'delta_e_threshold': universal_params['delta_e_threshold'],
                'min_clusters_required': universal_params['min_clusters_required'],
                'min_performance_threshold': universal_params['min_performance_threshold'],
                'swatch_type': config['rendering_options']['swatch_type'],
                'metric_column': config['metric_column'],
                'metric_description': config['metric_description']
            }
            
            st.success("✅ Universal regions analysis completed!")
            
        except Exception as e:
            st.error(f"Error during universal analysis: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def show_universal_results():
    """Display universal regions analysis results"""
    universal_results = st.session_state.universal_results
    params = st.session_state.universal_params
    
    universal_regions = universal_results['universal_regions']
    cluster_coverage = universal_results['cluster_coverage']
    centroid_colors = universal_results['centroid_colors']
    performance_summary = universal_results['performance_summary']
    
    # Results overview
    show_universal_overview(universal_results, universal_regions, params)
    
    if len(universal_regions) == 0:
        show_no_universal_regions_message(params)
        return
    
    # Debug information (optional)
    show_debug_information(universal_regions, centroid_colors, params)
    
    # Visualizations
    show_universal_visualizations(universal_results, universal_regions, centroid_colors)
    
    # Detailed table
    show_universal_detailed_table(universal_regions)
    
    # Hair rendering section
    show_universal_hair_rendering(universal_regions, centroid_colors)
    
    # Export section
    show_universal_export_section(universal_results, params)

def show_universal_overview(universal_results, universal_regions, params):
    """Show universal analysis overview"""
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

def show_no_universal_regions_message(params):
    """Show message when no universal regions are found"""
    st.warning(f"No universal color groups found with the current parameters:")
    st.write(f"- DeltaE threshold: {params['delta_e_threshold']}")
    st.write(f"- Minimum clusters required: {params['min_clusters_required']}")
    st.write(f"- Performance threshold: {params['min_performance_threshold']}%")
    st.write(f"- Metric used: {params['metric_description']}")
    st.info("Try increasing the DeltaE threshold or reducing the minimum clusters required.")

def show_debug_information(universal_regions, centroid_colors, params):
    """Show debug information if requested"""
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

def show_universal_visualizations(universal_results, universal_regions, centroid_colors):
    """Show universal regions visualizations"""
    # Performance visualization
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
    
    # Color swatches
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

def show_universal_detailed_table(universal_regions):
    """Show detailed universal regions table"""
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

def show_universal_hair_rendering(universal_regions, centroid_colors):
    """Show universal hair rendering section"""
    st.subheader("💇‍♀️ Hair Rendering - Universal Groups")

    try:
        if len(universal_regions) > 0 and len(centroid_colors) > 0:
            
            # Create selection options
            group_options, group_lookup = create_group_selection_options(universal_regions)
            
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
            selected_centroid = find_group_centroid(selected_group, centroid_colors)
            
            if selected_centroid:
                # Display selected group info
                show_selected_group_info(selected_group)
                
                # Color preview and rendering
                show_color_preview_and_rendering(selected_centroid, selected_group)
            else:
                st.warning("Could not find centroid data for selected group")
        
        else:
            st.warning("No universal groups or centroid data available for rendering")
            
    except Exception as e:
        st.error(f"Error in hair rendering section: {e}")
        if st.checkbox("Show rendering error details", key="debug_rendering"):
            import traceback
            st.code(traceback.format_exc())

def create_group_selection_options(universal_regions):
    """Create group selection options for rendering"""
    group_options = []
    group_lookup = {}
    
    for i, group in enumerate(universal_regions):
        option_text = f"UG{i+1}: {len(group['cluster_regions'])} clusters, {group['avg_performance']:.1f}% avg performance"
        group_options.append(option_text)
        group_lookup[option_text] = group
    
    return group_options, group_lookup

def find_group_centroid(selected_group, centroid_colors):
    """Find corresponding centroid for selected group"""
    for centroid in centroid_colors:
        if centroid['group_name'] == selected_group['group_name']:
            return centroid
    return None

def show_selected_group_info(selected_group):
    """Show selected group information"""
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

def show_color_preview_and_rendering(selected_centroid, selected_group):
    """Show color preview and rendering options"""
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
    show_advanced_rendering_options(selected_centroid, selected_group, swatch_template)

def show_advanced_rendering_options(selected_centroid, selected_group, swatch_template):
    """Show advanced rendering options"""
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

def show_universal_export_section(universal_results, params):
    """Show universal export section"""
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