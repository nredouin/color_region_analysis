import streamlit as st
import pandas as pd

def create_export_section(cluster_results, config):
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
                        cluster_data['metric_type'] = config['metric_type']
                        all_data.append(cluster_data)
                    
                    if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        
                        # Create Excel file
                        excel_data = st.session_state.helpers.create_export_excel(
                            combined_df, f"Analysis_{config['analysis_mode'].replace(' ', '_')}_{config['metric_type'].replace(' ', '_')}"
                        )
                        
                        st.download_button(
                            "📥 Download Excel File",
                            excel_data,
                            file_name=f"hair_color_analysis_{config['analysis_mode'].replace(' ', '_').lower()}_{config['metric_type'].replace(' ', '_').lower()}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("✅ Excel export ready!")
                except Exception as e:
                    st.error(f"Export error: {e}")
        
        with export_col2:
            st.info(f"Analysis includes {len(config['selected_clusters'])} cluster(s) using {config['metric_type']}")
    else:
        st.warning("No data available for export")

def export_comparison_results(comparison_results, cluster_1, cluster_2):
    """Export comparison results"""
    try:
        # Prepare export data
        export_data = {
            'similar_pairs': comparison_results['similar_pairs'],
            'both_criteria_pairs': comparison_results['both_criteria_pairs'],
            'comparison_summary': pd.DataFrame([{
                'cluster_1': cluster_1,
                'cluster_2': cluster_2,
                'total_comparisons': comparison_results['total_comparisons'],
                'main_similar_count': comparison_results['main_similar_count'],
                'both_criteria_count': comparison_results['both_criteria_count'],
                'reflect_checked_count': comparison_results['reflect_checked_count']
            }])
        }
        
        # Create Excel file
        from io import BytesIO
        buffer = BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            if len(export_data['similar_pairs']) > 0:
                export_data['similar_pairs'].to_excel(writer, sheet_name='Similar_Pairs', index=False)
            if len(export_data['both_criteria_pairs']) > 0:
                export_data['both_criteria_pairs'].to_excel(writer, sheet_name='Both_Criteria_Pairs', index=False)
            export_data['comparison_summary'].to_excel(writer, sheet_name='Summary', index=False)
        
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Export error: {e}")
        return None

def export_universal_results(universal_results, params):
    """Export universal analysis results"""
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
        
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Export error: {e}")
        return None