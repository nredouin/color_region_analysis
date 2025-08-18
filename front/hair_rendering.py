import streamlit as st
import pandas as pd
import os
import re

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

def create_short_universal_name(group_name, suffix=""):
    """Create a short name for universal groups to avoid filesystem limits"""
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