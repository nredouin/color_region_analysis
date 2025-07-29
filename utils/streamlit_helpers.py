import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import base64
from io import BytesIO

class StreamlitHelpers:
    """Helper functions for Streamlit UI components"""
    
    def __init__(self):
        self.loreal_colors = {
            'primary': '#2649B2',
            'secondary': '#4A74F3',
            'accent': '#8E7DE3',
            'purple': '#9D5CE6',
            'light': '#D4D9F0',
            'blue': '#6C8BE0',
            'violet': '#B55CE6'
        }
    
    def lab_to_rgb(self, lab_l, lab_a, lab_b):
        """Convert LAB color to RGB for visualization"""
        try:
            lab_color = LabColor(lab_l, lab_a, lab_b)
            rgb_color = convert_color(lab_color, sRGBColor)
            rgb = np.clip([rgb_color.rgb_r, rgb_color.rgb_g, rgb_color.rgb_b], 0, 1)
            return rgb
        except:
            return [0.5, 0.5, 0.5]  # Gray fallback
    
    def calculate_chroma_hue(self, a, b):
        """Calculate Chroma and Hue from a* and b* values"""
        if pd.isna(a) or pd.isna(b):
            return np.nan, np.nan
        
        # Chroma = sqrt(a² + b²)
        chroma = np.sqrt(a**2 + b**2)
        
        # Hue = atan2(b, a) * 180/π
        hue = np.arctan2(b, a) * 180 / np.pi
        # Ensure hue is in 0-360 range
        if hue < 0:
            hue += 360
        
        return chroma, hue
    
    def create_color_swatch_plotly(self, lab_l, lab_a, lab_b, title="Color Swatch"):
        """Create a color swatch using Plotly"""
        rgb = self.lab_to_rgb(lab_l, lab_a, lab_b)
        rgb_str = f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 0, 0],
            y=[0, 0, 1, 1, 0],
            fill="toself",
            fillcolor=rgb_str,
            line=dict(color="black", width=2),
            mode="lines",
            showlegend=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=200,
            height=200,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        return fig
    
    def create_performance_scatter(self, df, threshold=45):
        """Create interactive performance scatter plot"""
        # Filter for visualization (only regions with S2R data)
        viz_data = df[df['S2R_total'] > 0].copy()
        
        if len(viz_data) == 0:
            return None
        
        # Color points based on combined criteria threshold
        colors = ['green' if (not pd.isna(row['combined_criteria_pct']) and 
                            row['combined_criteria_pct'] >= threshold) 
                 else 'red' for _, row in viz_data.iterrows()]
        
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=viz_data['S1R_top2box_pct'],
            y=viz_data['S2R_top2box_pct'],
            mode='markers+text',
            marker=dict(
                size=viz_data['total_samples'] * 0.5,
                color=colors,
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            text=[f"R{int(row['color_regions'])}<br>{row['combined_criteria_pct']:.1f}%" 
                  for _, row in viz_data.iterrows()],
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            hovertemplate="<b>Region %{text}</b><br>" +
                         "S1R TOP2BOX: %{x:.1f}%<br>" +
                         "S2R TOP2BOX: %{y:.1f}%<br>" +
                         "Samples: %{marker.size}<br>" +
                         "<extra></extra>",
            showlegend=False
        ))
        
        # Add threshold lines
        fig.add_hline(y=40, line_dash="dash", line_color="orange", 
                     annotation_text="S2R Reference (40%)")
        fig.add_vline(x=50, line_dash="dash", line_color="blue", 
                     annotation_text="S1R Reference (50%)")
        
        fig.update_layout(
            title="Combined Criteria Analysis: TOP2BOX Performance",
            xaxis_title="S1R TOP2BOX % (Ratings 4-5)",
            yaxis_title="S2R TOP2BOX % (Ratings 2-3)",
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_lab_comparison_chart(self, main_lab, reflect_lab, region_id):
        """Create LAB comparison chart"""
        categories = ['L*', 'a*', 'b*']
        main_values = [main_lab[0], main_lab[1], main_lab[2]]
        reflect_values = [reflect_lab[0], reflect_lab[1], reflect_lab[2]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Main Color',
            x=categories,
            y=main_values,
            marker_color=self.loreal_colors['primary']
        ))
        
        fig.add_trace(go.Bar(
            name='Reflect Color',
            x=categories,
            y=reflect_values,
            marker_color=self.loreal_colors['secondary']
        ))
        
        fig.update_layout(
            title=f"LAB Values Comparison - Region {region_id}",
            xaxis_title="LAB Components",
            yaxis_title="Values",
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_performance_radar(self, region_data):
        """Create radar chart for performance metrics"""
        categories = ['S1R Mean', 'S2R Mean', 'S1R TOP2BOX%', 'S2R TOP2BOX%', 'Combined Score%']
        
        # Normalize values to 0-100 scale for radar chart
        values = [
            region_data['S1R_mean'] * 20,  # Scale 1-5 to 20-100
            region_data['S2R_mean'] * 33.33 if not pd.isna(region_data['S2R_mean']) else 0,  # Scale 1-3 to 33-100
            region_data['S1R_top2box_pct'],
            region_data['S2R_top2box_pct'] if not pd.isna(region_data['S2R_top2box_pct']) else 0,
            region_data['combined_criteria_pct']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor=self.loreal_colors['accent'],
            line_color=self.loreal_colors['primary'],
            name='Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title=f"Performance Radar - Region {int(region_data['color_regions'])}",
            height=400
        )
        
        return fig
    
    def format_metric_card(self, title, value, delta=None, delta_color="normal"):
        """Create a formatted metric card"""
        return f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: {self.loreal_colors['primary']};">{title}</h4>
            <h2 style="margin: 5px 0; color: #333;">{value}</h2>
            {f'<p style="margin: 0; color: {delta_color};">{delta}</p>' if delta else ''}
        </div>
        """
    
    def create_export_excel(self, data, sheet_name="Analysis"):
        """Create Excel file for download"""
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return buffer.getvalue()
    
    def create_download_link(self, data, filename, mime_type):
        """Create download link for data"""
        b64 = base64.b64encode(data).decode()
        return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    
    def display_color_info_table(self, lab_values, chroma_hue=None):
        """Display color information in a formatted table"""
        data = {
            'Component': ['L* (Lightness)', 'a* (Green-Red)', 'b* (Blue-Yellow)'],
            'Value': [f"{lab_values[0]:.2f}", f"{lab_values[1]:.2f}", f"{lab_values[2]:.2f}"],
            'Description': [
                '0=Black, 100=White',
                'Negative=Green, Positive=Red',
                'Negative=Blue, Positive=Yellow'
            ]
        }
        
        if chroma_hue:
            data['Component'].extend(['Chroma', 'Hue'])
            data['Value'].extend([f"{chroma_hue[0]:.2f}", f"{chroma_hue[1]:.2f}°"])
            data['Description'].extend(['Color saturation', 'Color angle (0-360°)'])
        
        return pd.DataFrame(data)
    
    def create_color_harmony_chart(self, region_data):
        """Create color harmony visualization"""
        # This would show color relationships and harmony
        # Implementation depends on specific color theory requirements
        pass
    
    def apply_loreal_styling(self):
        """Apply L'Oréal brand styling to Streamlit"""
        st.markdown(f"""
        <style>
            .stApp {{
                background-color: #fafafa;
            }}
            .main-header {{
                color: {self.loreal_colors['primary']};
                font-family: 'Arial', sans-serif;
            }}
            .stButton > button {{
                background-color: {self.loreal_colors['primary']};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 0.5rem 1rem;
                font-weight: bold;
            }}
            .stButton > button:hover {{
                background-color: {self.loreal_colors['secondary']};
            }}
            .stSelectbox > div > div {{
                border-color: {self.loreal_colors['light']};
            }}
        </style>
        """, unsafe_allow_html=True)