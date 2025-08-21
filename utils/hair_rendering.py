import pandas as pd
import luxpy as lx
import numpy as np
import os
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

class HairRenderer:
    """Class for hair swatch rendering"""
    
    def __init__(self):
        self.base_folder = "data/assets/"
        self.cluster_num = 2
        self.seed = 42
        
    def render_hair_swatch(self, excel_path, swatch_type="Medium", region_id=None):
        """Main rendering function for hair swatches"""
        try:
            # Load the target color data
            ref_data = pd.read_excel(excel_path)
            
            # Sort colors by lightness
            ref_data = self._sort_colors_by_lightness(ref_data)
            
            # Convert LAB to RGB/BGR
            ref_srgb1, ref_sbgr1, ref_hsv1 = self._convert_lab_to_color_spaces(
                ref_data, ['L_1', 'a_1', 'b_1']
            )
            ref_srgb2, ref_sbgr2, ref_hsv2 = self._convert_lab_to_color_spaces(
                ref_data, ['L_2', 'a_2', 'b_2']
            )
            
            # Get input swatch image
            image_path = self._get_swatch_path(swatch_type)
            if not os.path.exists(image_path):
                print(f"Error: Swatch image not found at {image_path}")
                return None
            
            # Process the swatch
            image = Image.open(image_path).convert("RGB")
            cluster_centers, input_hsv = self._analyze_swatch_colors(image)
            
            # Render for each color combination
            rendered_paths = []
            for n in range(ref_srgb1.shape[0]):
                model = ref_data.loc[n, 'model'] if 'model' in ref_data.columns else f"region_{region_id}"
                output_path = self._render_single_swatch(
                    image, cluster_centers, input_hsv,
                    ref_hsv1[n], ref_hsv2[n], model, region_id or n
                )
                if output_path:
                    rendered_paths.append(output_path)
            
            return rendered_paths[0] if rendered_paths else None
            
        except Exception as e:
            print(f"Error in hair rendering: {str(e)}")
            return None
    
    def _sort_colors_by_lightness(self, ref_data):
        """Sort colors by lightness value"""
        for ind, row in ref_data.iterrows():
            if row['L_2'] < row['L_1']:
                # Swap the colors
                ref_data.loc[ind, [f"{prefix}_{i}" for i in np.arange(1, 3) for prefix in ["L", "a", "b"]]] = \
                    row[[f"{prefix}_{i}" for i in [2, 1] for prefix in ["L", "a", "b"]]].values
        return ref_data
    
    def _convert_lab_to_color_spaces(self, ref_data, lab_columns):
        """Convert LAB to RGB, BGR, and HSV color spaces"""
        lab_values = ref_data.loc[:, lab_columns].values
        
        # LAB to RGB
        srgb = lx.xyz_to_srgb(lx.lab_to_xyz(lab_values))
        
        # RGB to BGR (for OpenCV)
        sbgr = srgb[:, [2, 1, 0]]
        
        # BGR to HSV
        hsv = cv2.cvtColor(
            sbgr.reshape(sbgr.shape[0], -1, 3).astype(np.float32), 
            cv2.COLOR_BGR2HSV
        )
        
        return srgb, sbgr, hsv
    
    def _get_swatch_path(self, swatch_type):
        """Get the path to the appropriate swatch template"""
        swatch_paths = {
            "Dark": "input_swatches/Dark/684-ChatainMarron-200000655541-46.png",
            "Medium": "input_swatches/Medium/1047-ChatainMarron-200000655228-39.png",
            "Light": "input_swatches/Light/14-BeigeGris-200000634437-14.png"
        }
        
        return os.path.join(self.base_folder, swatch_paths.get(swatch_type, swatch_paths["Medium"]))
    
    def _analyze_swatch_colors(self, image):
        """Analyze the base swatch to extract main colors"""
        from sklearn.cluster import MiniBatchKMeans
        
        pixels = np.array(image).reshape((-1, 3))
        
        # Remove black pixels
        non_black_pixel_indices = np.any(pixels != [0, 0, 0], axis=1)
        non_black_pixels = pixels[non_black_pixel_indices]
        
        if non_black_pixels.size == 0:
            raise ValueError("Image contains only black pixels")
        
        # Cluster colors
        kmeans = MiniBatchKMeans(n_clusters=self.cluster_num, random_state=self.seed, batch_size=256)
        kmeans.fit(non_black_pixels)
        labels = kmeans.labels_
        
        # Sort clusters by size
        counts = np.bincount(labels)
        sorted_indices = np.argsort(counts)[::-1]
        cluster_centers = kmeans.cluster_centers_[sorted_indices]
        cluster_bgr = cluster_centers[:, [2, 1, 0]]
        
        # Convert to HSV
        input_hsv = cv2.cvtColor(
            cluster_bgr.reshape(2, 1, 3).astype(np.float32), 
            cv2.COLOR_BGR2HSV
        ).flatten().reshape(2, 3)
        
        return cluster_centers, input_hsv
    
    def _render_single_swatch(self, image, cluster_centers, input_hsv, ref_hsv1, ref_hsv2, model, cluster):
        """Render a single swatch with new colors"""
        image_array = np.array(image)
        image_output = np.zeros_like(image_array)
        
        for y in range(image_array.shape[0]):
            for x in range(image_array.shape[1]):
                original_rgb = image_array[y, x]
                original_hsv = cv2.cvtColor(
                    (original_rgb[[2, 1, 0]]).reshape(1, 1, 3).astype(np.float32), 
                    cv2.COLOR_BGR2HSV
                ).flatten()
                
                # Find closest cluster
                closest_ind = self._find_closest_color(original_rgb, cluster_centers)
                
                if closest_ind == 0:
                    temp_hsv = ref_hsv1.flatten()
                    original_hsv[0] = temp_hsv[0]
                    original_hsv[1] = np.clip(
                        temp_hsv[1] * 0.6 + 0.4 * original_hsv[1] * (temp_hsv[1] / input_hsv[0][1]), 
                        0., 1
                    )
                    original_hsv[2] = np.clip(
                        temp_hsv[2] * 0.6 + 0.4 * original_hsv[2] * (temp_hsv[2] / input_hsv[0][2]), 
                        0., 255
                    )
                else:
                    temp_hsv = ref_hsv2.flatten()
                    original_hsv[0] = temp_hsv[0]
                    original_hsv[1] = np.clip(
                        temp_hsv[1] * 0.6 + 0.4 * original_hsv[1] * (temp_hsv[1] / input_hsv[1][1]), 
                        0., 1
                    )
                    original_hsv[2] = np.clip(
                        temp_hsv[2] * 0.6 + 0.4 * original_hsv[2] * (temp_hsv[2] / input_hsv[1][2]), 
                        0., 255
                    )
                
                bgr = cv2.cvtColor(original_hsv.reshape(1, 1, 3), cv2.COLOR_HSV2BGR).flatten()
                image_output[y, x] = bgr[[2, 1, 0]]
        
        # Save the rendered image
        output_folder = os.path.join(self.base_folder, "rendered_output", str(model))
        os.makedirs(output_folder, exist_ok=True)
        
        output_name = f"model{model}c{cluster}_{model}_{cluster}.jpg"
        output_path = os.path.join(output_folder, output_name)
        
        image_output = Image.fromarray(image_output)
        image_output.save(output_path)
        
        print(f"Rendered and saved: {output_path}")
        return output_path
    
    def _find_closest_color(self, pixel_rgb, cluster_centers):
        """Find the closest color in cluster_centers to pixel_rgb"""
        pixel_rgb = np.array(pixel_rgb)
        distances = np.linalg.norm(cluster_centers - pixel_rgb, axis=1)
        closest_index = np.argmin(distances)
        return closest_index