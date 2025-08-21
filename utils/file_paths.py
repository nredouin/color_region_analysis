import os

def build_file_path(cluster_id, category=None, eval_cluster=None, min_age=None):
    """
    Build dynamic file path based on the naming convention
    df_100_{category}_eval_{eval_cluster}_age_{min_age}-70_min_100_minwomen_50.0_s1r_0.0.csv
    """
    base_filename = "df_100"
    
    # Add category if specified
    if category and category.lower() != "all":
        base_filename += f"_{category}"
    
    # Add eval_cluster if specified (the specific cluster we're analyzing)
    if eval_cluster:
        base_filename += f"_eval_{eval_cluster}"
    
    # Add age if specified
    if min_age:
        base_filename += f"_age_{min_age}-70"
    
    # Add the fixed parts
    base_filename += "_min_100_minwomen_50.0_s1r_0.0.csv"
    
    # Construct full path
    file_path = f"data/color_regions/{base_filename}"
    
    return file_path