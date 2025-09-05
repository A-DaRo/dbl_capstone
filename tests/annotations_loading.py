import pandas as pd
import os
from tqdm import tqdm

def parse_annotations(csv_path, image_dir):
    """
    Parses a Seaview Survey annotation CSV into a structured format for feature extraction.

    It reads the CSV, builds the full image path for each annotation, converts
    1-based coordinates to 0-based, and creates mappings for class labels.

    Args:
        csv_path (str): Path to the annotation CSV file.
        image_dir (str): Path to the directory containing the images.

    Returns:
        tuple: A tuple containing:
            - annotations (list): A list of tuples `(image_path, x, y, label_str)`.
            - label_to_id (dict): Mapping from string label to a unique integer ID.
            - id_to_label (dict): Mapping from integer ID back to the string label.
    """
    print(f"Reading annotations from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Create label-to-ID mappings
    unique_labels = sorted(df['label'].unique())
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    annotations = []
    print("Processing and verifying annotations...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Annotations"):
        image_name = f"{row['quadratid']}.jpg"
        image_path = os.path.join(image_dir, image_name)
        
        # Convert 1-based coordinates from CSV to 0-based for array indexing
        x = row['x'] - 1
        y = row['y'] - 1
        label_str = row['label']
        
        # Basic validation: ensure the image file exists
        if os.path.exists(image_path):
            annotations.append((image_path, x, y, label_str))
        else:
            print(f"Warning: Image file not found for quadratid {row['quadratid']}, skipping point.")
            
    return annotations, label_to_id, id_to_label