import torch
import os
import random
from collections import defaultdict
from tqdm import tqdm

from model import load_model_and_weights, prepare_model_for_inference
from image_processing import preprocess_image_for_dino, extract_dino_features
from feature_upsampling import upsample_and_normalize_features
from annotations_loading import parse_annotations # New import

def get_full_pipeline_features(image_path, model):
    """
    A complete, functional pipeline from a single image path to a final feature map.
    
    Args:
        image_path (str): Path to the target image.
        model (torch.nn.Module): A fully prepared (device, eval mode) DINOv3 model.
        
    Returns:
        torch.Tensor: The final, dense, per-pixel normalized feature map.
    """
    
    device = next(model.parameters()).device
    patch_size = model.patch_embed.patch_size[0]
    img_tensor, original_dims = preprocess_image_for_dino(image_path, patch_size, device)
    feature_map, _ = extract_dino_features(model, img_tensor)
    
    normalized_dense_features = upsample_and_normalize_features(feature_map, original_dims)
    
    return normalized_dense_features

def extract_features_for_annotations(annotations, label_to_id_map, model):
    """
    Extracts DINOv3 features for a list of annotated points, optimizing by
    processing all points for a single image at once.

    Args:
        annotations (list): List of (image_path, x, y, label_str) tuples.
        label_to_id_map (dict): Mapping from string label to integer ID.
        model (torch.nn.Module): The prepared DINOv3 model.

    Returns:
        tuple: A tuple containing:
            - X_features (list): A list of feature tensors for the classifier.
            - y_labels (list): A list of corresponding integer labels.
    """
    X_features = []
    y_labels = []

    # Group annotations by image to process each image only once
    grouped_annotations = defaultdict(list)
    for img_path, x, y, label in annotations:
        grouped_annotations[img_path].append({'x': x, 'y': y, 'label': label})
    
    # Process each image and extract features for its annotated points
    for img_path, points in tqdm(grouped_annotations.items(), desc="Extracting Features"):
        # 1. Run the full feature extraction pipeline for the current image
        dense_feature_map = get_full_pipeline_features(img_path, model) # (1, H, W, D)
        
        # 2. For each annotated point in this image, extract the specific feature vector
        for point in points:
            x, y, label_str = point['x'], point['y'], point['label']
            
            # Extract the feature vector at the (y, x) coordinate
            pixel_feature = dense_feature_map[0, y, x, :]
            
            # Store the feature vector and its corresponding integer label
            X_features.append(pixel_feature)
            y_labels.append(label_to_id_map[label_str])
            
    return X_features, y_labels


if __name__ == '__main__':
    # --- Configuration ---
    REPO_DIR = r'C:\Users\20232788\Desktop\Year3\Misc\dinov3'
    WEIGHT_PATH = r'C:\Users\20232788\Desktop\Year3\dbl_capstone\models\dinov3_vits16_pretrain.pth'
    MODEL_NAME = 'dinov3_vits16'
    IMAGE_DIRECTORY = "../seaview/images/"
    ANNOTATION_CSV = "../seaview/annotations/annotations_PAC_TLS.csv" # Path to your annotations file

    # --- Model Loading and Preparation ---
    print("===== Starting: Model Setup =====")
    dino_model_cpu = load_model_and_weights(REPO_DIR, WEIGHT_PATH, MODEL_NAME)
    model, device = prepare_model_for_inference(dino_model_cpu)
    print(f"✅ Model loaded and prepared for inference on device: '{device}'.\n")

    # --- Task 4.1: Parse Annotations ---
    print("===== Phase 1: Parsing Annotations =====")
    annotations, label_to_id, id_to_label = parse_annotations(ANNOTATION_CSV, IMAGE_DIRECTORY)
    num_unique_images = len(set(ann[0] for ann in annotations))
    print(f"✅ Found {len(annotations)} annotation points across {num_unique_images} unique images.")
    print(f"✅ Found {len(label_to_id)} unique classes.\n")

    # --- Task 4.2: Extract Features for All Points ---
    print("===== Phase 2: Extracting Features for Sanity Check =====")
    X_features, y_labels = extract_features_for_annotations(annotations, label_to_id, model)
    
    # --- Verification ---
    print("\n===== Verification of Extracted Data =====")
    print(f"Total feature vectors extracted: {len(X_features)}")
    print(f"Total labels extracted: {len(y_labels)}")
    assert len(X_features) == len(annotations), "Mismatch: Number of features should equal number of annotations."
    
    # Check a sample
    sample_idx = random.randint(0, len(X_features) - 1)
    sample_feature = X_features[sample_idx]
    sample_label_id = y_labels[sample_idx]
    sample_label_name = id_to_label[sample_label_id]
    
    print(f"\n--- Sample Point ---")
    print(f"Shape of a sample feature vector: {sample_feature.shape}")
    print(f"Data type: {sample_feature.dtype}")
    print(f"Device: {sample_feature.device}")
    print(f"Sample Label: '{sample_label_name}' (ID: {sample_label_id})")
    print("\n🎉 Annotation parsing and feature extraction for classifier sanity check are complete! 🎉")