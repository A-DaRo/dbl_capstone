# File: main.py
# Description: Example script to demonstrate model instantiation and usage.

import torch
from PIL import Image
import requests
from transformers import SegformerFeatureExtractor
from typing import Tuple

# Assumes model.py is in the same directory
from MTL_approach.model import CoralMTLSegFormerHuggingFace

# --- Constants ---
NUM_GENUS_CLASSES = 15
NUM_HEALTH_CLASSES = 5
MODEL_ID = "nvidia/mit-b5"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def initialize_components(model_id: str) -> Tuple[CoralMTLSegFormerHuggingFace, SegformerFeatureExtractor]:
    """Loads the model and its corresponding preprocessor."""
    print(f"Initializing model with '{model_id}' encoder...")
    model = CoralMTLSegFormerHuggingFace(
        n_genus_classes=NUM_GENUS_CLASSES,
        n_health_classes=NUM_HEALTH_CLASSES,
        encoder_name=model_id,
        decoder_segmentation_channels=768
    )
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_id)
    return model, feature_extractor


def load_and_preprocess_image(url: str, feature_extractor: SegformerFeatureExtractor) -> torch.Tensor:
    """Downloads an image and prepares it for the model."""
    print(f"Loading image from: {url}")
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs['pixel_values']


def main():
    """Main execution function."""
    # 1. Setup
    model, feature_extractor = initialize_components(MODEL_ID)
    model.eval()

    # 2. Data Preparation
    pixel_values = load_and_preprocess_image(IMAGE_URL, feature_extractor)
    H, W = pixel_values.shape[-2:]

    # 3. Inference
    print(f"\nPerforming forward pass with input shape: {pixel_values.shape}")
    with torch.no_grad():
        outputs = model(pixel_values)

    # 4. Verification
    print("Forward pass successful. Output shapes:")
    for task, mask in outputs.items():
        print(f"  - Task '{task}': {mask.shape}")
    
    assert outputs["genus"].shape == (1, NUM_GENUS_CLASSES, H, W)
    assert outputs["health"].shape == (1, NUM_HEALTH_CLASSES, H, W)
    print("\nOutput dimensions are correct.")


if __name__ == "__main__":
    main()