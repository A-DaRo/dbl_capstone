import torch
import torch.nn.functional as F

def upsample_and_normalize_features(feature_map, target_dims):
    """
    Upsamples a coarse feature map to target dimensions and L2-normalizes each vector.

    Args:
        feature_map (torch.Tensor): The coarse feature map (B, H', W', D).
        target_dims (tuple): The target (H, W) for upsampling.

    Returns:
        torch.Tensor: The dense, per-pixel, normalized feature map (B, H, W, D).
    """
    h_orig, w_orig = target_dims

    # Permute from (B, H', W', D) to (B, D, H', W') for interpolation
    feature_map_for_interp = feature_map.permute(0, 3, 1, 2)
    
    upsampled_features = F.interpolate(
        feature_map_for_interp,
        size=(h_orig, w_orig),
        mode='bilinear',
        align_corners=False
    )
    
    # Permute back to (B, H, W, D) for intuitive access and normalization
    dense_features = upsampled_features.permute(0, 2, 3, 1)
    
    # L2 Normalization across the feature dimension
    norm = torch.linalg.norm(dense_features, dim=-1, keepdim=True)
    normalized_features = dense_features / (norm + 1e-6)
    
    return normalized_features
