import torch
from PIL import Image
from torchvision import transforms

def preprocess_image_for_dino(image_path, patch_size, device):
    """
    Loads, resizes, and transforms an image to be compatible with a ViT model.

    Args:
        image_path (str): Path to the input image.
        patch_size (int): The patch size of the ViT model.
        device (str): The device to move the final tensor to.

    Returns:
        tuple: A tuple containing:
            - img_tensor (torch.Tensor): The preprocessed image tensor (B, C, H, W).
            - original_dims (tuple): The original (H, W) of the image.
    """
    img = Image.open(image_path).convert("RGB")
    w_orig, h_orig = img.size

    # Calculate new dimensions that are multiples of the patch size
    h_proc = (h_orig // patch_size) * patch_size
    w_proc = (w_orig // patch_size) * patch_size
    
    transform = transforms.Compose([
        transforms.Resize((h_proc, w_proc)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)
    
    return img_tensor, (h_orig, w_orig)

def extract_dino_features(model, img_tensor):
    """
    Performs a forward pass to get patch features and reshapes them into a 2D map.

    Args:
        model (torch.nn.Module): The DINOv3 model.
        img_tensor (torch.Tensor): The preprocessed image tensor.

    Returns:
        tuple: A tuple containing:
            - feature_map (torch.Tensor): The coarse 2D feature map (B, H', W', D).
            - embedding_dim (int): The dimension of the feature vectors.
    """
    patch_size = model.patch_embed.patch_size[0]
    h_proc, w_proc = img_tensor.shape[2], img_tensor.shape[3]

    with torch.no_grad():
        features_dict = model.forward_features(img_tensor)
        patch_tokens = features_dict['x_norm_patchtokens']

    embedding_dim = patch_tokens.shape[-1]
    h_patches = h_proc // patch_size
    w_patches = w_proc // patch_size
    
    feature_map = patch_tokens.reshape(1, h_patches, w_patches, embedding_dim)
    
    return feature_map, embedding_dim
