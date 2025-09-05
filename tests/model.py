import torch

def load_model_and_weights(repo_dir, weight_path, model_name='dinov3_vitl16'):
    """
    Loads the DINOv3 model architecture and applies local weights.

    Args:
        repo_dir (str): Path to the local DINOv3 repository clone.
        weight_path (str): Path to the local .pth weight file.
        model_name (str): The name of the model entrypoint in hubconf.py.

    Returns:
        torch.nn.Module: The model with weights loaded.
    """
    model = torch.hub.load(
        repo_or_dir=repo_dir,
        model=model_name,
        source='local',
        pretrained=False,
        force_reload=True
    )
    
    state_dict = torch.load(weight_path)
    
    if 'student' in state_dict:
        state_dict = state_dict['student']

    model.load_state_dict(state_dict)
    
    return model

def prepare_model_for_inference(model):
    """
    Prepares a model for inference by moving to the correct device, setting
    to evaluation mode, and freezing its parameters.

    Args:
        model (torch.nn.Module): The model to prepare.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The prepared model.
            - device (str): The device the model was moved to ('cuda' or 'cpu').
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    # Freeze parameters by default for feature extraction
    for param in model.parameters():
        param.requires_grad = False
        
    return model, device
