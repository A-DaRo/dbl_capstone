import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

# --- Helper MLP Block ---
# A simple MLP block for creating decoders. Using Conv2d is standard for segmentation decoders.
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(output_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))

# --- The Main Hierarchical Context-Aware Decoder ---
class HierarchicalContextAwareDecoder(nn.Module):
    """
    Implements the Hierarchical Context-Aware Decoder for Multi-Task Coral Segmentation.

    This decoder takes features from a shared encoder and branches into five streams:
    - 2 Primary Streams (Genus, Health): Full MLP decoders, enriched via cross-attention.
    - 3 Auxiliary Streams (Fish, Human-Artifact, Substrate): Lightweight heads that provide context.

    The core idea is an asymmetric information flow where auxiliary tasks inform the
    primary tasks, but not vice-versa, enhancing efficiency and task focus.
    """
    def __init__(self,
                 encoder_channels: List[int],
                 decoder_channel: int,
                 num_classes: Dict[str, int],
                 attention_dim: int = 256):
        """
        Args:
            encoder_channels (List[int]): List of channel dimensions from the encoder's stages.
                                          Example for MiT-B2: [64, 128, 320, 512]
            decoder_channel (int): The unified channel dimension for all streams.
            num_classes (Dict[str, int]): A dictionary mapping task names to their number of classes.
                                          Example: {'genus': 9, 'health': 4, 'fish': 2, ...}
            attention_dim (int): The dimension for the query, key, and value in the attention module.
        """
        super().__init__()
        assert len(encoder_channels) == 4, "Requires features from 4 encoder stages."
        self.tasks = ['genus', 'health', 'fish', 'human_artifacts', 'substrate']
        self.primary_tasks = ['genus', 'health']
        self.aux_tasks = ['fish', 'human_artifacts', 'substrate']
        self.decoder_channel = decoder_channel

        # --- 1. Channel Unification MLPs ---
        # Projects encoder features from each stage to the same `decoder_channel` dimension.
        self.linear_c = nn.ModuleList([
            MLP(input_dim=c, output_dim=decoder_channel) for c in encoder_channels
        ])
        
        # Total channels after fusing the 4 stages
        fused_channels = decoder_channel * 4

        # --- 2. Asymmetric Decoder Heads ---
        # Primary tasks get a full MLP decoder block
        self.genus_decoder = MLP(fused_channels, decoder_channel)
        self.health_decoder = MLP(fused_channels, decoder_channel)

        # Auxiliary tasks get lightweight 1x1 Conv heads (simpler than a full MLP block)
        self.fish_head = nn.Conv2d(fused_channels, decoder_channel, kernel_size=1)
        self.human_artifacts_head = nn.Conv2d(fused_channels, decoder_channel, kernel_size=1)
        self.substrate_head = nn.Conv2d(fused_channels, decoder_channel, kernel_size=1)
        
        # Store heads in a ModuleDict for easy access
        self.decoders = nn.ModuleDict({
            'genus': self.genus_decoder,
            'health': self.health_decoder,
            'fish': self.fish_head,
            'human_artifacts': self.human_artifacts_head,
            'substrate': self.substrate_head
        })

        # --- 3. Projections for Cross-Attention Module ---
        # Create Q, K, V projections for each of the 5 tasks.
        self.to_qkv = nn.ModuleDict()
        for task in self.tasks:
            self.to_qkv[task] = nn.ModuleList([
                nn.Conv2d(decoder_channel, attention_dim, 1, bias=False), # Query
                nn.Conv2d(decoder_channel, attention_dim, 1, bias=False), # Key
                nn.Conv2d(decoder_channel, attention_dim, 1, bias=False)  # Value
            ])
        
        # *** FIX 2: Add projection layer to map attention output back to decoder_channel for residual connection ***
        self.to_out = nn.ModuleDict({
            task: nn.Conv2d(attention_dim, decoder_channel, 1) for task in self.primary_tasks
        })

        # --- 4. Final Prediction Layers ---
        # Each task gets a final 1x1 Conv layer to predict its mask.
        self.predictors = nn.ModuleDict({
            task: nn.Conv2d(decoder_channel, n_cls, kernel_size=1)
            for task, n_cls in num_classes.items()
        })


    def _fuse_encoder_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Unify channels, upsample, and concatenate encoder features."""
        target_size = features[0].shape[2:] # H/4, W/4
        
        processed_features = []
        for i, (feature, linear_c) in enumerate(zip(features, self.linear_c)):
            # Unify channel dimension
            f = linear_c(feature)
            # Upsample to match the first feature map's size
            f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            processed_features.append(f)

        # Concatenate along the channel dimension
        fused_features = torch.cat(processed_features, dim=1)
        return fused_features


    # *** FIX 1: Rename parameter `F` to `decoded_features` to avoid shadowing ***
    def _perform_cross_attention(self, query_task: str, decoded_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs attention where a query task attends to context from all other tasks.

        Args:
            query_task (str): The name of the task generating the Query (e.g., 'genus').
            decoded_features (Dict[str, torch.Tensor]): Dictionary of feature maps for all tasks.

        Returns:
            torch.Tensor: The context-enriched feature map for the query task.
        """
        b, _, h, w = decoded_features[query_task].shape
        
        # 1. Generate Query from the primary task
        q = self.to_qkv[query_task][0](decoded_features[query_task]).flatten(2).transpose(1, 2)  # B, HW, C_attn

        # 2. Generate and concatenate context Keys and Values
        context_keys = []
        context_values = []
        for task in self.tasks:
            if task != query_task:
                k_task = self.to_qkv[task][1](decoded_features[task]).flatten(2).transpose(1, 2) # Shape: B, HW, C_attn
                v_task = self.to_qkv[task][2](decoded_features[task]).flatten(2).transpose(1, 2) # Shape: B, HW, C_attn
                context_keys.append(k_task)
                context_values.append(v_task)
        
        k_context = torch.cat(context_keys, dim=1) # Shape: B, (4*HW), C_attn
        v_context = torch.cat(context_values, dim=1) # Shape: B, (4*HW), C_attn

        # 3. Compute Attention - This now correctly uses `torch.nn.functional`
        out = F.scaled_dot_product_attention(q, k_context, v_context)
        
        # Reshape back to B, C_attn, H, W
        out = out.transpose(1, 2).reshape(b, -1, h, w) 
        
        # *** FIX 2 (continued): Project back to decoder_channel for residual connection ***
        out = self.to_out[query_task](out)
        
        return out


    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the decoder.

        Args:
            features (List[torch.Tensor]): A list of 4 feature maps from the encoder backbone.
                                           Expected sizes: [B,C1,H/4,W/4], [B,C2,H/8,W/8], ...

        Returns:
            Dict[str, torch.Tensor]: A dictionary of output logits for each task,
                                     upsampled to 1/4 of the original image size.
        """
        # --- Operation 0: Fuse Encoder Features ---
        fused_features = self._fuse_encoder_features(features)
        
        # --- Operation 1: Multi-Stream Branching and Asymmetric Decoding ---
        # Generate initial feature maps for all 5 tasks
        # *** FIX 1 (continued): Rename variable to `decoded_features` ***
        decoded_features = {task: decoder(fused_features) for task, decoder in self.decoders.items()}

        # --- Operation 2: The Expanded Multi-Task Cross-Attention Module ---
        # Primary tasks query the context provided by all other tasks.
        enriched_features = {}
        for task in self.primary_tasks:
            enriched_features[task] = self._perform_cross_attention(task, decoded_features)

        # --- Operation 3: Final Predictions (Hierarchical Output) ---
        logits = {}
        
        # *** FIX 3: Correctly implement residual connection and prediction ***
        # Primary tasks use enriched features with a residual connection
        for task in self.primary_tasks:
            initial_feature = decoded_features[task]
            enriched_feature = enriched_features[task]
            final_feature = initial_feature + enriched_feature # Residual connection
            logits[task] = self.predictors[task](final_feature)
            
        # Auxiliary tasks predict directly from their lightweight heads
        for task in self.aux_tasks:
            logits[task] = self.predictors[task](decoded_features[task])

        # Upsample all logits before returning
        target_size = features[0].shape[2:]
        for task, logit in logits.items():
            logits[task] = F.interpolate(logit, size=target_size, mode='bilinear', align_corners=False)

        return logits


if __name__ == '__main__':
    # --- Example Usage and Sanity Check ---
    print("--- Running Sanity Check for HierarchicalContextAwareDecoder ---")

    # 1. Define model parameters
    B = 2 # Batch size
    H, W = 512, 512 # Original image size
    
    # Example encoder channels for MiT-B2
    encoder_channels = [64, 128, 320, 512]
    decoder_channel = 256
    attention_dim = 128
    
    num_classes = {
        'genus': 9,
        'health': 4,
        'fish': 2,
        'human_artifacts': 2,
        'substrate': 4
    }

    # 2. Create dummy input features from an encoder
    dummy_features = [
        torch.randn(B, encoder_channels[0], H // 4, W // 4),
        torch.randn(B, encoder_channels[1], H // 8, W // 8),
        torch.randn(B, encoder_channels[2], H // 16, W // 16),
        torch.randn(B, encoder_channels[3], H // 32, W // 32),
    ]
    print(f"Input feature shapes: {[f.shape for f in dummy_features]}")

    # 3. Instantiate the decoder
    decoder = HierarchicalContextAwareDecoder(
        encoder_channels=encoder_channels,
        decoder_channel=decoder_channel,
        num_classes=num_classes,
        attention_dim=attention_dim
    )
    
    # 4. Perform a forward pass
    output_logits = decoder(dummy_features)

    # 5. Check the output
    print("\n--- Output Logits Shapes ---")
    for task_name, logits in output_logits.items():
        print(f"Task '{task_name}': {list(logits.shape)}")
        # Check if the number of classes and spatial dimensions are correct
        assert logits.shape[0] == B
        assert logits.shape[1] == num_classes[task_name]
        assert logits.shape[2] == H // 4
        assert logits.shape[3] == W // 4

    print("\nSanity check passed! The decoder is working as expected.")