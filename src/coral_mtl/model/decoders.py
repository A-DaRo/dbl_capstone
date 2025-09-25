import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

# --- Helper MLP Block ---
class MLP(nn.Module):
    """A simple MLP block for creating decoders using Conv2d layers."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(output_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class SegFormerMLPDecoder(nn.Module):
    """
    The standard All-MLP decoder from the SegFormer paper.

    This decoder takes a list of multi-scale features from the encoder,
    unifies their channel dimensions, upsamples them to a common resolution,
    and fuses them into a single feature map.
    """
    def __init__(self,
                 encoder_channels: List[int],
                 decoder_channel: int,
                 dropout_prob: float = 0.1):
        """
        Args:
            encoder_channels (List[int]): A list of channel dimensions from the
                four stages of the MiT encoder.
            decoder_channel (int): The unified channel dimension for the decoder.
            dropout_prob (float): The probability for the dropout layer.
        """
        super().__init__()
        assert len(encoder_channels) == 4, "Requires features from 4 encoder stages."

        # --- Step 1: Channel Unification MLPs ---
        # Create a separate MLP for each encoder feature map to project it
        # to the common decoder_channel dimension.
        self.linear_c = nn.ModuleList([
            MLP(input_dim=c, output_dim=decoder_channel) for c in encoder_channels
        ])

        # --- Step 2: Feature Fusion MLP ---
        # This MLP takes the concatenated features (4 * decoder_channel) and
        # fuses them back down to a single decoder_channel dimension.
        self.linear_fuse = MLP(
            input_dim=decoder_channel * 4,
            output_dim=decoder_channel
        )

        # --- Step 3: Dropout Layer ---
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Defines the forward pass for the decoder.

        Args:
            features (List[torch.Tensor]): A list of feature maps from the encoder.
                                           The new smp encoder provides 6, so we take the last 4.

        Returns:
            torch.Tensor: The final fused feature map of shape (B, C_decoder, H/4, W/4).
        """
        # The new smp encoder provides 6 features, the decoder expects the last 4
        features = features[-4:]
        
        # The target spatial size is that of the first, largest feature map
        target_size = features[0].shape[-2:]
        
        # Process each feature map: unify channels and upsample
        processed_features = []
        for i, feature in enumerate(features):
            # Project to common channel dimension
            proj_feature = self.linear_c[i](feature)
            
            # Upsample to the target size (H/4, W/4)
            upsampled_feature = F.interpolate(
                proj_feature,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            processed_features.append(upsampled_feature)

        # Concatenate all features along the channel dimension
        fused_features = torch.cat(processed_features, dim=1)
        
        # Fuse the concatenated features and apply dropout
        fused_features = self.linear_fuse(fused_features)
        fused_features = self.dropout(fused_features)

        return fused_features


# --- The Main Hierarchical Context-Aware Decoder ---
class HierarchicalContextAwareDecoder(nn.Module):
    """
    Implements the Hierarchical Context-Aware Decoder for Multi-Task Coral Segmentation.

    This decoder takes features from a shared encoder and branches into multiple streams.
    Tasks designated as 'primary' receive full MLP decoders and are enriched via
    cross-attention. Tasks designated as 'auxiliary' receive lightweight heads.
    """
    def __init__(self,
                 encoder_channels: List[int],
                 decoder_channel: int,
                 num_classes: Dict[str, int],
                 primary_tasks: List[str],
                 aux_tasks: List[str],
                 attention_dim: int = 256):
        """
        Args:
            encoder_channels (List[int]): Channel dimensions from the encoder's stages.
            decoder_channel (int): The unified channel dimension for all streams.
            num_classes (Dict[str, int]): Mapping of task names to their number of classes.
            primary_tasks (List[str]): List of task names to be treated as primary.
            aux_tasks (List[str]): List of task names to be treated as auxiliary.
            attention_dim (int): Dimension for the query, key, and value in the attention module.
        """
        super().__init__()
        assert len(encoder_channels) == 4, "Requires features from 4 encoder stages."
        
        # --- Robustness Checks for Task Configuration ---
        all_defined_tasks = set(primary_tasks) | set(aux_tasks)
        assert not (set(primary_tasks) & set(aux_tasks)), "A task cannot be both primary and auxiliary."
        for task in all_defined_tasks:
            assert task in num_classes, f"Task '{task}' is defined in primary/aux_tasks but not in num_classes."

        self.primary_tasks = primary_tasks
        self.aux_tasks = aux_tasks
        self.tasks = primary_tasks + aux_tasks

        self.decoder_channel = decoder_channel

        # Channel Unification MLPs
        self.linear_c = nn.ModuleList([
            MLP(input_dim=c, output_dim=decoder_channel) for c in encoder_channels
        ])

        fused_channels = decoder_channel * 4

        # Dynamic Decoder Heads - create decoders for all requested tasks
        self.decoders = nn.ModuleDict()
        for task in self.tasks:
            # Use full MLP for primary tasks, lightweight Conv2d for auxiliary tasks
            if task in primary_tasks:
                self.decoders[task] = MLP(fused_channels, decoder_channel)
            else:
                self.decoders[task] = nn.Conv2d(fused_channels, decoder_channel, kernel_size=1)

        # Projections for Cross-Attention Module
        self.to_qkv = nn.ModuleDict()
        for task in self.tasks:
            self.to_qkv[task] = nn.ModuleList([
                nn.Conv2d(decoder_channel, attention_dim, 1, bias=False), # Query
                nn.Conv2d(decoder_channel, attention_dim, 1, bias=False), # Key
                nn.Conv2d(decoder_channel, attention_dim, 1, bias=False)  # Value
            ])

        self.context_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
        # Only create attention and gating layers for primary tasks
        self.attn_proj = nn.ModuleDict()
        self.gating_layers = nn.ModuleDict()
        for task in self.primary_tasks:
            self.attn_proj[task] = MLP(attention_dim, decoder_channel)
            self.gating_layers[task] = nn.Sequential(
                nn.Conv2d(decoder_channel, 1, kernel_size=1),
                nn.Sigmoid()
            )

        # Final Prediction Layers
        self.predictors = nn.ModuleDict({
            task: nn.Conv2d(decoder_channel, n_cls, kernel_size=1)
            for task, n_cls in num_classes.items() if task in self.tasks
        })

    def _fuse_encoder_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Unify channels, upsample, and concatenate encoder features."""
        # The new smp encoder provides 6 features, the decoder expects the last 4
        features = features[-4:]
        target_size = features[0].shape[2:]
        processed_features = [
            F.interpolate(linear_c(feature), size=target_size, mode='bilinear', align_corners=False)
            for feature, linear_c in zip(features, self.linear_c)
        ]
        return torch.cat(processed_features, dim=1)

    def _perform_cross_attention(self, query_task: str, decoded_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs attention where a query task attends to context from all other tasks."""
        b, _, h, w = decoded_features[query_task].shape
        
        query_feature = self.context_pool(decoded_features[query_task])
        q = self.to_qkv[query_task][0](query_feature).flatten(2).transpose(1, 2)

        context_keys, context_values = [], []
        for task in self.tasks:
            if task != query_task:
                context_feature = self.context_pool(decoded_features[task])
                k_task = self.to_qkv[task][1](context_feature).flatten(2).transpose(1, 2)
                v_task = self.to_qkv[task][2](context_feature).flatten(2).transpose(1, 2)
                context_keys.append(k_task)
                context_values.append(v_task)
        
        if not context_keys:
            # If there's no context, return a zero tensor that requires a gradient.
            # This ensures that the computation graph remains connected even in edge cases
            # (e.g., a model with only one task), allowing gradients to flow.
            attention_dim = self.to_qkv[query_task][0].out_channels
            pooled_h, pooled_w = h // 4, w // 4
            return torch.zeros(b, attention_dim, pooled_h, pooled_w, device=q.device, requires_grad=True)
        
        k_context = torch.cat(context_keys, dim=1)
        v_context = torch.cat(context_values, dim=1)
        
        out = F.scaled_dot_product_attention(q, k_context, v_context)
        pooled_h, pooled_w = h // 4, w // 4
        return out.transpose(1, 2).reshape(b, -1, pooled_h, pooled_w)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the decoder."""
        fused_features = self._fuse_encoder_features(features)
        decoded_features = {task: decoder(fused_features) for task, decoder in self.decoders.items()}

        enriched_features = {
            task: self._perform_cross_attention(task, decoded_features)
            for task in self.primary_tasks
        }

        logits = {}
        # Process primary tasks with attention
        for task in self.primary_tasks:
            f_original = decoded_features[task]
            f_enriched = enriched_features[task]
            
            original_h, original_w = f_original.shape[2], f_original.shape[3]
            f_enriched = F.interpolate(f_enriched, size=(original_h, original_w), mode='bilinear', align_corners=False)
            
            f_projected_enrichment = self.attn_proj[task](f_enriched)
            gate = self.gating_layers[task](f_original)
            final_feature = (gate * f_original) + ((1 - gate) * f_projected_enrichment)
            logits[task] = self.predictors[task](final_feature)

        # Process auxiliary tasks without attention
        for task in self.aux_tasks:
            logits[task] = self.predictors[task](decoded_features[task])

        target_size = features[0].shape[2:]
        for task, logit in logits.items():
            if logit.shape[2:] != target_size:
                 logits[task] = F.interpolate(logit, size=target_size, mode='bilinear', align_corners=False)

        return logits