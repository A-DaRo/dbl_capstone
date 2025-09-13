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

# --- The Main Hierarchical Context-Aware Decoder ---
class HierarchicalContextAwareDecoder(nn.Module):
    """
    Implements the Hierarchical Context-Aware Decoder for Multi-Task Coral Segmentation.

    This decoder takes features from a shared encoder and branches into five streams:
    - 2 Primary Streams (Genus, Health): Full MLP decoders, enriched via cross-attention.
    - 3 Auxiliary Streams (Fish, Human-Artifact, Substrate): Lightweight heads.
    """
    def __init__(self,
                 encoder_channels: List[int],
                 decoder_channel: int,
                 num_classes: Dict[str, int],
                 attention_dim: int = 256):
        """
        Args:
            encoder_channels (List[int]): Channel dimensions from the encoder's stages.
            decoder_channel (int): The unified channel dimension for all streams.
            num_classes (Dict[str, int]): Mapping of task names to their number of classes.
            attention_dim (int): Dimension for the query, key, and value in the attention module.
        """
        super().__init__()
        assert len(encoder_channels) == 4, "Requires features from 4 encoder stages."

        # Task definitions are now hardcoded in the model architecture for clarity
        self.primary_tasks = ['panoptic_shape', 'panoptic_health']
        self.aux_tasks = ['fish', 'human_artifacts', 'substrate']
        self.tasks = self.primary_tasks + self.aux_tasks
        
        self.decoder_channel = decoder_channel

        # Channel Unification MLPs
        self.linear_c = nn.ModuleList([
            MLP(input_dim=c, output_dim=decoder_channel) for c in encoder_channels
        ])

        fused_channels = decoder_channel * 4

        # Asymmetric Decoder Heads
        self.decoders = nn.ModuleDict({
            'panoptic_shape': MLP(fused_channels, decoder_channel),
            'panoptic_health': MLP(fused_channels, decoder_channel),
            'fish': nn.Conv2d(fused_channels, decoder_channel, kernel_size=1),
            'human_artifacts': nn.Conv2d(fused_channels, decoder_channel, kernel_size=1),
            'substrate': nn.Conv2d(fused_channels, decoder_channel, kernel_size=1)
        })

        # Projections for Cross-Attention Module
        self.to_qkv = nn.ModuleDict()
        for task in self.tasks:
            self.to_qkv[task] = nn.ModuleList([
                nn.Conv2d(decoder_channel, attention_dim, 1, bias=False), # Query
                nn.Conv2d(decoder_channel, attention_dim, 1, bias=False), # Key
                nn.Conv2d(decoder_channel, attention_dim, 1, bias=False)  # Value
            ])

        self.context_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
        self.attn_proj = nn.ModuleDict({
            task: MLP(attention_dim, decoder_channel) for task in self.primary_tasks
        })

        self.gating_layers = nn.ModuleDict({
            task: nn.Sequential(
                nn.Conv2d(decoder_channel, 1, kernel_size=1),
                nn.Sigmoid()
            ) for task in self.primary_tasks
        })

        # Final Prediction Layers
        self.predictors = nn.ModuleDict({
            task: nn.Conv2d(decoder_channel, n_cls, kernel_size=1)
            for task, n_cls in num_classes.items()
        })

    def _fuse_encoder_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Unify channels, upsample, and concatenate encoder features."""
        target_size = features[0].shape[2:]
        processed_features = [
            F.interpolate(linear_c(feature), size=target_size, mode='bilinear', align_corners=False)
            for feature, linear_c in zip(features, self.linear_c)
        ]
        return torch.cat(processed_features, dim=1)

    def _perform_cross_attention(self, query_task: str, decoded_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs attention where a query task attends to context from all other tasks."""
        b, _, h, w = decoded_features[query_task].shape
        q = self.to_qkv[query_task][0](decoded_features[query_task]).flatten(2).transpose(1, 2)

        context_keys, context_values = [], []
        for task in self.tasks:
            if task != query_task:
                context_feature = self.context_pool(decoded_features[task])
                k_task = self.to_qkv[task][1](context_feature).flatten(2).transpose(1, 2)
                v_task = self.to_qkv[task][2](context_feature).flatten(2).transpose(1, 2)
                context_keys.append(k_task)
                context_values.append(v_task)
        
        k_context = torch.cat(context_keys, dim=1)
        v_context = torch.cat(context_values, dim=1)
        
        out = F.scaled_dot_product_attention(q, k_context, v_context)
        return out.transpose(1, 2).reshape(b, -1, h, w)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the decoder."""
        fused_features = self._fuse_encoder_features(features)
        decoded_features = {task: decoder(fused_features) for task, decoder in self.decoders.items()}

        enriched_features = {
            task: self._perform_cross_attention(task, decoded_features)
            for task in self.primary_tasks
        }

        logits = {}
        for task in self.primary_tasks:
            f_original = decoded_features[task]
            f_projected_enrichment = self.attn_proj[task](enriched_features[task])
            gate = self.gating_layers[task](f_original)
            final_feature = (gate * f_original) + ((1 - gate) * f_projected_enrichment)
            logits[task] = self.predictors[task](final_feature)

        for task in self.aux_tasks:
            logits[task] = self.predictors[task](decoded_features[task])

        target_size = features[0].shape[2:]
        for task, logit in logits.items():
            if logit.shape[2:] != target_size:
                 logits[task] = F.interpolate(logit, size=target_size, mode='bilinear', align_corners=False)

        return logits