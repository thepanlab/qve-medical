#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token-wise depth fusion for multi-tap feature extraction.

Fuses features from multiple depths of a vision transformer at the token level,
preserving spatial information before pooling.
"""

import torch
import torch.nn as nn
from typing import List


class TokenWiseDepthFusion(nn.Module):
    """
    Fuse multi-tap features at token level before pooling.
    
    Three fusion strategies:
    1. 'scalar': Learn global depth weights (simplest)
    2. 'token_attention': Each token learns its own depth preference (recommended)
    3. 'gated': Feature-conditioned gating (most expressive)
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_taps: int,
        fusion_type: str = "token_attention",
        use_layernorm: bool = True
    ):
        """
        Args:
            hidden_size: Feature dimension (e.g., 4096)
            num_taps: Number of tapped layers (e.g., 3)
            fusion_type: 'scalar', 'token_attention', or 'gated'
            use_layernorm: Apply LayerNorm to each tap before fusion
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_taps = num_taps
        self.fusion_type = fusion_type
        
        # Optional per-tap normalization
        if use_layernorm:
            self.tap_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size) for _ in range(num_taps)
            ])
        else:
            self.tap_norms = None
        
        # Fusion-specific parameters
        if fusion_type == "scalar":
            # Simple learnable scalar weights (one per tap)
            self.depth_weights = nn.Parameter(torch.ones(num_taps))
            
        elif fusion_type == "token_attention":
            # Per-token attention over depths
            # Small MLP: features → scalar score per depth
            self.depth_scorer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 4, 1)
            )
            
        elif fusion_type == "gated":
            # Feature-conditioned gating
            # MLP: concatenated features → gate weights
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_size * num_taps, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_taps)
            )
            
        else:
            raise ValueError(
                f"Unknown fusion_type: {fusion_type}. "
                f"Must be 'scalar', 'token_attention', or 'gated'"
            )
    
    def forward(
        self, 
        tapped_features: List[torch.Tensor], 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multi-tap features at token level.
        
        Args:
            tapped_features: List of [B, N, D] tensors (one per tap)
            mask: [B, N] boolean mask (True = valid token, False = padding)
        
        Returns:
            fused: [B, N, D] fused features
        """
        # Validate inputs
        assert len(tapped_features) == self.num_taps, \
            f"Expected {self.num_taps} taps, got {len(tapped_features)}"
        
        B, N, D = tapped_features[0].shape
        assert D == self.hidden_size, \
            f"Feature dim mismatch: expected {self.hidden_size}, got {D}"
        
        # Ensure fusion module parameters match input dtype
        input_dtype = tapped_features[0].dtype
        if self.tap_norms is not None:
            # Convert LayerNorms to match input dtype
            self.tap_norms = self.tap_norms.to(dtype=input_dtype)
            if hasattr(self, 'depth_scorer'):
                self.depth_scorer = self.depth_scorer.to(dtype=input_dtype)
            if hasattr(self, 'gate_mlp'):
                self.gate_mlp = self.gate_mlp.to(dtype=input_dtype)
        
        # Normalize each tap (optional)
        if self.tap_norms is not None:
            tapped_features = [
                norm(feat) for norm, feat in zip(self.tap_norms, tapped_features)
            ]
        
        # Apply fusion strategy
        if self.fusion_type == "scalar":
            fused = self._scalar_fusion(tapped_features)
            
        elif self.fusion_type == "token_attention":
            fused = self._token_attention_fusion(tapped_features, mask)
            
        elif self.fusion_type == "gated":
            fused = self._gated_fusion(tapped_features)
        
        return fused
    
    def _scalar_fusion(self, tapped_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Scalar fusion: Learn global depth weights.
        
        Same weights applied to all tokens across all images.
        """
        # Softmax over depth weights: [num_taps]
        alpha = torch.softmax(self.depth_weights, dim=0)
        
        # Weighted sum: sum_i alpha[i] * features[i]
        fused = sum(w * feat for w, feat in zip(alpha, tapped_features))
        
        return fused
    
    def _token_attention_fusion(
        self, 
        tapped_features: List[torch.Tensor],
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Token-wise attention: Each token learns its own depth preference.
        
        This allows different tokens to emphasize different depths:
        - Fine texture tokens might prefer early layers
        - Semantic concept tokens might prefer late layers
        """
        B, N, D = tapped_features[0].shape
        
        # Stack: [B, N, num_taps, D]
        stacked = torch.stack(tapped_features, dim=2)
        
        # Compute attention scores for each depth: [B, N, num_taps, 1]
        scores = self.depth_scorer(stacked)
        
        # Softmax over depth dimension: [B, N, num_taps, 1]
        alpha = torch.softmax(scores, dim=2)
        
        # Weighted sum: [B, N, D]
        fused = (alpha * stacked).sum(dim=2)
        
        return fused
    
    def _gated_fusion(self, tapped_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Gated fusion: Gates conditioned on all three depths.
        
        More expressive than scalar or attention - gates can consider
        relationships between depths when deciding how to mix them.
        """
        B, N, D = tapped_features[0].shape
        
        # Concatenate all taps: [B, N, num_taps * D]
        concat = torch.cat(tapped_features, dim=-1)
        
        # Compute gates: [B, N, num_taps]
        gates = torch.sigmoid(self.gate_mlp(concat))
        
        # Stack features: [B, N, num_taps, D]
        stacked = torch.stack(tapped_features, dim=2)
        
        # Apply gates: [B, N, num_taps, 1] * [B, N, num_taps, D]
        gates = gates.unsqueeze(-1)  # [B, N, num_taps, 1]
        
        # Weighted sum: [B, N, D]
        fused = (gates * stacked).sum(dim=2)
        
        return fused


class PoolingLayer(nn.Module):
    """
    Various pooling strategies for aggregating token features.
    
    Takes [B, N, D] token features and produces [B, D] image features.
    """
    
    def __init__(
        self,
        pooling_type: str = "mean",
        hidden_size: int = None,
        num_queries: int = 1
    ):
        """
        Args:
            pooling_type: 'mean', 'max', 'attention', or 'cls'
            hidden_size: Required for 'attention' pooling
            num_queries: Number of learnable queries for 'attention' pooling
        """
        super().__init__()
        
        self.pooling_type = pooling_type
        
        if pooling_type == "attention":
            if hidden_size is None:
                raise ValueError("hidden_size required for attention pooling")
            
            # Learnable query vectors
            self.queries = nn.Parameter(torch.randn(num_queries, hidden_size))
            nn.init.xavier_uniform_(self.queries)
            
            self.num_queries = num_queries
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool tokens to image-level features.
        
        Args:
            tokens: [B, N, D] token features
            mask: [B, N] boolean mask (True = valid)
        
        Returns:
            pooled: [B, D] if num_queries=1, else [B, num_queries*D]
        """
        if self.pooling_type == "mean":
            return self._mean_pooling(tokens, mask)
        elif self.pooling_type == "max":
            return self._max_pooling(tokens, mask)
        elif self.pooling_type == "attention":
            return self._attention_pooling(tokens, mask)
        elif self.pooling_type == "cls":
            return self._cls_pooling(tokens, mask)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
    
    def _mean_pooling(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over valid tokens."""
        # Expand mask: [B, N, 1]
        mask_expanded = mask.unsqueeze(-1).float()
        
        # Masked sum
        sum_tokens = (tokens * mask_expanded).sum(dim=1)
        
        # Divide by number of valid tokens
        num_valid = mask_expanded.sum(dim=1).clamp(min=1.0)
        
        return sum_tokens / num_valid
    
    def _max_pooling(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Max pooling over valid tokens."""
        # Set padding tokens to -inf before max
        tokens_masked = tokens.clone()
        tokens_masked[~mask] = float('-inf')
        
        pooled = tokens_masked.max(dim=1)[0]
        
        return pooled
    
    def _attention_pooling(
        self, 
        tokens: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Attention pooling with learnable queries.
        
        Similar to cross-attention, but queries are learned (not from input).
        """
        B, N, D = tokens.shape
        
        # Ensure queries match input dtype
        if self.queries.dtype != tokens.dtype:
            self.queries.data = self.queries.data.to(dtype=tokens.dtype)
        
        # Expand queries for batch: [B, num_queries, D]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        
        # Compute attention: [B, num_queries, N]
        scores = torch.bmm(queries, tokens.transpose(1, 2)) / (D ** 0.5)
        
        # Mask padding tokens
        mask_expanded = mask.unsqueeze(1)  # [B, 1, N]
        scores = scores.masked_fill(~mask_expanded, -1e9)
        
        # Attention weights: [B, num_queries, N]
        attn = torch.softmax(scores, dim=-1)
        
        # Aggregate: [B, num_queries, D]
        pooled = torch.bmm(attn, tokens)
        
        # Flatten if multiple queries: [B, num_queries*D]
        if self.num_queries > 1:
            pooled = pooled.reshape(B, -1)
        else:
            pooled = pooled.squeeze(1)
        
        return pooled
    
    def _cls_pooling(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Use first token (CLS token style)."""
        return tokens[:, 0, :]


# =============================================================================
# Combined fusion + pooling module
# =============================================================================

class FusionAndPooling(nn.Module):
    """
    Combined depth fusion and pooling for multi-tap features.
    
    This is the complete "readout" module that:
    1. Fuses multi-tap features at token level
    2. Pools to image-level features
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_taps: int,
        fusion_type: str = "token_attention",
        pooling_type: str = "mean",
        use_layernorm: bool = True
    ):
        """
        Args:
            hidden_size: Feature dimension (e.g., 4096)
            num_taps: Number of tapped layers (e.g., 3)
            fusion_type: 'scalar', 'token_attention', or 'gated'
            pooling_type: 'mean', 'max', 'attention', or 'cls'
            use_layernorm: Apply LayerNorm before fusion
        """
        super().__init__()
        
        self.fusion = TokenWiseDepthFusion(
            hidden_size=hidden_size,
            num_taps=num_taps,
            fusion_type=fusion_type,
            use_layernorm=use_layernorm
        )
        
        self.pooling = PoolingLayer(
            pooling_type=pooling_type,
            hidden_size=hidden_size,
            num_queries=1
        )
    
    def forward(
        self,
        tapped_features: List[torch.Tensor],
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse and pool multi-tap features.
        
        Args:
            tapped_features: List of [B, N, D] tensors
            mask: [B, N] boolean mask
        
        Returns:
            features: [B, D] image-level features
        """
        # Fuse at token level: [B, N, D]
        fused = self.fusion(tapped_features, mask)
        
        # Pool to image level: [B, D]
        pooled = self.pooling(fused, mask)
        
        return pooled


# =============================================================================
# Testing utilities
# =============================================================================

def test_fusion_module():
    """Test the fusion module with dummy data."""
    print("="*80)
    print("TESTING DEPTH FUSION MODULE")
    print("="*80)
    
    B, N, D = 4, 256, 4096
    num_taps = 3
    
    # Create dummy multi-tap features
    tapped_features = [
        torch.randn(B, N, D) for _ in range(num_taps)
    ]
    
    # Create dummy mask (some padding)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, 200:] = False  # First image has padding
    mask[2, 220:] = False  # Third image has padding
    
    print(f"\nInput:")
    print(f"  Tapped features: {num_taps} × {tapped_features[0].shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Valid tokens per image: {mask.sum(dim=1).tolist()}")
    
    # Test each fusion type
    for fusion_type in ["scalar", "token_attention", "gated"]:
        print(f"\n{'─'*80}")
        print(f"Testing fusion_type='{fusion_type}'")
        print(f"{'─'*80}")
        
        fusion = TokenWiseDepthFusion(
            hidden_size=D,
            num_taps=num_taps,
            fusion_type=fusion_type,
            use_layernorm=True
        )
        
        # Forward pass
        fused = fusion(tapped_features, mask)
        
        print(f"  Output shape: {fused.shape}")
        print(f"  Expected: [{B}, {N}, {D}]")
        assert fused.shape == (B, N, D), "Shape mismatch!"
        
        # Count parameters
        num_params = sum(p.numel() for p in fusion.parameters())
        print(f"  Parameters: {num_params:,}")
        
        print(f"  ✓ Passed")
    
    # Test pooling
    print(f"\n{'─'*80}")
    print(f"Testing pooling methods")
    print(f"{'─'*80}")
    
    for pooling_type in ["mean", "max", "attention"]:
        pooling = PoolingLayer(
            pooling_type=pooling_type,
            hidden_size=D,
            num_queries=1
        )
        
        tokens = torch.randn(B, N, D)
        pooled = pooling(tokens, mask)
        
        print(f"  {pooling_type:12s}: {tokens.shape} → {pooled.shape}")
        assert pooled.shape == (B, D), f"Shape mismatch for {pooling_type}!"
    
    # Test combined module
    print(f"\n{'─'*80}")
    print(f"Testing combined FusionAndPooling")
    print(f"{'─'*80}")
    
    combined = FusionAndPooling(
        hidden_size=D,
        num_taps=num_taps,
        fusion_type="token_attention",
        pooling_type="mean"
    )
    
    output = combined(tapped_features, mask)
    print(f"  Input: {num_taps} × [{B}, {N}, {D}]")
    print(f"  Output: {output.shape}")
    print(f"  Expected: [{B}, {D}]")
    assert output.shape == (B, D), "Shape mismatch!"
    
    print(f"\n✓ All tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_fusion_module()