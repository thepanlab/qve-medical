# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# Multi-tap feature extraction from Qwen3-VL vision encoder.

# This module provides functionality to extract features from multiple depths
# of the vision transformer, enabling hierarchical feature fusion.
# """

# import torch
# import torch.nn as nn
# from typing import List, Tuple, Optional


# class MultiTapExtractor:
#     """
#     Extract features from multiple transformer blocks in Qwen3-VL vision encoder.
    
#     Usage:
#         extractor = MultiTapExtractor(vision_backbone, tap_layers=[6, 13, 20])
#         tapped_features, mask = extractor(pixel_values, grid_thw)
#     """
    
#     def __init__(self, vision_backbone: nn.Module, tap_layers: List[int]):
#         """
#         Args:
#             vision_backbone: The Qwen3VLVisionModel (from find_vision_backbone)
#             tap_layers: List of layer indices to extract features from (0-indexed)
#                        For 27-layer model, [6, 13, 20] gives early/mid/late features
#         """
#         self.vision_backbone = vision_backbone
#         self.tap_layers = sorted(tap_layers)
        
#         # Storage for hooked features
#         self.tapped_outputs = {}
#         self.hooks = []
        
#         # Validate and register hooks
#         self._validate_and_register_hooks()
    
#     def _validate_and_register_hooks(self):
#         """Find transformer blocks and register forward hooks."""
#         # Find the blocks ModuleList
#         blocks = None
        
#         # Try common paths
#         if hasattr(self.vision_backbone, 'blocks'):
#             blocks = self.vision_backbone.blocks
#         elif hasattr(self.vision_backbone, 'visual') and hasattr(self.vision_backbone.visual, 'blocks'):
#             blocks = self.vision_backbone.visual.blocks
#         elif hasattr(self.vision_backbone, 'encoder') and hasattr(self.vision_backbone.encoder, 'layers'):
#             blocks = self.vision_backbone.encoder.layers
        
#         if blocks is None:
#             raise AttributeError(
#                 "Could not find transformer blocks in vision_backbone. "
#                 "Expected attribute 'blocks', 'visual.blocks', or 'encoder.layers'"
#             )
        
#         num_blocks = len(blocks)
#         print(f"[MultiTapExtractor] Found {num_blocks} transformer blocks")
        
#         # Validate tap layers
#         for layer_idx in self.tap_layers:
#             if layer_idx < 0 or layer_idx >= num_blocks:
#                 raise ValueError(
#                     f"Tap layer {layer_idx} is out of range [0, {num_blocks-1}]"
#                 )
        
#         print(f"[MultiTapExtractor] Tapping layers: {self.tap_layers}")
        
#         # Register hooks
#         for layer_idx in self.tap_layers:
#             block = blocks[layer_idx]
            
#             def make_hook(idx):
#                 """Closure to capture layer_idx properly."""
#                 def hook_fn(module, input, output):
#                     # Store the output
#                     # Output can be a tensor or tuple depending on implementation
#                     if isinstance(output, tuple):
#                         self.tapped_outputs[idx] = output[0]
#                     else:
#                         self.tapped_outputs[idx] = output
#                 return hook_fn
            
#             hook = block.register_forward_hook(make_hook(layer_idx))
#             self.hooks.append(hook)
        
#         print(f"[MultiTapExtractor] Registered {len(self.hooks)} hooks")
    
#     def remove_hooks(self):
#         """Remove all registered hooks (call this when done to avoid memory leaks)."""
#         for hook in self.hooks:
#             hook.remove()
#         self.hooks = []
#         print("[MultiTapExtractor] Removed all hooks")
    
#     def __del__(self):
#         """Cleanup hooks on deletion."""
#         self.remove_hooks()
    
#     def forward(
#         self,
#         pixel_values: torch.Tensor,
#         grid_thw: torch.Tensor,
#         requires_grad: bool = False
#     ) -> Tuple[List[torch.Tensor], torch.Tensor]:
#         """
#         Extract features from multiple depths.
        
#         Args:
#             pixel_values: [B, C, H, W] or flattened [N_total, D] depending on processor
#             grid_thw: [B, 3] grid dimensions (temporal, height, width)
#             requires_grad: Whether to compute gradients through vision backbone
        
#         Returns:
#             tapped_features: List of [B, N_max, D] tensors (one per tap layer)
#             mask: [B, N_max] boolean mask (True = valid token, False = padding)
#         """
#         self.tapped_outputs.clear()
        
#         # Forward pass through vision backbone
#         if requires_grad:
#             output = self.vision_backbone(pixel_values, grid_thw)
#         else:
#             with torch.no_grad():
#                 output = self.vision_backbone(pixel_values, grid_thw)
        
#         # Extract the tapped features
#         if not self.tapped_outputs:
#             raise RuntimeError(
#                 "No features were captured by hooks. "
#                 "This may indicate the vision backbone structure has changed."
#             )
        
#         # Get features in order of tap layers
#         tapped_features_raw = [self.tapped_outputs[idx] for idx in self.tap_layers]
#         # for i, (layer_idx, feat) in enumerate(zip(self.tap_layers, tapped_features_raw)):
#         #     print(f"  Layer {layer_idx}: {feat.shape}")
#         # Convert to [B, N, D] format with proper masking
#         tapped_features, mask = self._process_tapped_features(
#             tapped_features_raw, grid_thw
#         )
        
#         return tapped_features, mask
    
#     def _process_tapped_features(
#         self,
#         tapped_features_raw: List[torch.Tensor],
#         grid_thw: torch.Tensor
#     ) -> Tuple[List[torch.Tensor], torch.Tensor]:
#         """
#         Convert raw hooked features to [B, N_max, D] format with masks.
        
#         Qwen3-VL vision encoder can output features in different formats:
#         - [N_total, D] where N_total = sum of (T*H*W) for all images in batch
#         - [B, N, D] if already batched
        
#         We need to:
#         1. Separate into per-image features based on grid_thw
#         2. Pad to N_max (longest sequence)
#         3. Create masks
#         """
#         batch_size = grid_thw.shape[0]
        
#         # Calculate number of patches per image from grid_thw
#         # grid_thw: [B, 3] where columns are (temporal, height, width)
#         patches_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
#         total_patches = sum(patches_per_image)
        
#         # Process each tap layer
#         processed_features = []
        
#         for tap_idx, raw_feat in enumerate(tapped_features_raw):
#             # raw_feat can be [N_total, D] or [B, N, D]
            
#             if raw_feat.ndim == 2:
#                 # Format: [N_total, D] - need to split by image
#                 N_total, D = raw_feat.shape
                
#                 if N_total != total_patches:
#                     print(f"[WARNING] Tap {self.tap_layers[tap_idx]}: "
#                           f"Expected {total_patches} tokens, got {N_total}")
                
#                 # Split into per-image features
#                 split_features = []
#                 start_idx = 0
#                 for num_patches in patches_per_image:
#                     end_idx = start_idx + num_patches
#                     img_feat = raw_feat[start_idx:end_idx]  # [N_i, D]
#                     split_features.append(img_feat)
#                     start_idx = end_idx
                
#                 # Pad to N_max
#                 N_max = max(patches_per_image)
#                 padded_features = []
#                 for img_feat in split_features:
#                     N_i = img_feat.shape[0]
#                     if N_i < N_max:
#                         padding = torch.zeros(
#                             N_max - N_i, D,
#                             dtype=img_feat.dtype,
#                             device=img_feat.device
#                         )
#                         img_feat = torch.cat([img_feat, padding], dim=0)
#                     padded_features.append(img_feat)
                
#                 # Stack: [B, N_max, D]
#                 batched_feat = torch.stack(padded_features, dim=0)
            
#             elif raw_feat.ndim == 3:
#                 # Format: [B, N, D] - already batched
#                 B, N, D = raw_feat.shape
                
#                 if B != batch_size:
#                     raise ValueError(
#                         f"Batch size mismatch: expected {batch_size}, got {B}"
#                     )
                
#                 # Check if padding is needed
#                 N_max = max(patches_per_image)
#                 if N < N_max:
#                     padding = torch.zeros(
#                         B, N_max - N, D,
#                         dtype=raw_feat.dtype,
#                         device=raw_feat.device
#                     )
#                     batched_feat = torch.cat([raw_feat, padding], dim=1)
#                 elif N > N_max:
#                     # Truncate (shouldn't happen normally)
#                     batched_feat = raw_feat[:, :N_max, :]
#                 else:
#                     batched_feat = raw_feat
            
#             else:
#                 raise ValueError(
#                     f"Unexpected feature tensor shape: {raw_feat.shape}"
#                 )
            
#             processed_features.append(batched_feat)
        
#         # Create mask: [B, N_max]
#         N_max = processed_features[0].shape[1]
#         mask = torch.zeros(batch_size, N_max, dtype=torch.bool, device=grid_thw.device)
#         for b, num_patches in enumerate(patches_per_image):
#             mask[b, :num_patches] = True
        
#         return processed_features, mask
    
#     def __call__(self, pixel_values, grid_thw, requires_grad=False):
#         """Convenience method for forward()."""
#         return self.forward(pixel_values, grid_thw, requires_grad)


# # ============================================================================
# # Testing utilities
# # ============================================================================

# def test_multi_tap_extractor():
#     """Test the multi-tap extractor with dummy data."""
#     print("="*80)
#     print("TESTING MULTI-TAP EXTRACTOR")
#     print("="*80)
    
#     from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
#     from PIL import Image
#     import numpy as np
    
#     # Load model
#     model_id = "Qwen/Qwen3-VL-8B-Instruct"
#     print(f"\nLoading {model_id}...")
    
#     processor = AutoProcessor.from_pretrained(model_id)
#     base_model = Qwen3VLForConditionalGeneration.from_pretrained(
#         model_id,
#         device_map="cpu",
#         torch_dtype=torch.float32,
#     )
    
#     # Find vision backbone (use your existing function)
#     vision_backbone = base_model.visual  # Or use find_vision_backbone()
    
#     print("✓ Model loaded")
    
#     # Create extractor
#     tap_layers = [6, 13, 20]  # Early, mid, late
#     print(f"\nCreating MultiTapExtractor with tap_layers={tap_layers}")
    
#     extractor = MultiTapExtractor(vision_backbone, tap_layers)
    
#     # Create dummy images
#     print("\nCreating dummy images...")
#     dummy_imgs = [
#         Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
#         Image.fromarray(np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)),
#     ]
    
#     # Process
#     print("Processing images...")
#     inputs = processor(
#         images=dummy_imgs,
#         text=["", ""],
#         return_tensors="pt",
#         min_pixels=224*224,
#         max_pixels=224*224,
#     )
    
#     pixel_values = inputs["pixel_values"]
#     grid_thw = inputs.get("image_grid_thw", inputs.get("grid_thw", None))
    
#     print(f"  pixel_values: {pixel_values.shape}")
#     print(f"  grid_thw: {grid_thw}")
    
#     # Extract features
#     print("\nExtracting multi-tap features...")
#     tapped_features, mask = extractor(pixel_values, grid_thw, requires_grad=False)
    
#     print(f"\n✓ Extraction successful!")
#     print(f"  Number of taps: {len(tapped_features)}")
#     print(f"  Mask shape: {mask.shape}")
#     print(f"  Valid tokens per image: {mask.sum(dim=1).tolist()}")
    
#     for i, feat in enumerate(tapped_features):
#         print(f"  Tap {tap_layers[i]}: {feat.shape}")
    
#     # Verify all features have same shape
#     shapes = [f.shape for f in tapped_features]
#     assert all(s == shapes[0] for s in shapes), "Feature shapes don't match!"
    
#     # Verify mask
#     B, N_max = mask.shape
#     assert B == len(dummy_imgs), "Batch size mismatch"
    
#     print("\n✓ All checks passed!")
#     print("="*80)
    
#     # Cleanup
#     extractor.remove_hooks()


# if __name__ == "__main__":
#     test_multi_tap_extractor()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-tap feature extraction from Qwen3-VL vision encoder.

This module provides functionality to extract features from multiple depths
of the vision transformer, enabling hierarchical feature fusion.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class MultiTapExtractor(nn.Module):
    """
    Extract features from multiple transformer blocks in Qwen3-VL vision encoder.
    
    Usage:
        extractor = MultiTapExtractor(vision_backbone, tap_layers=[6, 13, 20])
        tapped_features, mask = extractor(pixel_values, grid_thw)
    """
    
    def __init__(
        self, 
        vision_backbone: nn.Module, 
        tap_layers: List[int],
        project_to_dim: Optional[int] = None,
        use_layernorm: bool = True
    ):
        """
        Args:
            vision_backbone: The Qwen3VLVisionModel (from find_vision_backbone)
            tap_layers: List of layer indices to extract features from (0-indexed)
                       For 27-layer model, [6, 13, 20] gives early/mid/late features
            project_to_dim: If provided, project all taps to this dimension (e.g., 4096)
                           If None, keep native dimension (e.g., 1152)
            use_layernorm: If True, apply LayerNorm after projection
        """
        super().__init__()
        
        self.vision_backbone = vision_backbone
        self.tap_layers = sorted(tap_layers)
        self.project_to_dim = project_to_dim
        
        # Storage for hooked features
        self.tapped_outputs = {}
        self.hooks = []
        
        # Validate and register hooks
        self._validate_and_register_hooks()
        
        # Create projection layers if needed
        if project_to_dim is not None:
            self.projections = nn.ModuleList()
            self.norms = nn.ModuleList() if use_layernorm else None
            
            # We'll initialize these after we know the input dimension
            self._projections_initialized = False
            self._native_dim = None
        else:
            self.projections = None
            self.norms = None
    
    def _validate_and_register_hooks(self):
        """Find transformer blocks and register forward hooks."""
        # Find the blocks ModuleList
        blocks = None
        
        # Try common paths
        if hasattr(self.vision_backbone, 'blocks'):
            blocks = self.vision_backbone.blocks
        elif hasattr(self.vision_backbone, 'visual') and hasattr(self.vision_backbone.visual, 'blocks'):
            blocks = self.vision_backbone.visual.blocks
        elif hasattr(self.vision_backbone, 'encoder') and hasattr(self.vision_backbone.encoder, 'layers'):
            blocks = self.vision_backbone.encoder.layers
        
        if blocks is None:
            raise AttributeError(
                "Could not find transformer blocks in vision_backbone. "
                "Expected attribute 'blocks', 'visual.blocks', or 'encoder.layers'"
            )
        
        num_blocks = len(blocks)
        print(f"[MultiTapExtractor] Found {num_blocks} transformer blocks")
        
        # Validate tap layers
        for layer_idx in self.tap_layers:
            if layer_idx < 0 or layer_idx >= num_blocks:
                raise ValueError(
                    f"Tap layer {layer_idx} is out of range [0, {num_blocks-1}]"
                )
        
        print(f"[MultiTapExtractor] Tapping layers: {self.tap_layers}")
        
        # Register hooks
        for layer_idx in self.tap_layers:
            block = blocks[layer_idx]
            
            def make_hook(idx):
                """Closure to capture layer_idx properly."""
                def hook_fn(module, input, output):
                    # Store the output
                    # Output can be a tensor or tuple depending on implementation
                    if isinstance(output, tuple):
                        self.tapped_outputs[idx] = output[0]
                    else:
                        self.tapped_outputs[idx] = output
                return hook_fn
            
            hook = block.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)
        
        print(f"[MultiTapExtractor] Registered {len(self.hooks)} hooks")
    
    def _initialize_projections(self, native_dim: int):
        """Initialize projection layers once we know the native dimension."""
        if self._projections_initialized:
            return
        
        self._native_dim = native_dim
        
        for _ in self.tap_layers:
            proj = nn.Linear(native_dim, self.project_to_dim)
            self.projections.append(proj)
            
            if self.norms is not None:
                norm = nn.LayerNorm(self.project_to_dim)
                self.norms.append(norm)
        
        self._projections_initialized = True
        
        print(f"[MultiTapExtractor] Initialized {len(self.tap_layers)} projection layers: "
              f"{native_dim} → {self.project_to_dim}")
    
    def remove_hooks(self):
        """Remove all registered hooks (call this when done to avoid memory leaks)."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("[MultiTapExtractor] Removed all hooks")
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        requires_grad: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extract features from multiple depths.
        
        Args:
            pixel_values: [B, C, H, W] or flattened [N_total, D] depending on processor
            grid_thw: [B, 3] grid dimensions (temporal, height, width)
            requires_grad: Whether to compute gradients through vision backbone
        
        Returns:
            tapped_features: List of [B, N_max, D] tensors (one per tap layer)
            mask: [B, N_max] boolean mask (True = valid token, False = padding)
        """
        self.tapped_outputs.clear()
        
        # Forward pass through vision backbone
        if requires_grad:
            output = self.vision_backbone(pixel_values, grid_thw)
        else:
            with torch.no_grad():
                output = self.vision_backbone(pixel_values, grid_thw)
        
        # Extract the tapped features
        if not self.tapped_outputs:
            raise RuntimeError(
                "No features were captured by hooks. "
                "This may indicate the vision backbone structure has changed."
            )
        
        # Get features in order of tap layers
        tapped_features_raw = [self.tapped_outputs[idx] for idx in self.tap_layers]
        
        # Convert to [B, N, D] format with proper masking
        tapped_features, mask = self._process_tapped_features(
            tapped_features_raw, grid_thw
        )
        
        # Apply projections if needed
        if self.projections is not None:
            # Initialize projections if first time
            native_dim = tapped_features[0].shape[-1]
            if not self._projections_initialized:
                self._initialize_projections(native_dim)
                # Move projections to correct device and dtype
                device = tapped_features[0].device
                dtype = tapped_features[0].dtype
                self.projections = self.projections.to(device=device, dtype=dtype)
                if self.norms is not None:
                    self.norms = self.norms.to(device=device, dtype=dtype)
            
            # Project each tap
            projected_features = []
            for i, feat in enumerate(tapped_features):
                proj_feat = self.projections[i](feat)
                if self.norms is not None:
                    proj_feat = self.norms[i](proj_feat)
                projected_features.append(proj_feat)
            
            tapped_features = projected_features
        
        return tapped_features, mask
    
    def _process_tapped_features(
        self,
        tapped_features_raw: List[torch.Tensor],
        grid_thw: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Convert raw hooked features to [B, N_max, D] format with masks.
        
        Qwen3-VL vision encoder can output features in different formats:
        - [N_total, D] where N_total = sum of (T*H*W) for all images in batch
        - [B, N, D] if already batched
        
        We need to:
        1. Separate into per-image features based on grid_thw
        2. Pad to N_max (longest sequence)
        3. Create masks
        """
        batch_size = grid_thw.shape[0]
        
        # Calculate number of patches per image from grid_thw
        # grid_thw: [B, 3] where columns are (temporal, height, width)
        patches_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        total_patches = sum(patches_per_image)
        
        # Process each tap layer
        processed_features = []
        
        for tap_idx, raw_feat in enumerate(tapped_features_raw):
            # raw_feat can be [N_total, D] or [B, N, D]
            
            if raw_feat.ndim == 2:
                # Format: [N_total, D] - need to split by image
                N_total, D = raw_feat.shape
                
                if N_total != total_patches:
                    print(f"[WARNING] Tap {self.tap_layers[tap_idx]}: "
                          f"Expected {total_patches} tokens, got {N_total}")
                
                # Split into per-image features
                split_features = []
                start_idx = 0
                for num_patches in patches_per_image:
                    end_idx = start_idx + num_patches
                    img_feat = raw_feat[start_idx:end_idx]  # [N_i, D]
                    split_features.append(img_feat)
                    start_idx = end_idx
                
                # Pad to N_max
                N_max = max(patches_per_image)
                padded_features = []
                for img_feat in split_features:
                    N_i = img_feat.shape[0]
                    if N_i < N_max:
                        padding = torch.zeros(
                            N_max - N_i, D,
                            dtype=img_feat.dtype,
                            device=img_feat.device
                        )
                        img_feat = torch.cat([img_feat, padding], dim=0)
                    padded_features.append(img_feat)
                
                # Stack: [B, N_max, D]
                batched_feat = torch.stack(padded_features, dim=0)
            
            elif raw_feat.ndim == 3:
                # Format: [B, N, D] - already batched
                B, N, D = raw_feat.shape
                
                if B != batch_size:
                    raise ValueError(
                        f"Batch size mismatch: expected {batch_size}, got {B}"
                    )
                
                # Check if padding is needed
                N_max = max(patches_per_image)
                if N < N_max:
                    padding = torch.zeros(
                        B, N_max - N, D,
                        dtype=raw_feat.dtype,
                        device=raw_feat.device
                    )
                    batched_feat = torch.cat([raw_feat, padding], dim=1)
                elif N > N_max:
                    # Truncate (shouldn't happen normally)
                    batched_feat = raw_feat[:, :N_max, :]
                else:
                    batched_feat = raw_feat
            
            else:
                raise ValueError(
                    f"Unexpected feature tensor shape: {raw_feat.shape}"
                )
            
            processed_features.append(batched_feat)
        
        # Create mask: [B, N_max]
        N_max = processed_features[0].shape[1]
        mask = torch.zeros(batch_size, N_max, dtype=torch.bool, device=grid_thw.device)
        for b, num_patches in enumerate(patches_per_image):
            mask[b, :num_patches] = True
        
        return processed_features, mask
    
    def __call__(self, pixel_values, grid_thw, requires_grad=False):
        """Convenience method for forward()."""
        return self.forward(pixel_values, grid_thw, requires_grad)


# ============================================================================
# Testing utilities
# ============================================================================

def test_multi_tap_extractor():
    """Test the multi-tap extractor with dummy data."""
    print("="*80)
    print("TESTING MULTI-TAP EXTRACTOR")
    print("="*80)
    
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from PIL import Image
    import numpy as np
    
    # Load model
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"\nLoading {model_id}...")
    
    processor = AutoProcessor.from_pretrained(model_id)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    
    # Find vision backbone (use your existing function)
    vision_backbone = base_model.visual  # Or use find_vision_backbone()
    
    print("✓ Model loaded")
    
    # Create extractor
    tap_layers = [6, 13, 20]  # Early, mid, late
    print(f"\nCreating MultiTapExtractor with tap_layers={tap_layers}")
    
    extractor = MultiTapExtractor(vision_backbone, tap_layers)
    
    # Create dummy images
    print("\nCreating dummy images...")
    dummy_imgs = [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
        Image.fromarray(np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8)),
    ]
    
    # Process
    print("Processing images...")
    inputs = processor(
        images=dummy_imgs,
        text=["", ""],
        return_tensors="pt",
        min_pixels=224*224,
        max_pixels=224*224,
    )
    
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs.get("image_grid_thw", inputs.get("grid_thw", None))
    
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  grid_thw: {grid_thw}")
    
    # Extract features
    print("\nExtracting multi-tap features...")
    tapped_features, mask = extractor(pixel_values, grid_thw, requires_grad=False)
    
    print(f"\n✓ Extraction successful!")
    print(f"  Number of taps: {len(tapped_features)}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Valid tokens per image: {mask.sum(dim=1).tolist()}")
    
    for i, feat in enumerate(tapped_features):
        print(f"  Tap {tap_layers[i]}: {feat.shape}")
    
    # Verify all features have same shape
    shapes = [f.shape for f in tapped_features]
    assert all(s == shapes[0] for s in shapes), "Feature shapes don't match!"
    
    # Verify mask
    B, N_max = mask.shape
    assert B == len(dummy_imgs), "Batch size mismatch"
    
    print("\n✓ All checks passed!")
    print("="*80)
    
    # Cleanup
    extractor.remove_hooks()


if __name__ == "__main__":
    test_multi_tap_extractor()