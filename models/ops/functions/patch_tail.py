
if MSDA is None:
    import warnings
    warnings.warn("MultiScaleDeformableAttention C++ extension not found. Using pure PyTorch implementation (slower).")
    
    class MSDeformAttnFunction:
        @staticmethod
        def apply(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
            return ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)
