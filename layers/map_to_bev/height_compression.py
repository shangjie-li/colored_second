import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:
                semantic_features:

        """
        spatial_encoded_spconv_tensor = batch_dict['spatial_encoded_spconv_tensor']
        semantic_encoded_spconv_tensor = batch_dict['semantic_encoded_spconv_tensor']
        
        spatial_features = spatial_encoded_spconv_tensor.dense()
        semantic_features = semantic_encoded_spconv_tensor.dense()
        
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        
        N, C, D, H, W = semantic_features.shape
        semantic_features = semantic_features.view(N, C * D, H, W)
        
        batch_dict['spatial_features'] = spatial_features
        batch_dict['semantic_features'] = semantic_features
        return batch_dict
