import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionalFusionModule(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.w_1 = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )
        self.w_2 = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.01),
        )
        
    def forward(self, x_1, x_2):
        weight_1 = self.w_1(x_1)
        weight_2 = self.w_2(x_2)
        aw = torch.softmax(torch.cat([weight_1, weight_2], dim=1), dim=1)
        y = x_1 * aw[:, 0:1, :, :] + x_2 * aw[:, 1:2, :, :]
        return y.contiguous()


class GatedFusionModule(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
        )
        
    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)
        f_1 = torch.sigmoid(self.conv_1(x)) * x_1
        f_2 = torch.sigmoid(self.conv_2(x)) * x_2
        y = self.fusion_layer(torch.cat([f_1, f_2], dim=1))
        return y.contiguous()


class SharpeningFusionModule(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.fusion_layer_1x1 = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
        )
        self.fusion_layer_3x3 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01),
        )
        
    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)
        w = torch.sigmoid(self.fusion_layer_3x3(self.fusion_layer_1x1(x)))
        f_1 = w * x_1
        f_2 = (1 - w) * x_2
        sum_f = f_1 + f_2
        max_f = torch.max(f_1, f_2)
        mean_threshold = F.adaptive_avg_pool2d(sum_f, output_size=(1, 1)) # [B, C, 1, 1]
        y = torch.where(max_f > mean_threshold, max_f * 2, sum_f)
        return y.contiguous()


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        #~ self.fusion_layers = AttentionalFusionModule(input_channels)
        #~ self.fusion_layers = GatedFusionModule(input_channels)
        self.fusion_layers = SharpeningFusionModule(input_channels)
        
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        semantic_features = data_dict['semantic_features']
        
        #~ features = spatial_features # 20220115, SubMConv3d(7, 16), (x, y, z, i, r, g, b), Car 3d AP: 88.4851, 78.3029, 77.1806
        #~ features = spatial_features # 20220116, SubMConv3d(3, 16), (x, y, z), Car 3d AP: 88.4338, 78.2802, 77.0889
        #~ features = spatial_features # 20220117, SubMConv3d(3, 16), (r, g, b), Car 3d AP: 87.7182, 78.0415, 76.8148
        #~ features = spatial_features # 20220118, SubMConv3d(1, 16), occupancy, Car 3d AP: 87.1878, 77.8787, 76.6703
        
        features = self.fusion_layers(spatial_features, semantic_features)
        
        ups = []
        ret_dict = {}
        x = features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
