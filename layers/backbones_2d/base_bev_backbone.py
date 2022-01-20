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
        #~ self.fusion_layers = SharpeningFusionModule(input_channels)
        #~ self.fusion_layers = GatedFusionModule(input_channels)
        
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks_1 = nn.ModuleList()
        self.blocks_2 = nn.ModuleList()
        self.deblocks_1 = nn.ModuleList()
        self.deblocks_2 = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers_1 = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            cur_layers_2 = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers_1.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
                cur_layers_2.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks_1.append(nn.Sequential(*cur_layers_1))
            self.blocks_2.append(nn.Sequential(*cur_layers_2))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks_1.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_2.append(nn.Sequential(
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
                    self.deblocks_1.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_2.append(nn.Sequential(
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
            self.deblocks_1.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            self.deblocks_2.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        
        self.fusion_layers = AttentionalFusionModule(c_in)
        #~ self.fusion_layers = SharpeningFusionModule(c_in)
        #~ self.fusion_layers = GatedFusionModule(c_in)

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
        
        #~ features = self.fusion_layers(spatial_features, semantic_features) # 20220119, (x, y, z) + (r, g, b), AttentionalFusionModule after 3d backbone, 42ms, Car 3d AP: 88.1797, 78.1762, 76.9522
        #~ features = self.fusion_layers(spatial_features, semantic_features) # 20220120, (x, y, z) + (r, g, b), SharpeningFusionModule after 3d backbone, 49ms, Car 3d AP: 87.5043, 77.6660, 76.7594
        #~ features = self.fusion_layers(spatial_features, semantic_features) # 20220121, (x, y, z) + (r, g, b), GatedFusionModule after 3d backbone, 59ms, Car 3d AP: 88.4815, 78.3277, 77.0688
        #~ features = self.fusion_layers(spatial_features, semantic_features) # 20220122, (x, y, z, i) + (r, g, b), AttentionalFusionModule after 3d backbone, 44ms, Car 3d AP: 88.1691, 77.8451, 76.5505
        
        ups = []
        x = spatial_features
        for i in range(len(self.blocks_1)):
            x = self.blocks_1[i](x)
            if len(self.deblocks_1) > 0:
                ups.append(self.deblocks_1[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks_1) > len(self.blocks_1):
            x = self.deblocks_1[-1](x)
        final_spatial_features = x
        
        ups = []
        x = semantic_features
        for i in range(len(self.blocks_2)):
            x = self.blocks_2[i](x)
            if len(self.deblocks_2) > 0:
                ups.append(self.deblocks_2[i](x))
            else:
                ups.append(x)
        
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        
        if len(self.deblocks_2) > len(self.blocks_2):
            x = self.deblocks_2[-1](x)
        final_semantic_features = x
        
        final_features = self.fusion_layers(final_spatial_features, final_semantic_features) # 20220123, (x, y, z) + (r, g, b), AttentionalFusionModule after 2d backbone, 62ms, Car 3d AP: 88.7057, 78.4848, 77.2344
        #~ final_features = self.fusion_layers(final_spatial_features, final_semantic_features) # 20220124, (x, y, z) + (r, g, b), SharpeningFusionModule after 2d backbone, 83ms, Car 3d AP: 88.3249, 78.2478, 77.1672
        #~ final_features = self.fusion_layers(final_spatial_features, final_semantic_features) # 20220125, (x, y, z) + (r, g, b), GatedFusionModule after 2d backbone, 135ms, Car 3d AP: 88.2408, 78.4253, 77.4121
        
        data_dict['spatial_features_2d'] = final_features

        return data_dict
