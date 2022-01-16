from functools import partial
import torch.nn as nn

from utils.spconv_utils import spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = post_act_block
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        input_channels_1 = 3 # (x, y, z)
        input_channels_2 = 3 # (r, g, b)
        
        self.conv_input_1 = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels_1, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        self.conv_input_2 = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels_2, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )

        self.conv1_1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.conv1_2 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2_1 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )
        self.conv2_2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3_1 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        self.conv3_2 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4_1 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        self.conv4_2 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        self.conv_out_1 = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=0,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.conv_out_2 = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=0,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        batch_size = batch_dict['batch_size']
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        
        voxel_features_1 = voxel_features[:, 0:3] # consider (x, y, z) in voxel_features
        voxel_features_2 = voxel_features[:, 4:7] # consider (r, g, b) in voxel_features
        
        input_sp_tensor_1 = spconv.SparseConvTensor(
            features=voxel_features_1,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        input_sp_tensor_2 = spconv.SparseConvTensor(
            features=voxel_features_2,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_1 = self.conv_input_1(input_sp_tensor_1)
        x_conv1_1 = self.conv1_1(x_1)
        x_conv2_1 = self.conv2_1(x_conv1_1)
        x_conv3_1 = self.conv3_1(x_conv2_1)
        x_conv4_1 = self.conv4_1(x_conv3_1)
        out_1 = self.conv_out_1(x_conv4_1)
        
        x_2 = self.conv_input_2(input_sp_tensor_2)
        x_conv1_2 = self.conv1_2(x_2)
        x_conv2_2 = self.conv2_2(x_conv1_2)
        x_conv3_2 = self.conv3_2(x_conv2_2)
        x_conv4_2 = self.conv4_2(x_conv3_2)
        out_2 = self.conv_out_2(x_conv4_2)

        batch_dict.update({
            'spatial_encoded_spconv_tensor': out_1,
            'semantic_encoded_spconv_tensor': out_2,
        })

        return batch_dict
