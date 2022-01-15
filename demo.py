import argparse
import glob
from pathlib import Path
import time
import os

import open3d
from utils import open3d_vis_utils as V

import numpy as np
import torch

from data import cfg, cfg_from_yaml_file
from data import KittiDataset
from second_net import build_network, load_data_to_gpu
from utils import common_utils


class DemoDataset(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training=True, data_path=None, logger=None, ext='.bin'):
        """
        Args:
            dataset_cfg:
            class_names:
            training:
            data_path:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, logger=logger
        )
        self.data_path = data_path
        self.ext = ext
        file_list = glob.glob(str(data_path / f'*{self.ext}')) if self.data_path.is_dir() else [self.data_path]
        file_list.sort()
        self.sample_file_list = file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            sample_idx = os.path.basename(self.sample_file_list[index])[:-4]
            colored_points = self.get_colored_points_in_fov(sample_idx)
        else:
            raise NotImplementedError

        input_dict = {
            'points': colored_points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/config.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='data/kitti/training/velodyne/000008.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='weights/second_7862.pth',
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of SECOND-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        data_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    
    print()
    print('<<< model >>>')
    print(model)
    print()
    
    """
    <<< model >>>
    SECONDNet(
      (vfe): MeanVFE()
      (backbone_3d): VoxelBackBone8x(
        (conv_input): SparseSequential(
          (0): SubMConv3d(7, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
          (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (conv1): SparseSequential(
          (0): SparseSequential(
            (0): SubMConv3d(16, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (conv2): SparseSequential(
          (0): SparseSequential(
            (0): SparseConv3d(16, 32, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (1): SparseSequential(
            (0): SubMConv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (2): SparseSequential(
            (0): SubMConv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (conv3): SparseSequential(
          (0): SparseSequential(
            (0): SparseConv3d(32, 64, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (1): SparseSequential(
            (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (2): SparseSequential(
            (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (conv4): SparseSequential(
          (0): SparseSequential(
            (0): SparseConv3d(64, 64, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[0, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (1): SparseSequential(
            (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (2): SparseSequential(
            (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (conv_out): SparseSequential(
          (0): SparseConv3d(64, 128, kernel_size=[3, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
          (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (map_to_bev_module): HeightCompression()
      (backbone_2d): BaseBEVBackbone(
        (blocks): ModuleList(
          (0): Sequential(
            (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
            (1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
            (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (3): ReLU()
            (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (5): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (6): ReLU()
            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (8): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (9): ReLU()
            (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (11): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (12): ReLU()
            (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (14): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (15): ReLU()
            (16): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (17): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (18): ReLU()
          )
          (1): Sequential(
            (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
            (1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
            (2): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (3): ReLU()
            (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (5): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (6): ReLU()
            (7): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (8): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (9): ReLU()
            (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (11): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (12): ReLU()
            (13): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (14): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (15): ReLU()
            (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (17): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (18): ReLU()
          )
        )
        (deblocks): ModuleList(
          (0): Sequential(
            (0): ConvTranspose2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (1): Sequential(
            (0): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
      )
      (dense_head): AnchorHeadSingle(
        (cls_loss_func): SigmoidFocalClassificationLoss()
        (reg_loss_func): WeightedSmoothL1Loss()
        (dir_loss_func): WeightedCrossEntropyLoss()
        (conv_cls): Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))
        (conv_box): Conv2d(512, 42, kernel_size=(1, 1), stride=(1, 1))
        (conv_dir_cls): Conv2d(512, 12, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    """
    
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            
            print()
            print('<<< data_dict >>>')
            for key, val in data_dict.items():
                if isinstance(val, np.ndarray):
                    print(key, type(val), val.shape)
                    print(val)
                else:
                    print(key, type(val))
                    print(val)
            print()
            
            """
            <<< data_dict >>>
            points <class 'numpy.ndarray'> (17093, 8)
            [[ 0.00000000e+00  2.15540009e+01  2.80000009e-02 ...  2.11764708e-01
               2.90196091e-01  1.25490203e-01]
             [ 0.00000000e+00  2.12399998e+01  9.39999968e-02 ...  1.17647061e-02
               1.01960786e-01  1.33333340e-01]
             [ 0.00000000e+00  2.10559998e+01  1.58999994e-01 ...  2.74509817e-01
               2.35294119e-01  1.37254909e-01]
             ...
             [ 0.00000000e+00  6.31500006e+00 -3.09999995e-02 ...  8.11764717e-01
               7.88235307e-01  6.39215708e-01]
             [ 0.00000000e+00  6.30900002e+00 -2.09999997e-02 ...  8.94117653e-01
               7.45098054e-01  6.58823550e-01]
             [ 0.00000000e+00  6.31099987e+00 -1.00000005e-03 ...  8.11764717e-01
               7.49019623e-01  8.19607854e-01]]
            frame_id <class 'numpy.ndarray'> (1,)
            [0]
            voxels <class 'numpy.ndarray'> (13081, 5, 7)
            [[[ 2.15540009e+01  2.80000009e-02  9.38000023e-01 ...  2.11764708e-01
                2.90196091e-01  1.25490203e-01]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]
            
             [[ 2.12399998e+01  9.39999968e-02  9.26999986e-01 ...  1.17647061e-02
                1.01960786e-01  1.33333340e-01]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]
            
             [[ 2.10559998e+01  1.58999994e-01  9.21000004e-01 ...  2.74509817e-01
                2.35294119e-01  1.37254909e-01]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]
            
             ...
            
             [[ 6.32299995e+00 -1.29999995e-01 -1.65100002e+00 ...  8.27450991e-01
                7.60784328e-01  6.74509823e-01]
              [ 6.32600021e+00 -1.11000001e-01 -1.65199995e+00 ...  7.56862760e-01
                7.41176486e-01  6.11764729e-01]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]
            
             [[ 6.32600021e+00 -9.09999982e-02 -1.65199995e+00 ...  8.11764717e-01
                8.62745106e-01  6.94117665e-01]
              [ 6.31300020e+00 -7.10000023e-02 -1.64800000e+00 ...  7.60784328e-01
                7.37254918e-01  7.13725507e-01]
              [ 6.31500006e+00 -5.09999990e-02 -1.64900005e+00 ...  8.50980401e-01
                7.60784328e-01  7.88235307e-01]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]
            
             [[ 6.31500006e+00 -3.09999995e-02 -1.64900005e+00 ...  8.11764717e-01
                7.88235307e-01  6.39215708e-01]
              [ 6.30900002e+00 -2.09999997e-02 -1.64699996e+00 ...  8.94117653e-01
                7.45098054e-01  6.58823550e-01]
              [ 6.31099987e+00 -1.00000005e-03 -1.64800000e+00 ...  8.11764717e-01
                7.49019623e-01  8.19607854e-01]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]
              [ 0.00000000e+00  0.00000000e+00  0.00000000e+00 ...  0.00000000e+00
                0.00000000e+00  0.00000000e+00]]]
            voxel_coords <class 'numpy.ndarray'> (13081, 4)
            [[  0  39 800 431]
             [  0  39 801 424]
             [  0  39 803 421]
             ...
             [  0  13 797 126]
             [  0  13 798 126]
             [  0  13 799 126]]
            voxel_num_points <class 'numpy.ndarray'> (13081,)
            [1 1 1 ... 2 3 3]
            batch_size <class 'int'>
            1
            """
            
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict) # 0
            
            time_start = time.time()
            pred_dicts, _ = model.forward(data_dict) # 1
            pred_dicts, _ = model.forward(data_dict) # 2
            pred_dicts, _ = model.forward(data_dict) # 3
            pred_dicts, _ = model.forward(data_dict) # 4
            pred_dicts, _ = model.forward(data_dict) # 5
            pred_dicts, _ = model.forward(data_dict) # 6
            pred_dicts, _ = model.forward(data_dict) # 7
            pred_dicts, _ = model.forward(data_dict) # 8
            pred_dicts, _ = model.forward(data_dict) # 9
            pred_dicts, _ = model.forward(data_dict) # 10
            time_end = time.time()
            
            print()
            print('<<< pred_dicts[0] >>>') # It seems that there is only one element in the list of pred_dicts.
            for key, val in pred_dicts[0].items():
                try:
                    print(key, type(val), val.shape)
                    print(val)
                except:
                    print(key, type(val))
                    print(val)
            print()
            
            """
            <<< pred_dicts[0] >>>
            pred_boxes <class 'torch.Tensor'> torch.Size([21, 7])
            tensor([[ 14.7034,  -1.0156,  -0.7908,   3.7183,   1.5942,   1.5162,   5.9720],
                    [  8.1039,   1.2246,  -0.8014,   3.6681,   1.5736,   1.5927,   2.8528],
                    [  6.4204,  -3.8541,  -1.0380,   3.1251,   1.4750,   1.4339,   5.9508],
                    [ 33.5922,  -7.0622,  -0.4059,   4.2320,   1.7478,   1.7574,   2.8651],
                    [  3.7884,   2.7339,  -0.8443,   3.5369,   1.5355,   1.5042,   5.9825],
                    [ 25.0172, -10.3138,  -0.9609,   3.8812,   1.6154,   1.4384,   5.8650],
                    [ 20.3694,  -8.5029,  -0.9205,   2.7097,   1.4918,   1.5109,   5.9105],
                    [ 55.5053, -20.2227,  -0.5251,   4.2166,   1.6699,   1.5461,   2.8200],
                    [ 30.5321,  -3.7804,  -0.4234,   1.9425,   0.5567,   1.7145,   6.1368],
                    [ 40.9832,  -9.7833,  -0.6158,   3.7699,   1.5981,   1.5368,   5.9600],
                    [ 28.6852,  -1.7001,  -0.3665,   3.6220,   1.5295,   1.5430,   4.4034],
                    [ 37.1467,  -6.1300,  -0.4432,   1.7365,   0.3998,   1.6863,   5.9863],
                    [ 53.6708, -16.2549,  -0.3658,   1.7359,   0.5568,   1.7327,   3.1471],
                    [ 34.1015,  -4.9480,  -0.4008,   0.6967,   0.6611,   1.8126,   6.1350],
                    [ 52.7796, -21.9016,  -0.4704,   3.9643,   1.5561,   1.6040,   2.9217],
                    [ 37.0847, -16.6196,  -0.7037,   1.6016,   0.6065,   1.6433,   2.6428],
                    [ 29.5497, -13.8661,  -0.8787,   1.8443,   0.4905,   1.7013,   2.7248],
                    [ 40.5267,  -7.1324,  -0.3581,   0.7264,   0.6181,   1.8160,   6.1470],
                    [ 33.6677, -15.3931,  -0.5603,   1.7521,   0.4569,   1.7138,   2.8158],
                    [ 12.8258,   5.0398,  -0.5827,   0.5218,   0.4688,   1.6287,   4.0001],
                    [ 18.6303,   0.2634,  -0.7195,   0.4834,   0.5678,   1.6942,   4.2814]],
                   device='cuda:0')
            pred_scores <class 'torch.Tensor'> torch.Size([21])
            tensor([0.9614, 0.9472, 0.9336, 0.7807, 0.7074, 0.6851, 0.6587, 0.6257, 0.5544,
                    0.5148, 0.5015, 0.4297, 0.3264, 0.3127, 0.2383, 0.2364, 0.2152, 0.1829,
                    0.1472, 0.1323, 0.1091], device='cuda:0')
            pred_labels <class 'torch.Tensor'> torch.Size([21])
            tensor([1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 3, 2, 1, 3, 3, 2, 3, 2, 2],
                   device='cuda:0')
            """

            V.draw_scenes(
                points=data_dict['points'][:, 1:].cpu().numpy(),
                ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'],
                ref_labels=pred_dicts[0]['pred_labels'],
                point_colors=data_dict['points'][:, -3:].cpu().numpy()
            )
                
            print('Time cost per batch: %s' % (round((time_end - time_start) / 10, 3)))

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
