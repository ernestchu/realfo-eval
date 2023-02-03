import math

import torch
from torch import nn
import torch.nn.functional as F
from argparse import Namespace
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    RandomCrop,
    RandomHorizontalFlip,
    RandomGrayscale,
    Resize,
    Normalize,
)

from .csn import csn_temporal_no_head


class Detector(nn.Module):
    """
    config: https://github.com/ahaliassos/RealForensics/blob/main/stage2/conf/model/visual_backbone/csn_r101.yaml
    """
    def __init__(self):
        super().__init__()
        self.encoder = csn_temporal_no_head(model_depth=101)
        self.head = MeanLinear()
        self.transform = self._transform()

    def forward(self, x):
        return self.head(self.encoder(x))

    def _transform(self):
        return self._video_transform(mode='val')

    '''
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    The code below are ported from RealForensics repo
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    def _copied_args(self):
        '''
        data config copied from:
        https://github.com/ahaliassos/RealForensics/blob/main/stage2/conf/data/combined.yaml
        '''
        args = Namespace()
        # https://github.com/ahaliassos/RealForensics/blob/main/stage2/conf/data/channel/rgb.yaml
        args.channel = Namespace()
        args.channel.in_video_channels = 3
        args.channel.grayscale_prob = 0.5

        # https://github.com/ahaliassos/RealForensics/blob/main/stage2/conf/data/crop_type/full_face.yaml
        args.crop_type = Namespace()
        args.crop_type.random_crop_dim = 140
        args.crop_type.resize_dim = 112
        args.crop_type.random_erasing_prob = 0.5
        args.crop_type.random_erasing_scale = [0.02, 0.33]

        args.n_fft = 512
        args.n_mels = 80
        args.all_but = None
        args.audio2video = 4
        args.win_length = 320
        args.aug_prob = 1.0
        args.horizontal_flip_prob = 0.5
        args.num_frames = 25
        args.time_mask_video = 12
        args.n_time_mask_video = 1
        args.mask_version = 'v1'
        args.time_mask_prob_video = 0.5
        args.time_mask_targets = False
        args.clean_targets = True

        return args

    def _video_transform(self, mode):
        '''
        source: https://github.com/ahaliassos/RealForensics/blob/990243bd3aee8fff4742045e110411d53b504f13/stage2/data/combined_dm.py#L127
        note: we discard the part for `video_aug`, as it is unused in evaluation
              see: https://github.com/ahaliassos/RealForensics/blob/0ef7f9931a803262d82e71ccee6bcff5e0148143/stage2/combined_learner.py#L49
        '''
        args = self._copied_args()
        transform = [
                        # UniformTemporalSubsample(args.num_frames),
                        LambdaModule(lambda x: x / 255.),
                    ] + (
                        [
                            RandomCrop(args.crop_type.random_crop_dim),
                            Resize(args.crop_type.resize_dim),
                            RandomHorizontalFlip(args.horizontal_flip_prob)
                        ]
                        if mode == "train" else [CenterCrop(args.crop_type.random_crop_dim),
                                                 Resize(args.crop_type.resize_dim)]
                    )
        if args.channel.in_video_channels == 1:
            transform.extend(
                [LambdaModule(lambda x: x.transpose(0, 1)), Grayscale(), LambdaModule(lambda x: x.transpose(0, 1))])

        if args.channel.in_video_channels != 1 and math.isclose(args.channel.grayscale_prob, 1.0):
            transform.extend(
                [
                    LambdaModule(lambda x: x.transpose(0, 1)),
                    RandomGrayscale(args.channel.grayscale_prob),
                    LambdaModule(lambda x: x.transpose(0, 1))
                ]
            )

        transform.append(Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        return Compose(transform)


class LambdaModule(nn.Module):
    def __init__(self, lambda_fn):
        super().__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        return self.lambda_fn(x)


class MeanLinear(nn.Module):
    '''
    source: https://github.com/ahaliassos/RealForensics/blob/main/stage2/models/linear.py
    config: https://github.com/ahaliassos/RealForensics/blob/main/stage2/conf/model/df_predictor/linear.yaml
    '''
    def __init__(self, in_dim=2048, out_dim=1, norm_linear=True, scale=64):
        super().__init__()

        if norm_linear:
            self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.scale = scale
            self.linear = lambda x: F.linear(F.normalize(x), F.normalize(self.weight)) * self.scale
        else:
            self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.mean(-1)
        return self.linear(x)

