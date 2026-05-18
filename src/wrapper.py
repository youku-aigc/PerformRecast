# coding: utf-8
"""
PerformRecast inference wrapper.

Loads the four core sub-networks used in
"PerformRecast: Expression and Head Pose Disentanglement for Portrait Video Editing"
(CVPR 2026):

  * Appearance feature extractor  (F)  - DINOv2-based
  * Motion extractor              (M)  - ConvNeXt V2-based
  * Warping module                (W)
  * SPADE generator               (G)
"""
import os.path as osp
import contextlib
import numpy as np
import yaml

import torch

from .utils.timer import Timer
from .utils.rprint import rlog as log

from .modules.dinov2_3d_feature_extractor import Dinov2FeatureExtractor
from .modules.convnextv2 import ConvNeXtV2MotionExtractor
from .modules.warping_network import WarpingNetwork
from .modules.spade_generator import SPADEDecoder


class PerformRecastWrapper(object):
    """Wrapper that loads the PerformRecast checkpoints and exposes the basic
    image-tensor preparation / feature-extraction / warp-decode utilities used
    by the inference pipeline."""

    def __init__(self, inference_cfg):
        self.inference_cfg = inference_cfg
        self.device_id = inference_cfg.device_id
        self.device = 'cuda:' + str(self.device_id)
        self.compile = getattr(inference_cfg, 'compile', False)

        model_config = yaml.load(open(inference_cfg.models_config, 'r'), Loader=yaml.SafeLoader)

        # F: appearance feature extractor
        appearance_params = model_config['model_params']['appearance_feature_extractor_params']
        self.appearance_feature_extractor = Dinov2FeatureExtractor(**appearance_params).to(self.device)
        self.appearance_feature_extractor.load_state_dict(
            torch.load(inference_cfg.checkpoint_F, map_location=lambda storage, loc: storage))
        self.appearance_feature_extractor.eval()
        log(f'Load appearance_feature_extractor from {osp.realpath(inference_cfg.checkpoint_F)} done.')

        # M: motion extractor
        motion_params = model_config['model_params']['motion_extractor_params']
        self.motion_extractor = ConvNeXtV2MotionExtractor(**motion_params).to(self.device)
        self.motion_extractor.load_state_dict(
            torch.load(inference_cfg.checkpoint_M, map_location=lambda storage, loc: storage))
        self.motion_extractor.eval()
        log(f'Load motion_extractor from {osp.realpath(inference_cfg.checkpoint_M)} done.')

        # W: warping module
        warping_params = model_config['model_params']['warping_module_params']
        self.warping_module = WarpingNetwork(**warping_params).to(self.device)
        self.warping_module.load_state_dict(
            torch.load(inference_cfg.checkpoint_W, map_location=lambda storage, loc: storage))
        self.warping_module.eval()
        log(f'Load warping_module from {osp.realpath(inference_cfg.checkpoint_W)} done.')

        # G: SPADE generator
        spade_params = model_config['model_params']['spade_generator_params']
        self.spade_generator = SPADEDecoder(**spade_params).to(self.device)
        self.spade_generator.load_state_dict(
            torch.load(inference_cfg.checkpoint_G, map_location=lambda storage, loc: storage))
        self.spade_generator.eval()
        log(f'Load spade_generator from {osp.realpath(inference_cfg.checkpoint_G)} done.')

        self.timer = Timer()

    def inference_ctx(self):
        if self.device == "mps":
            return contextlib.nullcontext()
        return torch.autocast(
            device_type=self.device[:4],
            dtype=torch.float16,
            enabled=self.inference_cfg.flag_use_half_precision,
        )

    def prepare_videos(self, imgs) -> torch.Tensor:
        """Build a batch tensor from a list / ndarray of uint8 RGB frames.

        Input:  list of HxWx3 uint8, or ndarray of shape NxHxWx3 uint8.
        Output: tensor of shape Tx1x3xHxW, float32 in [0, 1], on self.device.
        """
        if isinstance(imgs, list):
            _imgs = np.array(imgs)[..., np.newaxis]
        elif isinstance(imgs, np.ndarray):
            _imgs = imgs
        else:
            raise ValueError(f'imgs type error: {type(imgs)}')

        y = _imgs.astype(np.float32) / 255.
        y = np.clip(y, 0, 1)
        y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)
        y = y.to(self.device)
        return y

    def extract_feature_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Run the appearance feature extractor F on a Bx3xHxW tensor."""
        with torch.no_grad(), self.inference_ctx():
            feature_3d = self.appearance_feature_extractor(x)
        return feature_3d.float()

    def warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor,
                    kp_driving: torch.Tensor) -> torch.Tensor:
        """Warp the source feature volume with the driving keypoints and decode
        with the SPADE generator: D(W(f_s; x_s, x'_d,i))."""
        with torch.no_grad(), self.inference_ctx():
            if self.compile:
                torch.compiler.cudagraph_mark_step_begin()
            ret_dct = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
            ret_dct['out'] = self.spade_generator(feature=ret_dct['out'])

            if self.inference_cfg.flag_use_half_precision:
                for k, v in ret_dct.items():
                    if isinstance(v, torch.Tensor):
                        ret_dct[k] = v.float()
        return ret_dct

    def parse_output(self, out: torch.Tensor) -> np.ndarray:
        """Convert a Bx3xHxW float tensor in [0,1] into an NxHxWx3 uint8 array."""
        out = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])
        out = np.clip(out, 0, 1)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return out
