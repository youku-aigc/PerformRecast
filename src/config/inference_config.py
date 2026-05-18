# coding: utf-8
"""
Config dataclass used at PerformRecast inference time.
"""
import cv2
from numpy import ndarray
from dataclasses import dataclass, field
from typing import Literal
from .base_config import PrintableConfig, make_abs_path


@dataclass(repr=False)  # use repr from PrintableConfig
class InferenceConfig(PrintableConfig):
    # PerformRecast model definition + checkpoints
    models_config: str = make_abs_path('./performrecast_models.yaml')
    checkpoint_F: str = make_abs_path('../../pretrained_weights/performrecast/appearance_feature_extractor.pth')
    checkpoint_M: str = make_abs_path('../../pretrained_weights/performrecast/motion_extractor.pth')
    checkpoint_G: str = make_abs_path('../../pretrained_weights/performrecast/spade_generator.pth')
    checkpoint_W: str = make_abs_path('../../pretrained_weights/performrecast/warping_module.pth')

    # Runtime flags
    flag_use_half_precision: bool = True
    device_id: int = 0
    flag_pasteback: bool = True
    flag_do_crop: bool = True
    flag_stitching: bool = True
    flag_smooth: bool = True
    # Larger -> smoother animated video, but loses motion accuracy.
    driving_smooth_observation_variance: float = 3e-7

    output_format: Literal['mp4', 'gif'] = 'mp4'
    crf: int = 15
    output_fps: int = 25

    mask_crop: ndarray = field(default_factory=lambda: cv2.imread(
        make_abs_path('../utils/resources/mask_template.png'), cv2.IMREAD_COLOR))
    size_gif: int = 256
