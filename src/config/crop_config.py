# coding: utf-8

"""
parameters used for crop faces
"""
from dataclasses import dataclass
from .base_config import PrintableConfig, make_abs_path


@dataclass(repr=False)  # use repr from PrintableConfig
class CropConfig(PrintableConfig):
    insightface_root: str = make_abs_path("../../pretrained_weights/insightface")
    landmark_ckpt_path: str = make_abs_path("../../pretrained_weights/landmark.onnx")

    device_id: int = 0  # gpu device id
    flag_force_cpu: bool = False  # force cpu inference, WIP
    det_thresh: float = 0.15 # detection threshold
    ########## source image or video cropping option ##########
    face_index: int = 0
    dsize: int = 512  # crop size
    scale: float = 1.5  # scale factor
    vx_ratio: float = 0  # vx ratio
    vy_ratio: float = -0.125  # vy ratio +up, -down
    flag_do_rot: bool = True
    max_face_num: int = 0  # max face number, 0 mean no limit
    ########## driving image or video cropping option ##########
    drv_face_index: int = 0
    drv_dsize: int = 256  # crop size
    drv_scale: float = 1.5  # scale factor
    drv_vx_ratio: float = 0  # vx ratio
    drv_vy_ratio: float = -0.125  # vy ratio +up, -down
    drv_flag_do_rot: bool = True
    direction: str = "large-small"  # direction of cropping
