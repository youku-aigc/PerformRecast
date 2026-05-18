# coding: utf-8

"""
Command-line argument config for PerformRecast inference.
"""
from dataclasses import dataclass
import tyro
from typing_extensions import Annotated
from typing import Optional
from .base_config import PrintableConfig, make_abs_path


@dataclass(repr=False)
class ArgumentConfig(PrintableConfig):
    ########## input / output ##########
    # Source / driving are both portrait videos (mp4/mov/avi/...).
    source: Annotated[str, tyro.conf.arg(aliases=["-s"])] = make_abs_path('../../assets/source/HZTX_EP01_S2_087_Comp.mp4')
    driving: Annotated[str, tyro.conf.arg(aliases=["-d"])] = make_abs_path('../../assets/driving/HZTX_EP01_S2_087_Comp_Drv_v001.mp4')
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = make_abs_path('../../animations/')

    # Optional reference image. When ref_flag == 1, the relative-expression branch
    # uses `reference` as the neutral reference instead of the first driving frame.
    ref_flag: int = 0
    reference: Optional[str] = None

    # Optional audio source: 1 = use source-video audio, 0 = use driving-video audio.
    audio: int = 1
    src_audio: Optional[str] = None
    drv_audio: Optional[str] = None

    # Inference mode controlling how the source/driving expressions are mixed:
    #   1 = replacement  (replace the source expression with the driving expression)
    #   2 = enhancement  (add the driving expression delta on top of the source expression)
    inference_mode: int = 1

    device_id: int = 0

    ########## source crop ##########
    face_index: int = 0
    dsize: int = 512
    scale: float = 1.5
    vx_ratio: float = 0.0
    vy_ratio: float = -0.125
    flag_align: int = 1

    ########## driving crop ##########
    drv_face_index: int = 0
    drv_dsize: int = 256
    drv_scale: float = 1.5
    drv_vx_ratio: float = 0.0
    drv_vy_ratio: float = -0.125
    drv_flag_align: int = 1

    ########## inference flags ##########
    flag_use_half_precision: bool = True
    flag_force_cpu: bool = False
    flag_pasteback: bool = True
    flag_do_crop: bool = True
    flag_stitching: bool = True
    flag_smooth: bool = False
    driving_smooth_observation_variance: float = 3e-7

    ########## crop ##########
    det_thresh: float = 0.15
