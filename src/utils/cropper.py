# coding: utf-8
import numpy as np
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
DTYPE = np.float32
CV2_INTERP = cv2.INTER_LINEAR

from typing import List, Tuple, Union
from dataclasses import dataclass, field
from math import sin, cos, acos, degrees

from ..config.crop_config import CropConfig
from .io import contiguous
from .rprint import rlog as log
from .face_analysis_diy import FaceAnalysisDIY
from .human_landmark_runner import LandmarkRunner as HumanLandmark


@dataclass
class Trajectory:
    start: int = -1  # start frame
    end: int = -1  # end frame
    lmk_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)
    M_c2o_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)
    lmk_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)
    frame_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)
    frame_rgb_crop_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)
    frame_rgb_crop_512_lst: Union[Tuple, List, np.ndarray] = field(default_factory=list)

class Cropper(object):
    def __init__(self, **kwargs) -> None:
        self.crop_cfg = kwargs.get("crop_cfg", None)
        device_id = kwargs.get("device_id", 0)
        device = f"cuda:{device_id}"
        face_analysis_wrapper_provider = ["CUDAExecutionProvider"]
        self.face_analysis_wrapper = FaceAnalysisDIY(
                    name="buffalo_l",
                    root=self.crop_cfg.insightface_root,
                    providers=face_analysis_wrapper_provider,
                )
        self.face_analysis_wrapper.prepare(ctx_id=device_id,
                                           det_size=(512, 512),
                                           det_thresh=self.crop_cfg.det_thresh)
        self.face_analysis_wrapper.warmup()

        self.human_landmark_runner = HumanLandmark(
            ckpt_path=self.crop_cfg.landmark_ckpt_path,
            onnx_provider=device,
            device_id=device_id,
        )
        self.human_landmark_runner.warmup()

    def _transform_pts(self, pts, M):
        return pts @ M[:2, :2].T + M[:2, 2]
    def _transform_img(self, img, M, dsize, flags=CV2_INTERP, borderMode=None):
        if isinstance(dsize, tuple) or isinstance(dsize, list):
            _dsize = tuple(dsize)
        else:
            _dsize = (dsize, dsize)

        if borderMode is not None:
            return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags, borderMode=borderMode,
                                  borderValue=(0, 0, 0))
        else:
            return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)
    def parse_pt2_from_pt203(self, pt203, use_lip=True):
        """
        parsing the 2 points according to the 203 points, which cancels the roll
        """
        pt_left_eye = np.mean(pt203[[0, 6, 12, 18]], axis=0)  # left eye center
        pt_right_eye = np.mean(pt203[[24, 30, 36, 42]], axis=0)  # right eye center
        if use_lip:
            # use lip
            pt_center_eye = (pt_left_eye + pt_right_eye) / 2
            pt_center_lip = (pt203[48] + pt203[66]) / 2
            pt2 = np.stack([pt_center_eye, pt_center_lip], axis=0)
        else:
            pt2 = np.stack([pt_left_eye, pt_right_eye], axis=0)
        return pt2
    def parse_pt2_from_pt_x(self, pts, use_lip=True):
        if pts.shape[0] == 203:
            pt2 = self.parse_pt2_from_pt203(pts, use_lip=use_lip)
        else:
            raise Exception(f'Unknow shape: {pts.shape}')
        if not use_lip:
            # NOTE: to compile with the latter code, need to rotate the pt2 90 degrees clockwise manually
            v = pt2[1] - pt2[0]
            pt2[1, 0] = pt2[0, 0] - v[1]
            pt2[1, 1] = pt2[0, 1] + v[0]

        return pt2
    def parse_rect_from_landmark(self, pts, scale=1.5, need_square=True, vx_ratio=0, vy_ratio=0,
                                 use_deg_flag=False, **kwargs):
        pt2 = self.parse_pt2_from_pt_x(pts, use_lip=kwargs.get('use_lip', True))

        uy = pt2[1] - pt2[0]
        l = np.linalg.norm(uy)
        if l <= 1e-3:
            uy = np.array([0, 1], dtype=DTYPE)
        else:
            uy /= l
        ux = np.array((uy[1], -uy[0]), dtype=DTYPE)

        # the rotation degree of the x-axis, the clockwise is positive, the counterclockwise is negative (image coordinate system)
        angle = acos(ux[0])
        if ux[1] < 0:
            angle = -angle

        # rotation matrix
        M = np.array([ux, uy])

        # calculate the size which contains the angle degree of the bbox, and the center
        center0 = np.mean(pts, axis=0)
        rpts = (pts - center0) @ M.T  # (M @ P.T).T = P @ M.T
        lt_pt = np.min(rpts, axis=0)
        rb_pt = np.max(rpts, axis=0)
        center1 = (lt_pt + rb_pt) / 2

        size = rb_pt - lt_pt
        if need_square:
            m = max(size[0], size[1])
            size[0] = m
            size[1] = m

        size *= scale  # scale size
        center = center0 + ux * center1[0] + uy * center1[1]  # counterclockwise rotation, equivalent to M.T @ center1.T
        center = center + ux * (vx_ratio * size) + uy * \
                 (vy_ratio * size)  # considering the offset in vx and vy direction

        if use_deg_flag:
            angle = degrees(angle)

        return center, size, angle
    def _estimate_similar_transform_from_pts(self, pts, dsize, scale=1.5, vx_ratio=0, vy_ratio=-0.1,
                                             flag_do_rot=True, **kwargs):
        center, size, angle = self.parse_rect_from_landmark(
            pts, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio,
            use_lip=kwargs.get('use_lip', True)
        )

        s = dsize / size[0]  # scale
        tgt_center = np.array([dsize / 2, dsize / 2], dtype=DTYPE)  # center of dsize

        if flag_do_rot:
            costheta, sintheta = cos(angle), sin(angle)
            cx, cy = center[0], center[1]  # ori center
            tcx, tcy = tgt_center[0], tgt_center[1]  # target center
            # need to infer
            M_INV = np.array(
                [[s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
                 [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)]],
                dtype=DTYPE
            )
        else:
            M_INV = np.array(
                [[s, 0, tgt_center[0] - s * center[0]],
                 [0, s, tgt_center[1] - s * center[1]]],
                dtype=DTYPE
            )

        M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
        M = np.linalg.inv(M_INV_H)

        # M_INV is from the original image to the cropped image, M is from the cropped image to the original image
        return M_INV, M[:2, ...]

    def crop_image(self, img, pts: np.ndarray, **kwargs):
        dsize = kwargs.get('dsize', 224)
        scale = kwargs.get('scale', 1.5)  # 1.5 | 1.6
        vy_ratio = kwargs.get('vy_ratio', -0.1)  # -0.0625 | -0.1

        M_INV, _ = self._estimate_similar_transform_from_pts(
            pts,
            dsize=dsize,
            scale=scale,
            vy_ratio=vy_ratio,
            flag_do_rot=kwargs.get('flag_do_rot', True),
        )

        img_crop = self._transform_img(img, M_INV, dsize)  # origin to crop
        pt_crop = self._transform_pts(pts, M_INV)

        M_o2c = np.vstack([M_INV, np.array([0, 0, 1], dtype=DTYPE)])
        M_c2o = np.linalg.inv(M_o2c)

        ret_dct = {
            'M_o2c': M_o2c,  # from the original image to the cropped image 3x3
            'M_c2o': M_c2o,  # from the cropped image to the original image 3x3
            'img_crop': img_crop,  # the cropped image
            'pt_crop': pt_crop,  # the landmarks of the cropped image
        }

        return ret_dct

    def crop_align_drving_video(self, source_rgb_lst, crop_cfg: CropConfig, index=0, **kwargs):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()
        direction = kwargs.get("direction", "large-small")
        for idx, frame_rgb in enumerate(source_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=crop_cfg.direction,
                    max_face_num=crop_cfg.max_face_num,
                )
                if len(src_face) == 0:
                    log(f"No face detected in the frame #{idx}")
                    continue
                elif len(src_face) > 1:
                    log(f"More than one face detected in the source frame_{idx}, "
                        f"only pick one face by rule {direction}.")
                src_face = src_face[index]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                # TODO: add IOU check for tracking
                lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
                trajectory.end = idx

            trajectory.lmk_lst.append(lmk)
            ret_dct = self.crop_image(
                frame_rgb,  # ndarray
                lmk,  # 106x2 or Nx2
                dsize=crop_cfg.drv_dsize,
                scale=crop_cfg.drv_scale,
                vx_ratio=crop_cfg.drv_vx_ratio,
                vy_ratio=crop_cfg.drv_vy_ratio,
                flag_do_rot=crop_cfg.drv_flag_do_rot,
            )

            trajectory.frame_crop_lst.append(ret_dct["img_crop"])
            trajectory.lmk_crop_lst.append(ret_dct["pt_crop"])

        return {
            "frame_crop_lst": trajectory.frame_crop_lst,
            "lmk_crop_lst": trajectory.lmk_crop_lst,
        }

    def crop_source_video(self, source_rgb_lst, crop_cfg: CropConfig, index=0, **kwargs):
        """Tracking based landmarks/alignment and cropping"""
        trajectory = Trajectory()
        direction = kwargs.get("direction", "large-small")
        for idx, frame_rgb in enumerate(source_rgb_lst):
            if idx == 0 or trajectory.start == -1:
                src_face = self.face_analysis_wrapper.get(
                    contiguous(frame_rgb[..., ::-1]),
                    flag_do_landmark_2d_106=True,
                    direction=crop_cfg.direction,
                    max_face_num=crop_cfg.max_face_num,
                )
                if len(src_face) == 0:
                    log(f"No face detected in the frame #{idx}")
                    continue
                elif len(src_face) > 1:
                    log(f"More than one face detected in the source frame_{idx}, only pick one face by rule {direction}.")
                src_face = src_face[index]
                lmk = src_face.landmark_2d_106
                lmk = self.human_landmark_runner.run(frame_rgb, lmk)
                trajectory.start, trajectory.end = idx, idx
            else:
                # TODO: add IOU check for tracking
                lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
                trajectory.end = idx


            ret_dct = self.crop_image(frame_rgb, lmk, dsize=crop_cfg.dsize, scale=crop_cfg.scale,
                                 vx_ratio=crop_cfg.vx_ratio, vy_ratio=crop_cfg.vy_ratio,
                                 flag_do_rot=crop_cfg.flag_do_rot)

            # update a 256x256 version for network input
            ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)

            trajectory.lmk_lst.append(lmk)
            trajectory.frame_rgb_crop_lst.append(ret_dct["img_crop_256x256"])
            trajectory.frame_rgb_crop_512_lst.append(ret_dct["img_crop"])
            trajectory.lmk_crop_lst.append(ret_dct["pt_crop"])
            trajectory.M_c2o_lst.append(ret_dct['M_c2o'])

        return {
            "frame_crop_lst": trajectory.frame_rgb_crop_lst,
            "frame_crop_512x512_lst": trajectory.frame_rgb_crop_512_lst,
            "lmk_crop_lst": trajectory.lmk_crop_lst,
            "M_c2o_lst": trajectory.M_c2o_lst,
            "lmk_lst": trajectory.lmk_lst,
        }