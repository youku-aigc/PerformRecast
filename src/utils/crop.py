# coding: utf-8

"""
Cropping helpers used by PerformRecast.
"""

import numpy as np
from math import sin, cos, acos, degrees
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

DTYPE = np.float32
CV2_INTERP = cv2.INTER_LINEAR


def _transform_img(img, M, dsize, flags=CV2_INTERP, borderMode=None):
    """Apply a similarity / affine transform to `img`."""
    if isinstance(dsize, (tuple, list)):
        _dsize = tuple(dsize)
    else:
        _dsize = (dsize, dsize)
    if borderMode is not None:
        return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags,
                              borderMode=borderMode, borderValue=(0, 0, 0))
    return cv2.warpAffine(img, M[:2, :], dsize=_dsize, flags=flags)


def _transform_pts(pts, M):
    """Apply a similarity / affine transform to a set of 2D points."""
    return pts @ M[:2, :2].T + M[:2, 2]


def parse_pt2_from_pt101(pt101, use_lip=True):
    pt_left_eye = np.mean(pt101[[39, 42, 45, 48]], axis=0)
    pt_right_eye = np.mean(pt101[[51, 54, 57, 60]], axis=0)
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt101[75] + pt101[81]) / 2
        return np.stack([pt_center_eye, pt_center_lip], axis=0)
    return np.stack([pt_left_eye, pt_right_eye], axis=0)


def parse_pt2_from_pt106(pt106, use_lip=True):
    pt_left_eye = np.mean(pt106[[33, 35, 40, 39]], axis=0)
    pt_right_eye = np.mean(pt106[[87, 89, 94, 93]], axis=0)
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt106[52] + pt106[61]) / 2
        return np.stack([pt_center_eye, pt_center_lip], axis=0)
    return np.stack([pt_left_eye, pt_right_eye], axis=0)


def parse_pt2_from_pt203(pt203, use_lip=True):
    pt_left_eye = np.mean(pt203[[0, 6, 12, 18]], axis=0)
    pt_right_eye = np.mean(pt203[[24, 30, 36, 42]], axis=0)
    if use_lip:
        pt_center_eye = (pt_left_eye + pt_right_eye) / 2
        pt_center_lip = (pt203[48] + pt203[66]) / 2
        return np.stack([pt_center_eye, pt_center_lip], axis=0)
    return np.stack([pt_left_eye, pt_right_eye], axis=0)


def parse_pt2_from_pt68(pt68, use_lip=True):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55], dtype=np.int32) - 1
    if use_lip:
        pt5 = np.stack([
            np.mean(pt68[lm_idx[[1, 2]], :], 0),
            np.mean(pt68[lm_idx[[3, 4]], :], 0),
            pt68[lm_idx[0], :],
            pt68[lm_idx[5], :],
            pt68[lm_idx[6], :],
        ], axis=0)
        return np.stack([(pt5[0] + pt5[1]) / 2, (pt5[3] + pt5[4]) / 2], axis=0)
    return np.stack([
        np.mean(pt68[lm_idx[[1, 2]], :], 0),
        np.mean(pt68[lm_idx[[3, 4]], :], 0),
    ], axis=0)


def parse_pt2_from_pt5(pt5, use_lip=True):
    if use_lip:
        return np.stack([(pt5[0] + pt5[1]) / 2, (pt5[3] + pt5[4]) / 2], axis=0)
    return np.stack([pt5[0], pt5[1]], axis=0)


def parse_pt2_from_pt9(pt9, use_lip=True):
    """9 pts: ['right eye right', 'right eye left', 'left eye right',
    'left eye left', 'nose tip', 'lip right', 'lip left',
    'upper lip', 'lower lip']."""
    if use_lip:
        pt9 = np.stack([
            (pt9[2] + pt9[3]) / 2,
            (pt9[0] + pt9[1]) / 2,
            pt9[4],
            (pt9[5] + pt9[6]) / 2,
        ], axis=0)
        return np.stack([(pt9[0] + pt9[1]) / 2, pt9[3]], axis=0)
    return np.stack([(pt9[2] + pt9[3]) / 2, (pt9[0] + pt9[1]) / 2], axis=0)


def parse_pt2_from_pt_x(pts, use_lip=True):
    n = pts.shape[0]
    if n == 101:
        pt2 = parse_pt2_from_pt101(pts, use_lip=use_lip)
    elif n == 106:
        pt2 = parse_pt2_from_pt106(pts, use_lip=use_lip)
    elif n == 68:
        pt2 = parse_pt2_from_pt68(pts, use_lip=use_lip)
    elif n == 5:
        pt2 = parse_pt2_from_pt5(pts, use_lip=use_lip)
    elif n == 203:
        pt2 = parse_pt2_from_pt203(pts, use_lip=use_lip)
    elif n > 101:
        pt2 = parse_pt2_from_pt101(pts[:101], use_lip=use_lip)
    elif n == 9:
        pt2 = parse_pt2_from_pt9(pts, use_lip=use_lip)
    else:
        raise Exception(f'Unknown shape: {pts.shape}')

    if not use_lip:
        # rotate pt2 90 degrees clockwise so that downstream code stays unchanged
        v = pt2[1] - pt2[0]
        pt2[1, 0] = pt2[0, 0] - v[1]
        pt2[1, 1] = pt2[0, 1] + v[0]
    return pt2


def parse_rect_from_landmark(pts, scale=1.5, need_square=True, vx_ratio=0,
                             vy_ratio=0, use_deg_flag=False, **kwargs):
    """Parse (center, size, angle) of the face bounding rectangle from landmarks."""
    pt2 = parse_pt2_from_pt_x(pts, use_lip=kwargs.get('use_lip', True))

    uy = pt2[1] - pt2[0]
    l = np.linalg.norm(uy)
    if l <= 1e-3:
        uy = np.array([0, 1], dtype=DTYPE)
    else:
        uy /= l
    ux = np.array((uy[1], -uy[0]), dtype=DTYPE)

    angle = acos(ux[0])
    if ux[1] < 0:
        angle = -angle

    M = np.array([ux, uy])

    center0 = np.mean(pts, axis=0)
    rpts = (pts - center0) @ M.T
    lt_pt = np.min(rpts, axis=0)
    rb_pt = np.max(rpts, axis=0)
    center1 = (lt_pt + rb_pt) / 2

    size = rb_pt - lt_pt
    if need_square:
        m = max(size[0], size[1])
        size[0] = m
        size[1] = m
    size *= scale

    center = center0 + ux * center1[0] + uy * center1[1]
    center = center + ux * (vx_ratio * size) + uy * (vy_ratio * size)
    if use_deg_flag:
        angle = degrees(angle)
    return center, size, angle


def _estimate_similar_transform_from_pts(pts, dsize, scale=1.5, vx_ratio=0,
                                         vy_ratio=-0.1, flag_do_rot=True, **kwargs):
    """Compute the original->crop affine matrix (and its inverse) from landmarks."""
    center, size, angle = parse_rect_from_landmark(
        pts, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio,
        use_lip=kwargs.get('use_lip', True),
    )
    s = dsize / size[0]
    tgt_center = np.array([dsize / 2, dsize / 2], dtype=DTYPE)

    if flag_do_rot:
        costheta, sintheta = cos(angle), sin(angle)
        cx, cy = center[0], center[1]
        tcx, tcy = tgt_center[0], tgt_center[1]
        M_INV = np.array(
            [[s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy)],
             [-s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy)]],
            dtype=DTYPE,
        )
    else:
        M_INV = np.array(
            [[s, 0, tgt_center[0] - s * center[0]],
             [0, s, tgt_center[1] - s * center[1]]],
            dtype=DTYPE,
        )

    M_INV_H = np.vstack([M_INV, np.array([0, 0, 1])])
    M = np.linalg.inv(M_INV_H)
    return M_INV, M[:2, ...]


def crop_image(img, pts: np.ndarray, **kwargs):
    """Crop a face out of `img` using the affine transform implied by `pts`."""
    dsize = kwargs.get('dsize', 224)
    scale = kwargs.get('scale', 1.5)
    vy_ratio = kwargs.get('vy_ratio', -0.1)

    M_INV, _ = _estimate_similar_transform_from_pts(
        pts, dsize=dsize, scale=scale, vy_ratio=vy_ratio,
        flag_do_rot=kwargs.get('flag_do_rot', True),
    )

    img_crop = _transform_img(img, M_INV, dsize)
    pt_crop = _transform_pts(pts, M_INV)

    M_o2c = np.vstack([M_INV, np.array([0, 0, 1], dtype=DTYPE)])
    M_c2o = np.linalg.inv(M_o2c)
    return {
        'M_o2c': M_o2c,  # original -> crop, 3x3
        'M_c2o': M_c2o,  # crop -> original, 3x3
        'img_crop': img_crop,
        'pt_crop': pt_crop,
    }


def prepare_paste_back(mask_crop, crop_M_c2o, dsize):
    """Project the cropped-space mask back into original-image space."""
    mask_ori = _transform_img(mask_crop, crop_M_c2o, dsize)
    return mask_ori.astype(np.float32) / 255.


def paste_back(img_crop, M_c2o, img_ori, mask_ori):
    """Composite the cropped (animated) image back into the original frame."""
    dsize = (img_ori.shape[1], img_ori.shape[0])
    result = _transform_img(img_crop, M_c2o, dsize=dsize)
    return np.clip(mask_ori * result + (1 - mask_ori) * img_ori, 0, 255).astype(np.uint8)
