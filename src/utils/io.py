# coding: utf-8
import os.path as osp

import imageio
import numpy as np
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)


def load_image_rgb(image_path: str) -> np.ndarray:
    if not osp.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_video(video_info, n_frames: int = -1):
    reader = imageio.get_reader(video_info, "ffmpeg")
    frames = []
    for idx, frame_rgb in enumerate(reader):
        if n_frames > 0 and idx >= n_frames:
            break
        frames.append(frame_rgb)
    reader.close()
    return frames


def contiguous(obj: np.ndarray) -> np.ndarray:
    if not obj.flags.c_contiguous:
        obj = obj.copy(order="C")
    return obj
