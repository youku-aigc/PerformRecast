# coding: utf-8
"""Video helpers used by the PerformRecast pipeline."""
import cv2

from .rprint import rlog as log


def get_fps(filepath: str, default_fps: int = 25) -> float:
    try:
        fps = cv2.VideoCapture(filepath).get(cv2.CAP_PROP_FPS)
        if fps in (0, None):
            fps = default_fps
    except Exception as e:
        log(e)
        fps = default_fps
    return fps
