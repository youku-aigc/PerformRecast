# coding: utf-8
"""
Entry point for PerformRecast inference.

Given a source portrait video and a driving portrait video, transfers the
driving expression onto the source while preserving the source head pose,
identity and appearance.

Example:
    python inference.py \
        -s ./assets/source/HZTX_EP01_S2_087_Comp.mp4 \
        -d ./assets/driving/HZTX_EP01_S2_087_Comp_Drv_v001.mp4 \
        -o ./animations/ \
        --inference-mode 1
"""
import os
import os.path as osp
import json
import subprocess
from datetime import datetime, timedelta, timezone

import tyro

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.pipeline import PerformRecastPipeline


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving not found: {args.driving}")
    if args.ref_flag == 1 and (args.reference is None or not osp.exists(args.reference)):
        raise FileNotFoundError(f"reference required when ref_flag=1, got: {args.reference}")


def main():
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    # Local ffmpeg binaries take priority over the system ones.
    ffmpeg_dir = osp.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)
    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (ffmpeg + ffprobe) "
            "before running PerformRecast. https://ffmpeg.org/download.html"
        )
    fast_check_args(args)

    # Beijing-time tag for the output filename so multiple runs don't collide.
    bj_time = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d_%H-%M-%S")

    # Default audio sources: by convention the source video carries the audio.
    if args.src_audio is None:
        args.src_audio = args.source
    if args.drv_audio is None:
        args.drv_audio = args.driving

    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)
    pipeline = PerformRecastPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)

    src_name = osp.splitext(osp.basename(args.source))[0]
    drv_name = osp.splitext(osp.basename(args.driving))[0]
    run_name = f'{src_name}__{drv_name}__{bj_time}'

    os.makedirs(args.output_dir, exist_ok=True)
    args.out_file = f'{run_name}.mp4'
    args.output_path = osp.join(args.output_dir, args.out_file)
    args.out_json = args.output_path.replace('.mp4', '.json')

    # Per-run frame directories used by the pipeline; cleaned up at the end.
    args.frames_dir = osp.join(args.output_dir, run_name + '_frames')
    args.crop_output_dir = osp.join(args.output_dir, run_name + '_crop')
    args.concat_output_dir = osp.join(args.output_dir, run_name + '_concat')
    os.makedirs(args.frames_dir, exist_ok=True)
    os.makedirs(args.crop_output_dir, exist_ok=True)
    os.makedirs(args.concat_output_dir, exist_ok=True)

    inference_log = {
        'video_name': args.out_file,
        'inference_version': 'PerformRecast',
        'source_crop_scale': args.scale,
        'src_crop_vy': args.vy_ratio,
        'src_crop_vx': args.vx_ratio,
        'src_crop_dsize': crop_cfg.dsize,
        'inference_mode': args.inference_mode,
        'drv_align': args.drv_flag_align,
        'driving_smooth_observation_variance': args.driving_smooth_observation_variance,
    }
    with open(args.out_json, 'w') as f:
        json.dump(inference_log, f, ensure_ascii=False, indent=4)

    pipeline.execute(args)


if __name__ == "__main__":
    main()
