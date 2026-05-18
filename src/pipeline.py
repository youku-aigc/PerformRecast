# coding: utf-8
import os
from rich.progress import track
import shutil
from PIL import Image
import numpy as np
import cv2;
cv2.setNumThreads(0);
cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from .utils.video import get_fps
from .utils.crop import prepare_paste_back, paste_back
from .utils.io import load_image_rgb, load_video
from .utils.helper import dct2device
from .utils.rprint import rlog as log
from .utils.filter import smooth
from .utils.cropper import Cropper
from .wrapper import PerformRecastWrapper
import pickle


class PerformRecastPipeline(object):
    """End-to-end pipeline for expression-only portrait video editing.

    Given a source video and a driving video, transfers the driving expression
    onto the source while preserving the source head pose, identity and
    appearance.
    """

    def __init__(self, inference_cfg, crop_cfg):

        self.wrapper = PerformRecastWrapper(inference_cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)
        self.flame_masks = pickle.load(open(f'./assets/FLAME_masks/FLAME_masks.pkl', 'rb'),
                                       encoding='latin1')
        self.final_mask = self.flame_masks['face'].tolist()
        # lip [0,8]
        self.flame_lip_index = [2840, 2896, 3509, 1793, 1582, 1734, 3531, 2852]
        # forehead [8,15]
        self.flame_forehead_index = [3505, 3524, 3540, 3729, 3773, 3874, 3899]
        # left eyebrow [15, 21]
        self.flame_left_eyebrow_index = [337, 338, 3154, 3712, 2178, 2177]
        # right eyebrow [21, 27]
        self.flame_right_eyebrow_index = [18, 27, 2135, 3868, 673, 672]
        # left eye [27, 28]
        self.flame_left_eye_index = [2495]
        # right eye [28, 29]
        self.flame_right_eye_index = [1294]
        # left face [29, 34]
        self.flame_left_face_index = [3710, 3743, 3116, 3467, 3465]
        # right face [34, 39]
        self.flame_right_face_index = [3866, 3881, 2081, 3717, 3715]
        # nose [39, 44]
        self.flame_nose_index = [3093, 2750, 3551, 1618, 2058]
        # contour [44, 47]
        self.flame_contour_index = [3416, 3414, 3635]
        self.flame_used_index = (self.flame_lip_index + self.flame_forehead_index +
                                 self.flame_left_eyebrow_index + self.flame_right_eyebrow_index +
                                 self.flame_left_eye_index + self.flame_right_eye_index +
                                 self.flame_left_face_index + self.flame_right_face_index +
                                 self.flame_nose_index + self.flame_contour_index)
        self.flame_used_index_in_face = [self.final_mask.index(item) for item in self.flame_used_index]
        self.flame_used_index_in_face += [-2, -1]

    def headpose_pred_to_degree_ours(self, pred):
        device = pred.device
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)
        pred = F.softmax(pred, dim=1)
        degree = torch.sum(pred * idx_tensor, axis=1) * 3 - 99

        return degree

    def get_rotation_matrix_ours(self, yaw, pitch, roll):
        yaw = yaw / 180 * 3.14
        pitch = pitch / 180 * 3.14
        roll = roll / 180 * 3.14

        roll = roll.unsqueeze(1)
        pitch = pitch.unsqueeze(1)
        yaw = yaw.unsqueeze(1)

        pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch),
                               torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                               torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
        pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

        yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw),
                             torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                             -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
        yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

        roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),
                              torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                              torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
        roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

        rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

        return rot_mat

    def keypoint_transformation(self, kp_canonical, he):
        kp = kp_canonical['kp']
        yaw, pitch, roll, t, exp, scale = he['yaw'], he['pitch'], he['roll'], he['t'], he['exp'], he['scale']
        bs = kp.size(0)
        kp = kp.view(bs, -1, 3)
        exp = exp.view(bs, -1, 3)
        # add orthographic expression deviation
        kp_e = kp + exp
        # keypoint rotation
        yaw = self.headpose_pred_to_degree_ours(yaw)
        pitch = self.headpose_pred_to_degree_ours(pitch)
        roll = self.headpose_pred_to_degree_ours(roll)
        rot_mat = self.get_rotation_matrix_ours(yaw, pitch, roll)  # (bs, 3, 3)
        kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp_e)
        repeat_scale = scale.unsqueeze(1).repeat(1, kp.shape[1], 1)
        kp_rotated *= repeat_scale
        # keypoint translation
        t[..., 2].fill_(0)
        t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
        kp_transformed = kp_rotated + t

        return kp_transformed, rot_mat, yaw, pitch, roll

    def make_motion_template(self, I_lst, **kwargs):
        runing_name = kwargs.get('name', ' ')
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': []}

        for i in track(range(n_frames),
                       description=f'Making {runing_name} motion templates...', total=n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            with torch.no_grad():
                x_i_info = self.wrapper.motion_extractor(I_i)
            x_s, R_i, yaw, pitch, roll = self.keypoint_transformation(x_i_info, x_i_info)
            x_i_info['exp'] = x_i_info['exp'].view(1, -1, 3)
            x_i_info['kp'] = x_i_info['kp'].view(1, -1, 3)

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'yaw': yaw.cpu().numpy().astype(np.float32),
                'pitch': pitch.cpu().numpy().astype(np.float32),
                'roll': roll.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

        return template_dct

    def execute(self, args):
        args.suffix = args.output_path.split('.')[-1]
        device = self.wrapper.device
        inf_cfg = self.wrapper.inference_cfg
        crop_cfg = self.cropper.crop_cfg
        ######## load source input ########
        source_fps = int(get_fps(args.source))
        log(f"Load source video from {args.source}, FPS is {source_fps}")
        source_rgb_lst = load_video(args.source)
        ######## process driving info ########
        driving_fps = int(get_fps(args.driving))
        log(f"Load driving video from: {args.driving}, FPS is {driving_fps}")
        driving_rgb_lst = load_video(args.driving)
        args.n_frames = min(len(source_rgb_lst), len(driving_rgb_lst))
        log(f"{args.n_frames} frames are processed.")
        source_rgb_lst = source_rgb_lst[:args.n_frames]
        driving_rgb_lst = driving_rgb_lst[:args.n_frames]

        if args.ref_flag == 1:
            ref_image = cv2.resize(load_image_rgb(args.reference), driving_rgb_lst[0].shape[:2])
        else:
            ref_image = driving_rgb_lst[0].copy()
        ######## make motion template ########
        log("Start making driving motion template...")
        if args.drv_flag_align == 0:
            driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
            ref_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in [ref_image]]
        elif args.drv_flag_align == 1:
            ret_d = self.cropper.crop_align_drving_video(driving_rgb_lst, crop_cfg, index=args.drv_face_index)
            driving_rgb_crop_256x256_lst, driving_lmk_crop_lst = (
                ret_d['frame_crop_lst'], ret_d['lmk_crop_lst'])

            ret_r = self.cropper.crop_align_drving_video([ref_image], crop_cfg, index=args.drv_face_index)
            ref_rgb_crop_256x256_lst, ref_lmk_crop_lst = (
                ret_r['frame_crop_lst'], ret_r['lmk_crop_lst'])
        else:
            raise ValueError("drv_flag_align should be 0 or 1")

        I_d_lst = self.wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
        I_r_lst = self.wrapper.prepare_videos(ref_rgb_crop_256x256_lst)
        driving_template_dct = self.make_motion_template(I_d_lst, name='driving', output_fps=driving_fps)
        ref_template_dct = self.make_motion_template(I_r_lst, name='reference', output_fps=driving_fps)
        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback:
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")
        ######## process source info ########
        log(f"Start making source motion template...")
        ret_s = self.cropper.crop_source_video(source_rgb_lst, crop_cfg, index=args.face_index)
        log(f'Source video is cropped, {len(ret_s["frame_crop_lst"])} frames are processed.')
        img_crop_256x256_lst, img_crop_512x512_lst, source_lmk_crop_lst, source_M_c2o_lst = \
            ret_s['frame_crop_lst'], ret_s['frame_crop_512x512_lst'], ret_s['lmk_crop_lst'], ret_s['M_c2o_lst']
        I_s_lst = self.wrapper.prepare_videos(img_crop_256x256_lst)
        I_s_512_lst = self.wrapper.prepare_videos(img_crop_512x512_lst)
        source_template_dct = self.make_motion_template(I_s_lst, name='source', output_fps=source_fps)

        # Build per-frame relative / absolute driving expression sequences.
        drv_relative_motion_lst = []
        drv_abs_motion_lst = []

        for drv_idx in range(args.n_frames):
            relative_exp = driving_template_dct['motion'][drv_idx]['exp'] - ref_template_dct['motion'][0]['exp']
            drv_relative_motion_lst.append(relative_exp)
            abs_exp = driving_template_dct['motion'][drv_idx]['exp']
            drv_abs_motion_lst.append(abs_exp)

        if args.flag_smooth:
            drv_rel_motion_smooth = smooth(drv_relative_motion_lst, driving_template_dct['motion'][0]['exp'].shape,
                                           device, args.driving_smooth_observation_variance)
            drv_abs_motion_smooth = smooth(drv_abs_motion_lst, driving_template_dct['motion'][0]['exp'].shape,
                                           device, args.driving_smooth_observation_variance)
        else:
            drv_rel_motion_smooth = [torch.tensor(item[0]).to(device) for item in drv_relative_motion_lst]
            drv_abs_motion_smooth = [torch.tensor(item[0]).to(device) for item in drv_abs_motion_lst]

        ######## animate ########
        I_p_lst = []
        for i in track(range(args.n_frames), description='🚀Animating...', total=args.n_frames):
            x_s_info = source_template_dct['motion'][i]
            x_s_info = dct2device(x_s_info, device)
            I_s = I_s_512_lst[i]
            f_s = self.wrapper.extract_feature_3d(I_s)
            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:  # prepare for paste back
                mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, source_M_c2o_lst[i],
                                                    dsize=(source_rgb_lst[i].shape[1], source_rgb_lst[i].shape[0]))
            # exp animation
            scale_s = x_s_info['scale']
            t_s = x_s_info['t']
            t_s[..., 2].fill_(0)  # zero tz
            x_s = x_s_info['x_s']
            x_s_c = x_s_info['kp']
            rot_mat_s = x_s_info['R']
            exp_s = x_s_info['exp']
            exp_d_rel = drv_rel_motion_smooth[i].unsqueeze(0)
            exp_d_abs = drv_abs_motion_smooth[i].unsqueeze(0)
            if args.inference_mode == 1:
                # "Replacement" mode (README label). Internally implemented
                # as a channel-wise modulation: start from the absolute
                # driving expression and blend back source-side eye / lip /
                # contour channels so micro identity cues (eye shape, lip
                # thickness, jawline) are preserved while the macro
                # expression follows the driver.
                modulated_exp = exp_d_abs.clone()
                modulated_exp[:, 31:34, 2] = exp_s[:, 31:34, 2]
                modulated_exp[:, 36:39, 2] = exp_s[:, 36:39, 2]
                modulated_exp[:, 44:47, 2] = exp_s[:, 44:47, 2]
                modulated_exp[:, 44:47, 0] = exp_s[:, 44:47, 0]
                modulated_exp[:, 44:47, 1] = exp_s[:, 44:47, 1] * 0.2 + exp_d_abs[:, 44:47, 1] * 0.8
                modulated_exp[:, 31:34, :2] = exp_s[:, 31:34, :2] * 0.3 + exp_d_abs[:, 31:34, :2] * 0.7
                modulated_exp[:, 36:39, :2] = exp_s[:, 36:39, :2] * 0.3 + exp_d_abs[:, 36:39, :2] * 0.7
                kp_e = x_s_c + modulated_exp
            elif args.inference_mode == 2:
                # "Enhancement" mode (README label): add the driving
                # expression delta on top of the source expression.
                kp_e = x_s_c + exp_s + exp_d_rel
            else:
                raise ValueError(f"inference_mode {args.inference_mode} is not supported.")

            kp_rotated_s = torch.einsum('bmp,bkp->bkm', rot_mat_s, kp_e)
            kp_rotated_s *= scale_s
            x_d_i = kp_rotated_s + t_s
            out = self.wrapper.warp_decode(f_s, x_s, x_d_i)
            I_p_i = self.wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                I_p_i_pil = Image.fromarray(I_p_i)
                img_pil_resized = I_p_i_pil.resize((512, 512), Image.LANCZOS)
                I_p_pstbk = paste_back(np.array(img_pil_resized), source_M_c2o_lst[i], source_rgb_lst[i],
                                       mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        for step in range(args.n_frames):
            png_save_name = f"frame{step:04d}.png"
            cv2.imwrite(os.path.join(args.frames_dir, png_save_name),
                        I_p_pstbk_lst[step][:, :, ::-1])

        for step in range(args.n_frames):
            png_save_name = f"frame{step:04d}.png"
            cv2.imwrite(os.path.join(args.crop_output_dir, png_save_name),
                        I_p_lst[step][:, :, ::-1])

        img_crop_512_lst = [cv2.resize(img_item, (512, 512)) for img_item in img_crop_256x256_lst]
        drv_512_lst = [cv2.resize(img_item, (512, 512)) for img_item in driving_rgb_crop_256x256_lst]
        ref_512 = cv2.resize(ref_rgb_crop_256x256_lst[0], (512, 512))

        for step in range(args.n_frames):
            png_save_name = f"frame{step:04d}.png"
            cv2.imwrite(os.path.join(args.concat_output_dir, png_save_name),
                        np.concatenate((ref_512, drv_512_lst[step],
                                        img_crop_512_lst[step], I_p_lst[step]), axis=1)[:, :, ::-1])

        audio_src = args.src_audio if args.audio == 1 else args.drv_audio
        ffmpeg_tmpl = (
            f"ffmpeg -loglevel warning -framerate {source_fps} -i {{frames}}/frame%04d.png "
            f"-i {audio_src} -c:v libx264 -pix_fmt yuv420p -c:a aac "
            f"-map 0:v:0 -map 1:a:0? -shortest -y {{out}}")

        os.system(ffmpeg_tmpl.format(frames=args.frames_dir, out=args.output_path))
        os.system(ffmpeg_tmpl.format(
            frames=args.concat_output_dir,
            out=args.output_path.replace(f'.{args.suffix}', f'_concat.{args.suffix}')))
        os.system(ffmpeg_tmpl.format(
            frames=args.crop_output_dir,
            out=args.output_path.replace(f'.{args.suffix}', f'_crop.{args.suffix}')))

        shutil.rmtree(args.frames_dir, ignore_errors=True)
        shutil.rmtree(args.concat_output_dir, ignore_errors=True)
        shutil.rmtree(args.crop_output_dir, ignore_errors=True)


