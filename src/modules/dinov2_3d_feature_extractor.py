# coding: utf-8

"""
Appearance extractor(F) defined in paper, which maps the source image s to a 3D appearance feature volume.
"""
import os
import numpy as np

import torch
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
import timm

from .util_3d import ResBlock3d

class DinoWrapper_Isolated(L.LightningModule):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """
    def __init__(self, model_name: str, is_train: bool = False):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        self.freeze(is_train)

    def forward(self, image):
        # image: [N, C, H, W], on cpu
        # RGB image with [0,1] scale and properly size
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model.forward_features(self.processor(image))

        return outputs[:, 1:]

    def freeze(self, is_train: bool = False):
        print(f"======== image encoder is_train: {is_train} ========")
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = is_train

    @staticmethod
    def _build_dino(model_name: str):
        """Construct a DINO ViT-Base/16 backbone.

        Expected location (populated by ``scripts/download_weights.sh``):
            ``<repo_root>/pretrained_weights/<model_name>/pytorch_model.bin``

        For non-default layouts, set ``PERFORMRECAST_DINO_CKPT`` to the
        absolute path of the checkpoint. As a last resort, if no local file
        is found we fall back to ``timm``'s Hugging Face auto-download.
        """
        env_path = os.environ.get('PERFORMRECAST_DINO_CKPT')
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        default_path = os.path.join(
            repo_root, 'pretrained_weights', model_name, 'pytorch_model.bin')

        for ckpt in (env_path, default_path):
            if ckpt and os.path.exists(ckpt):
                model = timm.create_model(
                    model_name, pretrained=False,
                    checkpoint_path=ckpt, dynamic_img_size=True)
                break
        else:
            print(f'[DINO] Local checkpoint not found at {default_path}; '
                  f'falling back to Hugging Face Hub auto-download.')
            model = timm.create_model(
                model_name, pretrained=True, dynamic_img_size=True)

        data_config = timm.data.resolve_model_data_config(model)
        processor = transforms.Normalize(
            mean=data_config['mean'], std=data_config['std'])
        return model, processor


class Dinov2FeatureExtractor(nn.Module):
    def __init__(self, image_channel, block_expansion, num_down_blocks, max_features, reshape_channel,
                 reshape_depth, num_resblocks, dinov2_finetune):
        super(Dinov2FeatureExtractor, self).__init__()
        self.image_channel = image_channel
        self.block_expansion = block_expansion
        self.num_down_blocks = num_down_blocks
        self.max_features = max_features
        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth
        self.finetune = dinov2_finetune

        self.dinov2_img_encoder = DinoWrapper_Isolated(
            model_name='vit_base_patch16_224.dino',
            is_train=self.finetune)
        self.dinov2_feat_map_size = 32

        self.t_conv = nn.ConvTranspose2d(768, 768, 2, stride=2)
        self.conv = nn.Conv2d(in_channels=768, out_channels=max_features, kernel_size=1, stride=1)
        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        self.t_conv.apply(self.weights_init)
        self.conv.apply(self.weights_init)
        self.resblocks_3d.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, source_image):

        B, C, H, W = source_image.shape
        if not self.finetune:
            with torch.no_grad():
                img_feats = torch.einsum('blc->bcl', self.dinov2_img_encoder(source_image))
        else:
            img_feats = torch.einsum('blc->bcl', self.dinov2_img_encoder(source_image))

        token_size = int(np.sqrt(H * W / img_feats.shape[-1]))
        img_feats = img_feats.reshape(*img_feats.shape[:2], H // token_size, W // token_size)
        img_feats = F.gelu(self.t_conv(img_feats, output_size=(64, 64)))
        out = self.conv(img_feats)

        _, c, h, w = out.shape
        f_s = out.view(B, self.reshape_channel, self.reshape_depth, h, w)  # ->Bx32x16x64x64
        f_s = self.resblocks_3d(f_s)  # ->Bx32x16x64x64
        return f_s

