# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import math
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn_ce_init

from scipy.signal import stft

from pavsodr_model.modeling.audio_decoder import Audio_Unmixer
from pavsodr_model.modeling.audio_decoder import Audio_Visual_Fusion_Seq

from pavsodr_model.modeling.audio_encoder.soundnet import SoundNet
from pavsodr_model.modeling.audio_encoder.htsat import HTSAT


# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
                 ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn_ce_init):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn_ce_init(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


def get_theta_phi(_x, _y, _z):
    dv = math.sqrt(_x * _x + _y * _y + _z * _z)
    x = _x / dv
    y = _y / dv
    z = _z / dv
    theta = math.atan2(y, x)
    phi = math.asin(z)

    return theta, phi


def cube2equi(x, y, side, cw, W, H):
    u = 2 * (float(x) / cw - 0.5)
    v = 2 * (float(y) / cw - 0.5)

    if side == "front":
        theta, phi = get_theta_phi(1, u, v)
    elif side == "right":
        theta, phi = get_theta_phi(-u, 1, v)
    elif side == "left":
        theta, phi = get_theta_phi(u, -1, v)
    elif side == "back":
        theta, phi = get_theta_phi(-1, -u, v)
    elif side == "bottom":
        theta, phi = get_theta_phi(-v, u, 1)
    elif side == "top":
        theta, phi = get_theta_phi(v, u, -1)

    _u = 0.5 + 0.5 * (theta / math.pi)
    _v = 0.5 + (phi / math.pi)

    return _u * W, _v * H


def equi2cube(u, v, cw, W, H):
    theta = (u / W - 0.5) * 2 * math.pi
    phi = (v / H - 0.5) * math.pi

    x = math.cos(phi) * math.sin(theta)
    y = math.sin(phi)
    z = math.cos(phi) * math.cos(theta)

    abs_x = abs(x)
    abs_y = abs(y)
    abs_z = abs(z)

    is_x_positive = x > 0
    is_y_positive = y > 0
    is_z_positive = z > 0

    if abs_x >= abs_y and abs_x >= abs_z:
        if is_x_positive:
            xx, yy, side = -z, y, "right"
        else:
            xx, yy, side = z, y, "left"
        max_axis = abs_x
    elif abs_y >= abs_x and abs_y >= abs_z:
        if is_y_positive:
            xx, yy, side = x, -z, "bottom"
        else:
            xx, yy, side = x, z, "top"
        max_axis = abs_y
    else:
        if is_z_positive:
            xx, yy, side = x, y, "front"
        else:
            xx, yy, side = -x, y, "back"
        max_axis = abs_z

    xx /= max_axis
    yy /= max_axis

    cube_x = (xx + 1) / 2 * cw
    cube_y = (yy + 1) / 2 * cw

    return cube_x, cube_y, side


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, n_head, n_points, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        current_directory = os.path.dirname(os.path.abspath(__file__))
        reference_points_list_cube = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            reference_points_pth = os.path.join(current_directory, "reference_points.pt")
            if not os.path.exists(reference_points_pth):
                open(reference_points_pth, 'w').close()
            if os.path.getsize(reference_points_pth) > 0:
                loaded_dict = torch.load(reference_points_pth)
            else:
                loaded_dict = {}

            if (str(H_.item()) + "x" + str(W_.item())) in loaded_dict:
                reference_points_list_cube.append(loaded_dict[str(H_.item()) + "x" + str(W_.item())])
            else:
                ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                              torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
                ref = torch.stack((ref_x, ref_y), dim=-1)

                cube_refs = []
                for x in range(ref.shape[0]):
                    for y in range(ref.shape[1]):
                        cube_x, cube_y, side = equi2cube(ref[x, y, 0], ref[x, y, 1], W_ // 4, W_, H_)
                        if side == "top":    cube_refs.append({"type": "top", "coords": (cube_x, cube_y)})
                        if side == "bottom": cube_refs.append({"type": "bottom", "coords": (cube_x, cube_y)})
                        if side == "left":   cube_refs.append({"type": "left", "coords": (cube_x, cube_y)})
                        if side == "right":  cube_refs.append({"type": "right", "coords": (cube_x, cube_y)})
                        if side == "front":  cube_refs.append({"type": "front", "coords": (cube_x, cube_y)})
                        if side == "back":   cube_refs.append({"type": "back", "coords": (cube_x, cube_y)})

                reference_points_list_cube_pre = []
                for i_point in range(n_points):
                    thetas = torch.arange(n_head, dtype=torch.float32) * (2.0 * math.pi / n_head)
                    grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
                    grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                    grid_init = grid_init * (i_point + 1)

                    cube_refs_ = cube_refs.copy()
                    for i, cube_ref_ in enumerate(cube_refs_):
                        cube_coors = [torch.tensor((max(0.0, min(cube_ref_["coords"][0] + dx, W_ // 4)),
                                                    max(0.0, min(cube_ref_["coords"][1] + dy, W_ // 4))),
                                                   device=device).unsqueeze(dim=0) for dx, dy in grid_init]
                        cube_refs_[i] = {"type": cube_ref_["type"], "coords": torch.cat(cube_coors, dim=0)}

                    equi_coors = []
                    for i in range(n_head):
                        equi_coors_aux = []
                        for cube_ref_ in cube_refs_:
                            equi_x, equi_y = cube2equi(cube_ref_["coords"][i][0], cube_ref_["coords"][i][1],
                                                       cube_ref_["type"], W_ // 4, W_, H_)
                            equi_coors_aux.append(torch.tensor((equi_x, equi_y), device=device).unsqueeze(dim=0))
                        equi_coors.append(torch.cat(equi_coors_aux, dim=0).unsqueeze(dim=0))

                    equi_coors = torch.cat(equi_coors, dim=0)  # [8, 450, 2]
                    equi_coors[..., 0] = equi_coors[..., 0] / W_
                    equi_coors[..., 1] = equi_coors[..., 1] / H_

                    equi_coors = equi_coors[:, :, None, :]
                    reference_points_list_cube_pre.append(equi_coors)

                reference_points_list_cube_pre = torch.cat(reference_points_list_cube_pre, dim=2)
                reference_points_list_cube.append(reference_points_list_cube_pre)

                loaded_dict[str(H_.item()) + "x" + str(W_.item())] = reference_points_list_cube_pre
                with open(reference_points_pth, 'wb') as file:
                    torch.save(loaded_dict, file)
                    file.close()

        reference_points_cube = torch.cat(reference_points_list_cube, dim=1)
        reference_points = reference_points[:, :, None, :, None, :].repeat(1, 1, n_head, 1, 2, 1)
        reference_points_cube = reference_points_cube.permute(1, 0, 2, 3)[None, :, :, None, :, :]. \
            repeat(valid_ratios.shape[0], 1, 1, valid_ratios.shape[1], 1, 1)
        reference_points_cube = reference_points_cube[:, :, :, :, :2, :]

        reference_points = torch.cat((reference_points, reference_points_cube), dim=4)

        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, 8, 4, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder_Final(nn.Module):
    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            transformer_dropout: float,
            transformer_nheads: int,
            transformer_dim_feedforward: int,
            transformer_enc_layers: int,
            conv_dim: int,
            mask_dim: int,
            norm: Optional[Union[str, Callable]] = None,
            # deformable transformer encoder args
            transformer_in_features: List[str],
            common_stride: int,
            audio_encoder_name: str = "soundnet",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]

        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.audio_encoder_name = audio_encoder_name
        if audio_encoder_name == "soundnet":
            self.audio_encoder = SoundNet()
            self.audio_proj = nn.Conv2d(1024, 256, 1)
            self.av_sp_fus = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 256, 3, stride=1, padding=1),
                                                          nn.BatchNorm2d(256),
                                                          nn.ReLU(inplace=True))] * 3)
            self.audio_sep = Audio_Unmixer(num_queries=50, in_channels=1024, dim=256, depth=3)
        elif audio_encoder_name == "htsat":
            self.audio_encoder = HTSAT()
            self.audio_proj = nn.Conv2d(768, 256, 1)
            self.av_sp_fus = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 256, 3, stride=1, padding=1),
                                                          nn.BatchNorm2d(256),
                                                          nn.ReLU(inplace=True))] * 3)
            self.audio_sep = Audio_Unmixer(num_queries=50, in_channels=768, dim=256, depth=3)

        self.audio_visual_ca = nn.ModuleList([Audio_Visual_Fusion_Seq(depth=2, dim=256, d_k=256, d_v=256, h=8, mlp_dim=256)] * 3)

        self.audio_visual_post = nn.ModuleList([nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1),
                                                              nn.BatchNorm2d(256),
                                                              nn.ReLU(inplace=True))] * 3)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        ret["audio_encoder_name"] = cfg.AUDIO_ENCODER_NAME
        return ret

    @autocast(enabled=False)
    def forward_features(self, features, audios, images):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # audio
        out_av = []
        audio_sep = None
        if self.audio_encoder_name == "soundnet":
            audio_features = []
            for b in range(len(audios)):
                audio_feature = self.audio_encoder(audios[b])
                audio_features.append(audio_feature)
                audio_features.append(audio_feature)
            audio_features = torch.stack(audio_features).unsqueeze(1)  # [batch * frames, 1, 1, 1024]
            audio_features = audio_features.permute(0, 3, 1, 2)

            audio_features_proj = self.audio_proj(audio_features)  # 4, 1024, 1, 1
            audio_sep = self.audio_sep(audio_features)

            for i, features_res in enumerate(out, start=0):
                audio_features_proj_res = audio_features_proj.repeat(1, 1, features_res.shape[2], features_res.shape[3])
                features_av = features_res + audio_features_proj_res
                features_av = torch.cat((features_av, features_res), dim=1)
                features_av = self.av_sp_fus[i](features_av)

                audio_sep, features_av = self.audio_visual_ca[i](features_av, audio_sep, images)
                out_av.append(features_av)
        elif self.audio_encoder_name == "htsat":
            audio_features = []
            for b in range(len(audios)):
                audio_feature = self.audio_encoder(audios[b][0])
                audio_features.append(audio_feature)
                audio_features.append(audio_feature)
            audio_features = torch.stack(audio_features).unsqueeze(1)  # [batch * frames, 1, 1, 1024]
            audio_features = audio_features.permute(0, 3, 1, 2)

            audio_features_proj = self.audio_proj(audio_features)  # 4, 1024, 1, 1
            audio_sep = self.audio_sep(audio_features)

            for i, features_res in enumerate(out, start=0):
                audio_features_proj_res = audio_features_proj.repeat(1, 1, features_res.shape[2], features_res.shape[3])
                features_av = features_res + audio_features_proj_res
                features_av = torch.cat((features_av, features_res), dim=1)
                features_av = self.av_sp_fus[i](features_av)

                audio_sep, features_av = self.audio_visual_ca[i](features_av, audio_sep, images)
                out_av.append(features_av)

        # post processing:
        for i, features_res_av in enumerate(out_av, start=0):
            f_av = out[i] + features_res_av
            out[i] = self.audio_visual_post[i](f_av)

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), out[0], multi_scale_features, audio_sep


def spectrum_fast(x, nperseg=512, noverlap=128, window='hamming', cut_dc=True,
                  output_phase=True, cut_last_timeframe=True):
    f, t, seg_stft = stft(x, window=window, nperseg=nperseg, noverlap=noverlap)
    output = np.abs(seg_stft)
    if output_phase:
        phase = np.angle(seg_stft)
        output = np.concatenate((output, phase), axis=-3)
    if cut_dc:
        output = output[:, 1:, :]
    if cut_last_timeframe:
        output = output[:, :, :-1]
    return output

