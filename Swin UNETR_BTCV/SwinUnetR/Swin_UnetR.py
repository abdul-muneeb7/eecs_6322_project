import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "SwinUNETR",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]

from SwinUnetR import Swin_Transformer_Block
SwinTransformerBlock = Swin_Transformer_Block.SwinTransformerBlock

from SwinUnetR.comon import window_partition
from SwinUnetR.comon import get_window_size
from SwinUnetR.comon import window_reverse
from SwinUnetR.comon import compute_mask



from SwinUnetR import Patch_Merging
PatchMergingV2 = Patch_Merging.PatchMergingV2
PatchMerging = Patch_Merging.PatchMerging



MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}



__all__ = [
    "SwinUNETR",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]


#====================================================================

# Here we implement the 3 core classes to build SwinUnetR Architecture
#  1) SwinUNETR
    # Here we define the encoder and decoer blocks of UnetR
#  2) SwinTransformer
#  3) BasicLayer

 
# To Implement these we use resources from
# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
# Paper: https://arxiv.org/abs/2103.14030
# Code: https://github.com/microsoft/Swin-Transformer
 
#====================================================================


class SwinUNETR(nn.Module):


    def __init__(
            self,
            img_size: Union[Sequence[int], int],
            in_channels: int,
            out_channels: int,
            depths: Sequence[int] = (2, 2, 2, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            feature_size: int = 24,
            norm_name: Union[Tuple, str] = "instance", #feature normalization
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            normalize: bool = True,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            downsample="merging",
        ) -> None:
      

        # For BTCV we use
        # # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
        # >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

        # For BraTS we use
        # # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
        # >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))


            super().__init__()

            img_size = ensure_tuple_rep(img_size, spatial_dims)
            patch_size = ensure_tuple_rep(2, spatial_dims)
            window_size = ensure_tuple_rep(7, spatial_dims)

            if spatial_dims not in (2, 3):
                raise ValueError("spatial dimension should be 2 or 3.")

            for m, p in zip(img_size, patch_size):
                for i in range(5):
                    if m % np.power(p, i + 1) != 0:
                        raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

            if not (0 <= drop_rate <= 1):
                raise ValueError("dropout rate should be between 0 and 1.")

            if not (0 <= attn_drop_rate <= 1):
                raise ValueError("attention dropout rate should be between 0 and 1.")

            if not (0 <= dropout_path_rate <= 1):
                raise ValueError("drop path rate should be between 0 and 1.")

            if feature_size % 12 != 0:
                raise ValueError("feature_size should be divisible by 12.")

            self.normalize = normalize

            self.swinViT = SwinTransformer(
                in_chans=in_channels,
                embed_dim=feature_size,
                window_size=window_size,
                patch_size=patch_size,
                depths=depths,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dropout_path_rate,
                norm_layer=nn.LayerNorm,
                use_checkpoint=use_checkpoint,
                spatial_dims=spatial_dims,
                downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            )

            self.encoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )

            self.encoder2 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )

            self.encoder3 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=2 * feature_size,
                out_channels=2 * feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )

            self.encoder4 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=4 * feature_size,
                out_channels=4 * feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )

            self.encoder10 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=16 * feature_size,
                out_channels=16 * feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )

            self.decoder5 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=16 * feature_size,
                out_channels=8 * feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

            self.decoder4 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

            self.decoder3 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

            self.decoder1 = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

            self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)


    def load_from(self, weights):

            with torch.no_grad():
                self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
                self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
                for bname, block in self.swinViT.layers1[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers1")
                self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                    weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
                )
                self.swinViT.layers1[0].downsample.norm.weight.copy_(
                    weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
                )
                self.swinViT.layers1[0].downsample.norm.bias.copy_(
                    weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
                )
                for bname, block in self.swinViT.layers2[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers2")
                self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                    weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
                )
                self.swinViT.layers2[0].downsample.norm.weight.copy_(
                    weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
                )
                self.swinViT.layers2[0].downsample.norm.bias.copy_(
                    weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
                )
                for bname, block in self.swinViT.layers3[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers3")
                self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                    weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
                )
                self.swinViT.layers3[0].downsample.norm.weight.copy_(
                    weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
                )
                self.swinViT.layers3[0].downsample.norm.bias.copy_(
                    weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
                )
                for bname, block in self.swinViT.layers4[0].blocks.named_children():
                    block.load_from(weights, n_block=bname, layer="layers4")
                self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                    weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
                )
                self.swinViT.layers4[0].downsample.norm.weight.copy_(
                    weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
                )
                self.swinViT.layers4[0].downsample.norm.bias.copy_(
                    weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
                )

    def forward(self, x_in):
            hidden_states_out = self.swinViT(x_in, self.normalize)
            enc0 = self.encoder1(x_in)
            enc1 = self.encoder2(hidden_states_out[0])
            enc2 = self.encoder3(hidden_states_out[1])
            enc3 = self.encoder4(hidden_states_out[2])
            dec4 = self.encoder10(hidden_states_out[4])
            dec3 = self.decoder5(dec4, hidden_states_out[3])
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            dec0 = self.decoder2(dec1, enc1)
            out = self.decoder1(dec0, enc0)
            logits = self.out(out)
            return logits



class SwinTransformer(nn.Module):

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",  # merging and  mergingV2
    ) -> None:
    

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]


class BasicLayer(nn.Module):
 
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0, #  ratio of mlp hidden dim to embedding dim
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ) -> None:

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x
