import torch
import timm
from segmentation_models_pytorch import Unet
import torch.nn as nn
from models.vit import CrossAttentionBlock,Block
from models.resnet import BasicBlock,Bottleneck
from models.unet import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

torch_version = torch.__version__
is_torch2 = torch_version.startswith('2.')


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def dwt_init(self,x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return x_LL, x_HL, x_LH, x_HH

    def forward(self, x):
        return self.dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def iwt_init(self,x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, :out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

        h = torch.zeros([out_batch, out_channel, out_height,
                         out_width]).float().to(x.device)

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

    def forward(self, x):
        return self.iwt_init(x)


class DualUNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=6, num_classes=5, dim=768, num_heads=12,depths_vit=[2,2,2],
                 depths_convnextv2=[3,4,6], dims_convnextv2=[96, 192, 384, 768],
                 num_blocks=16, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),head_init_scale=1.):
        super().__init__()

        self.dims_convnextv2 = dims_convnextv2

        NUM0 = 0
        NUM1 = depths_vit[0]
        NUM2 = NUM1 + depths_vit[1]
        NUM3 = NUM2 + depths_vit[2]
        num_blocks = NUM3 + 4

        # Patch Embedding
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, int(dim)))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]

        for depth in range(len(depths_vit)):
            start_idx = eval(f'NUM{depth}')
            end_idx = eval(f'NUM{depth + 1}')
            self.add_module('vit_stage_' + str(depth+1),
                            nn.Sequential(*[
                                Block(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,self.trans_dpr[i], drop_path_rate, norm_layer=norm_layer)
                                for i in range(start_idx, end_idx)])
                            )
            self.add_module('CrossAttentionBlock_' + str(depth + 1),
                                CrossAttentionBlock(encoder_dim=dim, decoder_dim=dim, num_heads=12, mlp_ratio=4.,
                                qkv_bias=False, qk_scale=None, drop=0.0,attn_drop=0., drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                self_attn=True))

        for i in range(4):
            self.add_module('vit_view_' + str(i),
                Block(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, self.trans_dpr[i+NUM3], drop_path_rate,norm_layer=norm_layer)
            )
            self.add_module('cnn_view_' + str(i),
                            nn.Sequential(*[
                                Bottleneck(dims_convnextv2[-1], dims_convnextv2[-1], stride=1, downsample=None),
                                LayerNorm(dims_convnextv2[-1], eps=1e-6, data_format="channels_first"),
                                nn.Conv2d(dims_convnextv2[-1], dims_convnextv2[-1], kernel_size=2, stride=2)
                            ])
                            )

        self.cnn_stem = nn.Sequential(
            nn.Conv2d(in_channels, dims_convnextv2[0], kernel_size=4, stride=2,padding=1),
            LayerNorm(dims_convnextv2[0], eps=1e-6, data_format="channels_first")
        )

        self.stages = nn.ModuleList()
        for i in range(len(depths_convnextv2)):
            modules = []
            modules.extend([Bottleneck(dims_convnextv2[i], dims_convnextv2[i], stride=1, downsample=None) for j in range(depths_convnextv2[i])])
            if i < len(depths_convnextv2):
                modules.append(LayerNorm(dims_convnextv2[i], eps=1e-6, data_format="channels_first"))
                modules.append(nn.Conv2d(dims_convnextv2[i], dims_convnextv2[i + 1], kernel_size=2, stride=2))
            self.add_module('cnn_stage_' + str(i + 1), nn.Sequential(*modules))

        self.adaptor1 = nn.Conv2d(dims_convnextv2[1], dim, kernel_size=4, stride=4, padding=0)
        self.adaptor2 = nn.Conv2d(dims_convnextv2[2], dim, kernel_size=2, stride=2, padding=0)
        self.adaptor3 = nn.Conv2d(dims_convnextv2[3], dim, kernel_size=1, stride=1, padding=0)

        self.unet = UNet(n_channels=(in_channels - 3)*4, dims= [i // 2 for i in dims_convnextv2],
                         n_classes= (in_channels - 3)*4, bilinear=False)
        self.dwt = DWT()
        self.iwt = IWT()

        self.norm = norm_layer(dim) #
        self.norm = nn.LayerNorm(dims_convnextv2[-1], eps=1e-6)
        self.head_vit = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_cnn = nn.Linear(dims_convnextv2[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.head_cnn.weight.data.mul_(head_init_scale)
        self.head_cnn.bias.data.mul_(head_init_scale)
        self.alpha = nn.Parameter(torch.full((1,), 1.0))

        self.apply(self._init_weights)
        self.LL_pooling = nn.AvgPool2d(kernel_size=2, stride=2)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, x):
        B, C, H, W = x.shape
        B = B // 4
        mask = x[:,3:,:,:]
        _,c,_,_ = mask.shape
        mask = mask.reshape(B,4*c,H,W)

        fea_t = self.patch_embed(x)
        fea_t = rearrange(fea_t, 'b d h w -> b (h w) d')
        cls_tokens = self.cls_token.expand(4 * B, -1, -1)
        fea_t = torch.cat((cls_tokens, fea_t), dim=1)
        fea_t = fea_t + self.pos_embed
        B,L,D = fea_t.size()

        fea_c = self.cnn_stem(x)
        fea_t1 = self.vit_stage_1(fea_t)
        fea_c1 = self.cnn_stage_1(fea_c)
        _fea_c1 =self.adaptor1(fea_c1).reshape(B,D,L-1).permute(0,2,1)
        fea_1 = self.CrossAttentionBlock_1(fea_t1,_fea_c1)

        fea_t2 = self.vit_stage_2(fea_1)
        fea_c2 = self.cnn_stage_2(fea_c1)
        _fea_c2 = self.adaptor2(fea_c2).reshape(B, D, L - 1).permute(0, 2, 1)
        fea_2 = self.CrossAttentionBlock_2(fea_t2, _fea_c2)

        fea_t3 = self.vit_stage_3(fea_2)
        fea_c3 = self.cnn_stage_3(fea_c2)
        _fea_c3 = self.adaptor3(fea_c3).reshape(B, D, L - 1).permute(0, 2, 1)
        fea_3 = self.CrossAttentionBlock_3(fea_t3, _fea_c3)

        fea_tv = fea_3.reshape(B // 4, 4, L, D)
        cls_out_vit = torch.zeros_like(fea_tv[:, 0, 0])
        B_, C_, H_, W_ = fea_c3.size()
        fea_c3 = fea_c3.reshape(B_ // 4, 4, C_, H_, W_)
        mv_c = []
        for i in range(4):
            _fea_tv = eval('self.vit_view_' + str(i))(fea_tv[:, i, :, :].squeeze())
            _fea_tv = self.norm(_fea_tv)
            _cls = _fea_tv[:, 0]
            cls_out_vit += _cls
            _fea_ce = eval('self.cnn_view_' + str(i))(fea_c3[:, i, :, :, :].squeeze())
            mv_c.append(_fea_ce)
        x_c = torch.stack(mv_c, 1)
        x_c = rearrange(x_c, "b (nh nw) c h w -> b c (nh h) (nw w)", nh=2, nw=2)
        mask_encoder, mask_out = self.unet(mask)
        x_LL, x_HL, x_LH, x_HH = self.dwt(mask_encoder)
        c_LL, c_HL, c_LH, c_HH = self.dwt(x_c)
        x_HL = torch.mean(x_HL, dim=0).unsqueeze(0).repeat(768, 1, 1, 1)
        x_LH = torch.mean(x_LH, dim=0).unsqueeze(0).repeat(768, 1, 1, 1)
        x_HH = torch.mean(x_HH, dim=0).unsqueeze(0).repeat(768, 1, 1, 1)
        HL = F.conv2d(x_c,x_HL, stride=2, padding=3)
        LH = F.conv2d(x_c, x_LH, stride=2, padding=3)
        HH = F.conv2d(x_c, x_HH, stride=2, padding=3)
        idwt = torch.cat((c_LL, HL + c_HL, LH + c_LH, HH + c_HH), dim=1)
        x_c = self.iwt(idwt)
        x_c = self.pooling(x_c).flatten(1)
        conv_cls = self.head_cnn(x_c)
        vit_cls = self.head_vit(cls_out_vit)
        out = vit_cls + self.alpha *  conv_cls
        mask_out = rearrange(mask_out, "b (v c) h w -> (b v) c h w", v = 4)
        return out,mask_out


















