import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch_dct as DCT
from einops import rearrange
from torch import nn, einsum

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Dct_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)  
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Dct_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        
        x = self.proj(x)
    
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class share_MLP(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.latent_dim = 64
        self.prompt_key_proj_d = nn.Linear(d_model, self.latent_dim)
        self.prompt_key_proj_u = nn.Linear(self.latent_dim, d_model)
        
        self.softmax = nn.Softmax(dim=-1)
        self.num = 10
        self.prompt = nn.Parameter(torch.zeros(1, self.num, d_model))
        trunc_normal_(self.prompt, std=.02)
        self.scale = d_model ** -0.5
        
        self.gelu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.prompt_key_proj_u(self.gelu(self.prompt_key_proj_d(x)))
        B, N, C = x.shape
        H, W = int(math.sqrt(N)),int(math.sqrt(N))
        channel = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        channel = self.avg(channel)
        score = self.sigmoid(channel).squeeze(-1).permute(0,2,1)
        cha_att = x*score

        #############
        prompt_attn = (self.prompt@ x.transpose(-2, -1)* self.scale)
        prompt_attn = self.softmax(prompt_attn) 
        prompt_out = (prompt_attn @ x)
        prompt = x + torch.cat([prompt_out,torch.zeros(x.shape[0],x.shape[1]-self.num,x.shape[-1], dtype=x.dtype, device=x.device)], dim=1)
        #############
        x = cha_att + prompt
        
        return x
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.dim = dim
        ###
        self.prompt_proj = share_MLP(self.dim)
        self.prompt_gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W)) + self.prompt_proj(x)*self.prompt_gate 
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        #_, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 decoder_embedding_dim=256, num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, 
                 drop_rate=0.,attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        
        # transformer decoder
        self.in_channels = embed_dims
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        self.decoder_embedding_dim = decoder_embedding_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.decoder_embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.decoder_embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.decoder_embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.decoder_embedding_dim)
        self.linear_pred = nn.Conv2d(self.decoder_embedding_dim*4, self.num_classes, kernel_size=1) 
        self.up = nn.Upsample(size=None, scale_factor=4, mode='bilinear', align_corners=False)
        
        #####
        self.DTC_conv1_1 = nn.Conv2d(320, 192, kernel_size=1, stride=1, bias=False, padding =0)
        self.DTC_high_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128*2, dropout=0.0)
        self.DTC_low_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128*2, dropout=0.0)
        
        self.DTC_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128*2, dropout=0.0)
        self.DTC_spatial = Transformer(dim=192, depth=1, heads=2, dim_head=64, mlp_dim=64*2, dropout=0.0)
        self.DTC_lnnorm = nn.LayerNorm(192)
        self.DTC_pos_embed_high = nn.Parameter(torch.zeros(1, 96, 256))
        self.DTC_pos_embed_low = nn.Parameter(torch.zeros(1, 96, 256))
            
    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        x = self.patch_embed1(x)
        B, N, C = x.shape
        H, W = int(math.sqrt(N)),int(math.sqrt(N))
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
       
        
        x = self.patch_embed2(x)
        B, N, C = x.shape
        H, W = int(math.sqrt(N)),int(math.sqrt(N))
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        #x, H, W = self.patch_embed3(x)
        x = self.patch_embed3(x)
        B, N, C = x.shape
        H, W = int(math.sqrt(N)),int(math.sqrt(N))
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        
         ##########
        feat_DCT= self.DTC_conv1_1(x)
        #feat_DCT = feat_DCT.float()
        num_batchsize,size = x.shape[0], x.shape[2]
        feat_DCT = DCT.dct_2d(x, norm='ortho')
        origin_feat_DCT = feat_DCT
        #print(origin_feat_DCT.shape)
        #feat_DCT = feat_DCT.half()
        
        feat_y = feat_DCT[:, 0:64, :, :] 
        feat_Cb = feat_DCT[:, 64:128, :, :] 
        feat_Cr = feat_DCT[:, 128:192, :, :]
        high = torch.cat([feat_y[:, 32:, :, :], feat_Cb[:, 32:, :, :], feat_Cr[:, 32:, :, :]], 1)
        low = torch.cat([feat_y[:, :32, :, :], feat_Cb[:, :32, :, :], feat_Cr[:, :32, :, :]], 1)
        b, n, h, w = high.shape
        #print(high.shape)
        high = torch.nn.functional.interpolate(high, size=(16, 16))
        low = torch.nn.functional.interpolate(low, size=(16, 16))
        high = rearrange(high, 'b n h w -> b n (h w)')
        low = rearrange(low, 'b n h w -> b n (h w)')
        #print(high.shape,low.shape)
        high = self.DTC_pos_embed_high + high
        low = self.DTC_pos_embed_low + low
        high = self.DTC_high_band(high)
        low = self.DTC_low_band(low)
        y_h, b_h, r_h = torch.split(high, 32, 1)
        y_l, b_l, r_l = torch.split(low, 32, 1)
        feat_y = torch.cat([y_l, y_h], 1)
        feat_Cb = torch.cat([b_l, b_h], 1)
        feat_Cr  = torch.cat([r_l, r_h], 1)
        feat_DCT = torch.cat((feat_y, feat_Cb, feat_Cr), 1) 
        feat_DCT = self.DTC_band(feat_DCT)
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = self.DTC_spatial(feat_DCT)
        feat_DCT = self.DTC_lnnorm (feat_DCT)
        ##########

        # stage 4
        #x, H, W = self.patch_embed4(x)
        x = self.patch_embed4(x)
        B, N, C = x.shape
        H, W = int(math.sqrt(N)),int(math.sqrt(N))
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs,feat_DCT

    def forward(self, x):
        
        [c1, c2, c3, c4],feat_DCT = self.forward_features(x)
        
        q1 = F.interpolate(c2, size=(32, 32),mode = 'bilinear').flatten(start_dim=2)
        q2 = F.interpolate(c2, size=(24, 24),mode = 'bilinear').flatten(start_dim=2)
        q3 = F.interpolate(c2, size=(16, 16),mode = 'bilinear').flatten(start_dim=2) 
        
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = nn.Upsample(size=c1.size()[2:], scale_factor=None, mode='bilinear', align_corners=False) (_c4)
        
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = nn.Upsample(size=c1.size()[2:], scale_factor=None, mode='bilinear', align_corners=False) (_c3)
        
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = nn.Upsample(size=c1.size()[2:], scale_factor=None, mode='bilinear', align_corners=False) (_c2)
        
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        
        
        x = self.linear_pred(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        '''
        q1 = F.interpolate(x, size=(32, 32),mode = 'bilinear').flatten(start_dim=2)
        q2 = F.interpolate(x, size=(24, 24),mode = 'bilinear').flatten(start_dim=2)
        q3 = F.interpolate(x, size=(16, 16),mode = 'bilinear').flatten(start_dim=2) 
        '''
        x = self.up (x)
        return x, [q1,q2,q3],feat_DCT

def Adapter_FRC_SegFormer_B4(image_size,num_classes):
    model = MixVisionTransformer(img_size=image_size, patch_size=4, in_chans=3, num_classes=num_classes, embed_dims=[64, 128, 320, 512],
                                 decoder_embedding_dim=768, num_heads=[1, 2, 5, 8], 
                                 mlp_ratios=[4, 4, 4, 4],qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                 depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],drop_rate=0.0, drop_path_rate=0.1)
    
    checkpoint = torch.load('./segformerb4_512x512ade160k.pth',map_location=torch.device('cpu'))
    new_checkpoint = {}
    for k in list(checkpoint.keys()):
        if k.startswith('backbone.'):
            new_checkpoint[k[len("backbone."):]] = checkpoint[k]
        if k.startswith('decode_head.'):
            new_checkpoint[k[len("decode_head."):]] = checkpoint[k]

    model_dict = model.state_dict()
    matched_dict = {k: v for k, v in new_checkpoint.items() if k in model_dict and v.shape==model_dict[k].shape}
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    print('matched keys:',len(matched_dict))
    
    return model

# x=torch.rand((1,3,512,512)) 
# model = Adapter_FRC_SegFormer_B4(512,2)
# print('model parameters: ', sum(p.numel() for p in model.parameters())/1e6,'M' )
# x, [q1,q2,q3],feat_DCT = model(x)
# print(feat_DCT.shape)
