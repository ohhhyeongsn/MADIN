import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange as rearrange

from src.modules import LayerNorm, spectral_norm
from src.attention import MADA

class MADIN(nn.Module):
    def __init__(self, 
                 ngf=64, num_block=[1, 2, 4, 6], img_size=[256, 128, 64, 32], heads=[1, 2, 4, 8],
                 strides=[8, 4, 2, 1], ksizes=[9, 7, 5, 3], offset_range_factor=[2, 2, 2, 2], groups=[1, 2, 4, 8], factor=2.66):
        super().__init__()

        self.start = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.GELU()
        )
        
        # Transformer on 256
        self.trane256 = nn.ModuleList([
            TransformerBlock(in_ch=ngf, fmap_size=img_size[0], n_groups=groups[0],
                             heads=heads[0], stride=strides[0],
                             offset_range_factor=offset_range_factor[0], ksize=ksizes[0],
                             expansion_factor=factor, use_mask=True)
            for _ in range(num_block[0])
        ])
        self.down128 = Downsample(num_ch=ngf)
        
        # Transformer on 128
        self.trane128 = nn.ModuleList([
            TransformerBlock(in_ch=ngf*2, fmap_size=img_size[1], n_groups=groups[1],
                             heads=heads[1], stride=strides[1],
                             offset_range_factor=offset_range_factor[1], ksize=ksizes[1],
                             expansion_factor=factor, use_mask=True)
            for _ in range(num_block[1])
        ])
        self.down64 = Downsample(num_ch=ngf*2)
        
        # Transformer on 64x64
        self.trane64 = nn.ModuleList([
            TransformerBlock(in_ch=ngf*4, fmap_size=img_size[2], n_groups=groups[2],
                             heads=heads[2], stride=strides[2],
                             offset_range_factor=offset_range_factor[2], ksize=ksizes[2],
                             expansion_factor=factor, use_mask=True)
            for _ in range(num_block[2])
        ])
        self.down32 = Downsample(num_ch=ngf*4)
        
        # Transformer on 32x32
        self.trane32 = nn.ModuleList([
            TransformerBlock(in_ch=ngf*8, fmap_size=img_size[3], n_groups=groups[3],
                             heads=heads[3], stride=strides[3],
                             offset_range_factor=offset_range_factor[3], ksize=ksizes[3],
                             expansion_factor=factor, use_mask=True)
            for _ in range(num_block[3])
        ])

        # Decoder up & fuse at 64
        self.up64   = Upsample(ngf*8)
        self.fuse64 = nn.Conv2d(in_channels=ngf*4*2, out_channels=ngf*4, kernel_size=1, stride=1, bias=False)
        self.trand64 = nn.ModuleList([
            TransformerBlock(in_ch=ngf*4, fmap_size=img_size[2], n_groups=groups[2],
                             heads=heads[2], stride=strides[2],
                             offset_range_factor=offset_range_factor[2], ksize=ksizes[2],
                             expansion_factor=factor, use_mask=False)
            for _ in range(num_block[2])
        ])

        # Decoder up & fuse at 128
        self.up128   = Upsample(ngf*4)
        self.fuse128 = nn.Conv2d(in_channels=ngf*2*2, out_channels=ngf*2, kernel_size=1, stride=1, bias=False)
        self.trand128 = nn.ModuleList([
            TransformerBlock(in_ch=ngf*2, fmap_size=img_size[1], n_groups=groups[1],
                             heads=heads[1], stride=strides[1],
                             offset_range_factor=offset_range_factor[1], ksize=ksizes[1],
                             expansion_factor=factor, use_mask=False)
            for _ in range(num_block[1])
        ])

        # Decoder up & fuse at 256
        self.up256   = Upsample(ngf*2)
        self.fuse256 = nn.Conv2d(in_channels=ngf*2, out_channels=ngf, kernel_size=1, stride=1, bias=False)
        self.trand256 = nn.ModuleList([
            TransformerBlock(in_ch=ngf, fmap_size=img_size[0], n_groups=groups[0],
                             heads=heads[0], stride=strides[0],
                             offset_range_factor=offset_range_factor[0], ksize=ksizes[0],
                             expansion_factor=factor, use_mask=False)
            for _ in range(num_block[0])
        ])

        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, padding=0)
        )

    def forward(self, x, mask=None):
        noise = torch.normal(mean=torch.zeros_like(x), std=torch.ones_like(x)*(1./128.))
        x = x + noise
        feature = torch.cat([x, mask], dim=1)
        feature256 = self.start(feature)

        for block in self.trane256:
            feature256 = block(feature256, mask)
        feature128, mask = self.down128(feature256, mask)
        for block in self.trane128:
            feature128 = block(feature128, mask)
        feature64, mask = self.down64(feature128, mask)
        for block in self.trane64:
            feature64 = block(feature64, mask)
        feature32, mask = self.down32(feature64, mask)
        for block in self.trane32:
            feature32 = block(feature32, mask)
        out64 = self.up64(feature32)
        out64 = self.fuse64(torch.cat([feature64, out64], dim=1))
        for block in self.trand64:
            out64  = block(out64)
        out128 = self.up128(out64)
        out128 = self.fuse128(torch.cat([feature128, out128], dim=1))
        for block in self.trand128:
            out128  = block(out128)
        out256 = self.up256(out128)
        out256 = self.fuse256(torch.cat([feature256, out256], dim=1))
        for block in self.trand256:
            out256  = block(out256)
        return torch.tanh(self.out(out256))



class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

    
class TransformerBlock(nn.Module):
    def __init__(self, 
                 in_ch, fmap_size, n_groups, 
                 heads, stride, offset_range_factor,
                 ksize, expansion_factor, use_mask
                 ):
        super().__init__()
        hc = in_ch // heads
        self.lpu = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.G_norm = LayerNorm(in_ch, 'WithBias')
        self.G_attn = MADA(fmap_size, heads, hc , n_groups,
                                         stride, offset_range_factor, ksize, use_mask)
        self.ffn = FeedForward(dim=in_ch, expansion_factor=expansion_factor,LayerNorm_type='WithBias')
        self.use_mask = use_mask
        
    def forward(self, x, mask=None):
        x = self.lpu(x) + x
        if self.use_mask:
            x1= self.G_attn(self.G_norm(x), mask)
        else: 
            x1= self.G_attn(self.G_norm(x))
        x1 = x1 + x
        x2 = self.ffn(x1) + x1
        
        return x2

class Downsample(nn.Module):
    def __init__(self, num_ch=32):
        super().__init__()

        # self.conv = PartialConv2d(in_channels=num_ch, out_channels=num_ch*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv = nn.Conv2d(in_channels=num_ch, out_channels=num_ch*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(num_features=num_ch*2, track_running_stats=False)
        self.act = nn.GELU()

    def forward(self, x, mask=None):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        if mask is not None:
            mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')
            return x, mask
        else:
            return x
        

class Upsample(nn.Module):
    def __init__(self, num_ch=32):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=num_ch, out_channels=num_ch//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(num_features=num_ch//2, track_running_stats=False)
        self.act = nn.GELU()

    def forward(self, x, mask = None):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
        
            
class FeedForward(nn.Module):
    def __init__(self, dim=64, expansion_factor=2.66,LayerNorm_type='WithBias'):
        super().__init__()

        num_ch = int(dim * expansion_factor)
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=num_ch*2, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=num_ch*2, out_channels=num_ch*2, kernel_size=3, stride=1, padding=1, groups=num_ch*2, bias=False)
        )
        self.linear = nn.Conv2d(in_channels=num_ch, out_channels=dim, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.norm(x)
        x1, x2 = self.conv(out).chunk(2, dim=1)
        out = F.gelu(x1) * x2
        out = self.linear(out)
        return out