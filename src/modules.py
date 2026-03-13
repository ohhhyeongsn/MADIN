import numbers
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange as rearrange

    
#Twostage_Maskware_OffsetExtractor
class TMOE(nn.Module): 
    def __init__(self, n_group_channels, ksize, stride, padding):
        super().__init__()
        self.n_group_channels = n_group_channels
        self.fusion = nn.Conv2d(n_group_channels+1, n_group_channels, kernel_size=1, stride=1, bias=False)
        
        self.mask_large_offset_extractor = nn.Sequential(
            nn.Conv2d(n_group_channels, n_group_channels, kernel_size=ksize, stride=stride,
                    padding=padding, groups=n_group_channels, bias=False),
            LayerNorm(n_group_channels, 'WithBias'),
            nn.GELU(),
        )
        
        self.mask_small_offset_extractor = nn.Sequential(
            nn.AvgPool2d(kernel_size=ksize, stride=stride, padding=padding),
            nn.Conv2d(n_group_channels, n_group_channels, kernel_size=ksize, stride=1,
                    padding=padding, groups=n_group_channels, bias=False),
            LayerNorm(n_group_channels, 'WithBias'),
            nn.GELU(),
        )
        
        self.to_offset = nn.Conv2d(n_group_channels*2, 2, 1, 1, 0, bias=False)
        
    def forward(self, x, mask=None):
        x = self.fusion(torch.cat([x, mask], dim=1))
        
        x1 = self.mask_large_offset_extractor(x)
        x2 = self.mask_small_offset_extractor(x)
        
        x = self.to_offset(torch.cat([x1, x2], dim=1))
        
        return x
    
class TOE(nn.Module): #Twostage_OffsetExtractor
    def __init__(self, n_group_channels, ksize, stride, padding):
        super().__init__()
        self.n_group_channels = n_group_channels
        
        self.mask_large_offset_extractor = nn.Sequential(
            nn.Conv2d(n_group_channels, n_group_channels, kernel_size=ksize, stride=stride,
                    padding=padding, groups=n_group_channels, bias=False),
            LayerNorm(n_group_channels, 'WithBias'),
            nn.GELU(),
        )
        
        self.mask_small_offset_extractor = nn.Sequential(
            nn.AvgPool2d(kernel_size=ksize, stride=stride, padding=padding),
            nn.Conv2d(n_group_channels, n_group_channels, kernel_size=ksize, stride=1,
                    padding=padding, groups=n_group_channels, bias=False),
            LayerNorm(n_group_channels, 'WithBias'),
            nn.GELU(),
        )
        
        self.to_offset = nn.Conv2d(n_group_channels*2, 2, 1, 1, 0, bias=False)
        
    def forward(self, x):
        
        x1 = self.mask_large_offset_extractor(x)
        x2 = self.mask_small_offset_extractor(x)
        
        x = self.to_offset(torch.cat([x1, x2], dim=1))
        
        return x
    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
