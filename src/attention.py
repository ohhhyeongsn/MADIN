import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.modules import TMOE, TOE

class MADA(nn.Module):
    def __init__(
        self,
        q_size,
        n_heads,
        n_head_channels,
        n_groups,
        stride,
        offset_range_factor,
        ksize,
        use_mask,
    ):
        super().__init__()
        self.stride = stride
        self.ksize = ksize
        self.n_head_channels = n_head_channels
        self.scale = n_head_channels**-0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size, q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.offset_range_factor = offset_range_factor
        self.padding = ksize // 2
        self.use_mask = use_mask

        if self.use_mask:
            self.offset_extractor = TMOE(
                self.n_group_channels, ksize, stride, self.padding
            )
        else:
            self.offset_extractor = TOE(
                self.n_group_channels, ksize, stride, self.padding
            )

        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

    def mask_based_range_factor(self, mask):
        p = self.padding

        mask_local = F.pad(mask, (p, p, p, p), mode="reflect")
        mask_local = F.avg_pool2d(
            mask_local, kernel_size=self.ksize, stride=self.stride, padding=0
        )

        mask_down = F.pad(mask, (p, p, p, p), mode="reflect")
        mask_down = F.avg_pool2d(
            mask_down, kernel_size=self.ksize, stride=self.stride, padding=0
        )

        mask_coarse = F.pad(mask_down, (p, p, p, p), mode="reflect")
        mask_coarse = F.avg_pool2d(
            mask_coarse, kernel_size=self.ksize, stride=1, padding=0
        )
        mask_degree = (mask_local + mask_coarse) / 2.0

        factor = self.offset_range_factor + (1.0 - mask_degree) * 1

        return factor

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing="ij",
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, x, mask=None):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = rearrange(
            q, "b (g c) h w -> (b g) c h w", g=self.n_groups, c=self.n_group_channels
        )

        if self.use_mask:
            mask_exp = (
                mask.unsqueeze(1)
                .expand(-1, self.n_groups, -1, -1, -1)
                .reshape(B * self.n_groups, 1, H, W)
            )
            q_offset_feat = self.offset_extractor(q_off, mask_exp)
        else:
            q_offset_feat = self.offset_extractor(q_off)

        Hk, Wk = q_offset_feat.size(2), q_offset_feat.size(3)
        offset_range = torch.tensor(
            [1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device
        ).reshape(1, 2, 1, 1)
        q_offset_feat = q_offset_feat.tanh().mul(offset_range)

        if self.use_mask:
            m_off_range_f = self.mask_based_range_factor(mask)
            m_off_range_f = m_off_range_f.repeat_interleave(self.n_groups, dim=0)
            q_offset_feat = q_offset_feat.mul(m_off_range_f)
        else:
            q_offset_feat = q_offset_feat.mul(self.offset_range_factor)


        q_offset_feat = rearrange(q_offset_feat, "b p h w -> b h w p")


        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        pos = q_offset_feat + reference
        n_sample = Hk * Wk

        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],
            mode="bilinear",
            align_corners=True,
        )
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(
            B * self.n_heads, self.n_head_channels, n_sample
        )
        v = self.proj_v(x_sampled).reshape(
            B * self.n_heads, self.n_head_channels, n_sample
        )

        attn = torch.einsum("b c m, b c n -> b m n", q, k)
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        out = torch.einsum("b m n, b c n -> b c m", attn, v)
        out = out.reshape(B, C, H, W)
        y = self.proj_out(out)
        return y
