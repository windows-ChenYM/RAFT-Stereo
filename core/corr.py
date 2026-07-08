import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler

try:
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = corr_sampler.forward(volume, coords, radius)
        return corr
    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = corr_sampler.backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None

class CorrBlockFast1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        # all pairs correlation
        corr = CorrBlockFast1D.corr(fmap1, fmap2)
        batch, h1, w1, dim, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, 1, w2)
        for i in range(self.num_levels):
            self.corr_pyramid.append(corr.view(batch, h1, w1, -1, w2//2**i))
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])

    def __call__(self, coords):
        out_pyramid = []
        bz, _, ht, wd = coords.shape
        coords = coords[:, [0]]
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i].squeeze(3), coords/2**i, self.radius)
            out_pyramid.append(corr.view(bz, -1, ht, wd))
        return torch.cat(out_pyramid, dim=1)

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / (D ** 0.5)


class PytorchAlternateCorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.fmap1 = fmap1
        self.fmap2 = fmap2

    def corr(self, fmap1, fmap2, coords):
        B, D, H, W = fmap2.shape
        # map grid coordinates to [-1,1]
        xgrid, ygrid = coords.split([1,1], dim=-1)
        xgrid = 2*xgrid/(W-1) - 1
        ygrid = 2*ygrid/(H-1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        output_corr = []
        for grid_slice in grid.unbind(3):
            fmapw_mini = F.grid_sample(fmap2, grid_slice, align_corners=True)
            corr = torch.sum(fmapw_mini * fmap1, dim=1)
            output_corr.append(corr)
        corr = torch.stack(output_corr, dim=1).permute(0,2,3,1)

        return corr / (D ** 0.5)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        fmap1 = self.fmap1
        fmap2 = self.fmap2
        out_pyramid = []
        for i in range(self.num_levels):
            dx = torch.zeros(1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)
            centroid_lvl = coords.reshape(batch, h1, w1, 1, 2).clone()
            centroid_lvl[...,0] = centroid_lvl[...,0] / 2**i
            coords_lvl = centroid_lvl + delta.view(-1, 2)
            corr = self.corr(fmap1, fmap2, coords_lvl)
            fmap2 = F.avg_pool2d(fmap2, [1, 2], stride=[1, 2])
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous()
        if not torch.jit.is_tracing():
            out = out.float()
        return out


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, 1, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        out = out.permute(0, 3, 1, 2).contiguous()
        if not torch.jit.is_tracing():
            out = out.float()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / (D ** 0.5)


class CorrBlock1DNoGridSample:
    """Correlation lookup without grid_sample/gather for fixed-shape ncnn export.

    This mirrors CorrBlock1D for the first RAFT iteration, where coords are the
    initial integer x-grid. It avoids GridSample by applying fixed one-hot
    interpolation masks along the correlation-volume width.
    """

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock1D.corr(fmap1, fmap2)
        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch, h1, w1, w2)

        self.corr_pyramid.append(corr)
        corr_pool = corr.reshape(batch * h1 * w1, 1, 1, w2)
        for i in range(self.num_levels):
            corr_pool = F.avg_pool2d(corr_pool, [1, 2], stride=[1, 2])
            corr_lvl = corr_pool.reshape(batch, h1, w1, w2 // 2 ** (i + 1))
            self.corr_pyramid.append(corr_lvl)

    def __call__(self, coords):
        r = self.radius
        batch, _, h1, w1 = coords.shape
        out_pyramid = []

        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            w2 = corr.shape[-1]

            x = torch.arange(w1, device=coords.device, dtype=torch.float32)
            x = x.view(1, 1, w1, 1) / 2 ** i
            dx = torch.arange(-r, r + 1, device=coords.device, dtype=torch.float32)
            dx = dx.view(1, 1, 1, 2 * r + 1)
            x = x + dx

            x0 = torch.floor(x)
            x1 = x0 + 1.0
            wx = x - x0

            cols = torch.arange(w2, device=coords.device, dtype=torch.float32)
            cols = cols.view(1, 1, w2, 1)
            x0 = x0.view(1, w1, 1, 2 * r + 1)
            x1 = x1.view(1, w1, 1, 2 * r + 1)
            wx = wx.view(1, w1, 1, 2 * r + 1).to(dtype=corr.dtype)

            mask0 = (cols == x0).to(dtype=corr.dtype) * (1.0 - wx)
            mask1 = (cols == x1).to(dtype=corr.dtype) * wx
            mask = (mask0 + mask1).view(1, 1, w1, w2, 2 * r + 1)

            sampled = (corr.unsqueeze(-1) * mask).sum(dim=3)
            out_pyramid.append(sampled)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous()


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        raise NotImplementedError
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / (dim ** 0.5)
