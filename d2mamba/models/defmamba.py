# This file is migrated from leiyeliu/DefMamba classification/models/utils.py and vmamba.py.
# Only the components used by D2Mamba (DSSM and the deformable scan) are kept.

import torch
import torch.nn as nn
import math
from einops import repeat
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import einops

if hasattr(torch, 'amp') and hasattr(torch.amp, 'custom_fwd'):
    def selective_scan_custom_fwd(fn):
        return torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')(fn)

    def selective_scan_custom_bwd(fn):
        return torch.amp.custom_bwd(device_type='cuda')(fn)
else:
    def selective_scan_custom_fwd(fn):
        return torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)(fn)

    def selective_scan_custom_bwd(fn):
        return torch.cuda.amp.custom_bwd(fn)


try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    import selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @selective_scan_custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @selective_scan_custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class DeformablePathTrans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, de_index):
        B, C, N = x.shape
        _, indices = torch.topk(de_index, k=N, dim=-1, largest=False)
        x_gathered = torch.gather(x, 2, indices.unsqueeze(1).expand(-1, C, -1)).contiguous()
        x_out = x_gathered.permute(0, 2, 1).contiguous()
        ctx.save_for_backward(x, de_index, indices)
        return x_out, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        x, de_index, indices = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_x.scatter_add_(2, indices.unsqueeze(1).expand(-1, x.shape[1], -1), grad_output.permute(0, 2, 1).contiguous()).contiguous()
        grad_de_index = (grad_output.permute(0, 2, 1).contiguous()-grad_x).mean(dim=1)
        grad_de_index = grad_de_index.view_as(de_index)
        return grad_x, grad_de_index


class ConvOffset(nn.Module):
    def __init__(self, embed_dim, kk, pad_size):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kk, 1, pad_size, groups=embed_dim)
        self.ca = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//16),
                nn.GELU(),
                nn.Linear(embed_dim//16, embed_dim),
                nn.Sigmoid()
                )
        self.ln = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim, 3, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x_c = F.adaptive_avg_pool2d(x, (1, 1))
        x_c = self.ca(x_c.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x1 * x_c.expand_as(x)
        x = self.gelu(self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = self.conv2(x)
        return x


class DeformableLayer(nn.Module):
    def __init__(
            self, index=0, embed_dim=192, debug=False, h=0, w=0):
        super().__init__()
        self.ksize = [9, 7, 5, 3]
        self.stride = 1
        kk = self.ksize[index]
        pad_size = kk // 2 if kk != 1 else 0
        self.debug = debug
        self.conv_offset = ConvOffset(embed_dim, kk, pad_size)
        self.rpe_table = nn.Parameter(
            torch.zeros(embed_dim, 7, 7)
        )
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1)  # B H W 2

        return ref

    @torch.no_grad()
    def _get_key_ref_points(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0, H, H, dtype=dtype, device=device),
            torch.linspace(0, W, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1)  # B H W 2

        return ref

    @torch.no_grad()
    def _get_path_ref_points(self, N, B, dtype, device):
        ref_path = torch.linspace(0.5, N - 0.5, N, dtype=dtype, device=device),
        ref_path[0].div_(N - 1.0).mul_(2.0).sub_(1.0)
        ref = ref_path[0][None, ...].expand(B, -1)  # B H W 1
        return ref

    @torch.no_grad()
    def _get_path_key_ref_points(self, N, B, dtype, device):
        ref_path = torch.linspace(0, N, N, dtype=dtype, device=device),
        ref_path[0].div_(N - 1.0).mul_(2.0).sub_(1.0)
        ref = ref_path[0][None, ...].expand(B, -1).flatten(1)  # B H W 1
        return ref

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        dtype, device = x.dtype, x.device
        B, C, H, W = x.size()
        N = H * W

        offset = self.conv_offset(x).contiguous()  # B 2 Hg Wg
        offset, de_index = torch.split(offset, [2, 1], dim=1)
        Hk, Wk = offset.size(2), offset.size(3)

        offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range)

        offset = einops.rearrange(offset, 'b p h w -> b h w p').contiguous()
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        de_index = de_index.tanh().flatten(1)
        path_reference = self._get_path_ref_points(N, B, dtype, device)

        pos = offset + reference

        path_pos = de_index + path_reference

        x_sampled = F.grid_sample(
            input=x,
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B, C, Hg, Wg

        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        rpe_bias = F.interpolate(rpe_bias, size=(H, W), mode='bilinear', align_corners=False)
        key_grid = self._get_key_ref_points(H, W, B, dtype, device)
        displacement = (key_grid - pos) * 0.5
        pos_bias = F.grid_sample(
            input=rpe_bias,
            grid=displacement[..., (1, 0)],
            mode='bilinear', align_corners=True
        )
        x = x_sampled + pos_bias
        x = x.flatten(2)
        x, indices = DeformablePathTrans.apply(x, path_pos) #B N C
        return x, indices


class DeformableLayerReverse(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, indices=None):
        x = x.flatten(2)
        B, C, N = x.size()
        index_re = torch.zeros_like(indices, device=x.device)
        index_re.scatter_add_(1, indices, torch.arange(indices.size(-1), device=x.device).unsqueeze(0).expand(indices.size(0), -1))
        x = torch.gather(x, 2, index_re.unsqueeze(1).expand(-1, C, -1))
        return x


def x_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        force_fp32=True,
        stage=0,
        DS=None,
        DR=None,
        **kwargs,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
    K, D, R = dt_projs_weight.shape
    B, D, H, W = x.shape
    L = H * W
    _, N = A_logs.shape

    xs = x.new_empty((B, 3, D, H * W))
    xs[:, 0] = x.flatten(2, 3)
    xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
    temp, indices = DS(x)
    xs[:, 2] = temp.permute(0, 2, 1)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    )

    ys = ys.view(B, K, -1, H, W)
    ys = ys.view(B, K, D, -1)
    y = (ys[:, 0] + ys[:, 1].flip(dims=[-1]) + DR(ys[:, 2], indices)) / 3.

    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    if K!=1: y = y.view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y), {'ys':ys, 'xs':xs, 'dts':dts, 'As':A_logs, 'Bs':Bs, 'Cs':Cs, 'Ds':Ds, 'delta_bias':delta_bias}


# ==================== DefMamba backbone ====================


# =====================================================


class DSSM(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        # ======================
        forward_type="v2",
        # ======================
        stage = 0,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.d_inner = d_inner
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.stage = stage

        self.K = 3
        self.K2 = self.K

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            stride = 1
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                stride=stride,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.DS = DeformableLayer(index=stage, embed_dim=d_inner, debug=False)
        self.DR = DeformableLayerReverse()

        # other kwargs =======================================
        self.kwargs = kwargs
        if simple_init:
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        self.debug = False
        self.outnorm = None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A) 
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D) 
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if self.debug: debug_rec = []
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = x_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.outnorm,
            nrows=nrows, delta_softplus=True, force_fp32=self.training, stage=self.stage, DS=self.DS, DR=self.DR,
            **self.kwargs,
        )
        x, debug_rec = x[0], x[1]
        if self.debug:
            return x, debug_rec
        return  x

    def forward(self, x: torch.Tensor,h_tokens=None,w_tokens=None, **kwargs):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        b, h, w, d = x.shape
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


    


