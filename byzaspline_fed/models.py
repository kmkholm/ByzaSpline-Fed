"""Mamba encoder (pure-PyTorch selective scan) + B-spline Kolmogorov-Arnold head.

KAN layer: phi_ji(u) = w_ji * SiLU(u) + sum_p c_jip B_p(u; tau_i), cubic B-splines (degree k = 3) on G = 8 intervals,
initial grid [-2, 2] extended by k knots per side (P = G + k = 11 coefficients per edge). Every input i has its own
knot vector tau_i (buffer `grid`, shape (in, G + 2k + 1)); clients adapt it to their own activations after local
training (update_grid), so the uploaded knot vectors differ between clients and the server has to align them.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from splines import batched_basis


# ------------------------------------------------------------------ Mamba
def selective_scan(u, delta, A, B, C, D, chunk=8):
    """h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t,  y_t = C_t . h_t + D u_t   (diagonal A, per channel).
    Exact chunked evaluation; the per-step log decay is floored at -6 so exp() stays inside float32 range."""
    b, l, d = u.shape
    n = A.shape[1]
    pad = (-l) % chunk
    if pad:
        z = lambda t: F.pad(t, (0, 0, 0, pad))
        u, delta, B, C = z(u), z(delta), z(B), z(C)
    L = u.shape[1]
    nc = L // chunk
    dA = (delta.unsqueeze(-1) * A).clamp(min=-6.0).view(b, nc, chunk, d, n)
    dBu = ((delta * u).unsqueeze(-1) * B.unsqueeze(2)).view(b, nc, chunk, d, n)
    S = torch.cumsum(dA, dim=2)
    eS = torch.exp(S)
    h_local = eS * torch.cumsum(torch.exp(-S) * dBu, dim=2)
    ends, decays = h_local[:, :, -1].unbind(1), eS[:, :, -1].unbind(1)
    carry = [torch.zeros(b, d, n, device=u.device, dtype=u.dtype)]
    for c in range(nc - 1):
        carry.append(decays[c] * carry[-1] + ends[c])
    H = (h_local + eS * torch.stack(carry, 1).unsqueeze(2)).view(b, L, d, n)
    y = torch.einsum("bldn,bln->bld", H, C.view(b, L, n)) + D * u
    return y[:, :l]


class MambaBlock(nn.Module):
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2):
        super().__init__()
        di = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * di)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(di, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d_model)
        with torch.no_grad():
            dt = torch.exp(torch.rand(di) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.d_state = d_state

    def forward(self, x):
        r = x
        x = self.norm(x)
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2))
        dt, B, C = torch.split(self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        y = selective_scan(u, F.softplus(self.dt_proj(dt)), -torch.exp(self.A_log), B, C, self.D)
        return r + self.out_proj(y * F.silu(z))


class MambaEncoder(nn.Module):
    """Patch embedding (Conv1d stride p, GELU, BatchNorm, learnable positions) -> N Mamba blocks -> pooling."""

    def __init__(self, c_in, length, d_model=64, n_layers=3, patch=8, d_state=16, d_conv=4, pool="mean", norm="bn"):
        super().__init__()
        self.pool = pool
        self.embed = nn.Conv1d(c_in, d_model, patch, stride=patch)
        # under differential privacy GroupNorm is used: BatchNorm running statistics would be released without noise
        self.bn = nn.BatchNorm1d(d_model) if norm == "bn" else nn.GroupNorm(8, d_model)
        self.pos = nn.Parameter(torch.zeros(1, length // patch, d_model))
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)
        self.dim = d_model

    def forward(self, x):
        h = self.bn(F.gelu(self.embed(x))).transpose(1, 2)
        h = h + self.pos[:, : h.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        return h[:, -1] if self.pool == "last" else h.mean(1)


# ------------------------------------------------------------------ B-spline KAN
class BSplineKAN(nn.Module):
    def __init__(self, d_in, d_out, G=8, k=3, lo=-2.0, hi=2.0, grid_eps=0.5, margin=0.01):
        super().__init__()
        self.d_in, self.d_out, self.G, self.k = d_in, d_out, G, k
        self.grid_eps, self.margin = grid_eps, margin
        h = (hi - lo) / G
        grid = (torch.arange(-k, G + k + 1, dtype=torch.float32) * h + lo).expand(d_in, -1).contiguous()
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        self.spline_weight = nn.Parameter(torch.randn(d_out, d_in, G + k) * (0.1 / math.sqrt(d_in)))

    def b_splines(self, x):                                   # (b, in) -> (b, in, P)
        return batched_basis(x, self.grid, self.k)

    def forward(self, x):
        base = F.linear(F.silu(x), self.base_weight)
        spl = F.linear(self.b_splines(x).reshape(x.shape[0], -1), self.spline_weight.reshape(self.d_out, -1))
        return base + spl

    @torch.no_grad()
    def update_grid(self, x):
        """Adapt the knot vectors to the activations x (n, in) (mix of quantile and uniform grids, as in
        efficient-KAN) and refit the coefficients by least squares so the spline outputs on x are preserved."""
        n = x.shape[0]
        old = torch.einsum("bip,oip->bio", self.b_splines(x), self.spline_weight)         # (n, in, out)
        xs = torch.sort(x, dim=0)[0]
        adaptive = xs[torch.linspace(0, n - 1, self.G + 1, dtype=torch.int64, device=x.device)]
        step = (xs[-1] - xs[0] + 2 * self.margin) / self.G
        uniform = torch.arange(self.G + 1, dtype=x.dtype, device=x.device).unsqueeze(1) * step + xs[0] - self.margin
        g = self.grid_eps * uniform + (1 - self.grid_eps) * adaptive
        ext = torch.arange(1, self.k + 1, dtype=x.dtype, device=x.device).unsqueeze(1)
        g = torch.cat([g[:1] - step * ext.flip(0), g, g[-1:] + step * ext], dim=0)       # (G + 2k + 1, in)
        self.grid.copy_(g.T)
        A = self.b_splines(x).transpose(0, 1).double().cpu()                               # (in, n, P)
        Bt = old.transpose(0, 1).double().cpu()                                            # (in, n, out)
        sol = torch.linalg.lstsq(A, Bt, driver="gelsd").solution                           # (in, P, out)
        self.spline_weight.copy_(sol.permute(2, 0, 1).to(self.spline_weight))


class Net(nn.Module):
    """task in {cls, rul, ae}. cls/rul: encoder -> KAN(d -> H) -> KAN(H -> n_out); ae: encoder -> KAN(d -> H) -> decoder."""

    def __init__(self, task, n_out, c_in, length, patch=8, d_model=64, n_layers=3, hidden=32, G=8, k=3, norm="bn"):
        super().__init__()
        self.task = task
        self.enc = MambaEncoder(c_in, length, d_model=d_model, n_layers=n_layers, patch=patch,
                                pool="last" if task == "rul" else "mean", norm=norm)
        self.kan1 = BSplineKAN(d_model, hidden, G, k)
        self.kan2 = BSplineKAN(hidden, n_out, G, k) if task in ("cls", "rul") else None
        if task == "ae":
            self.c_in, self.length = c_in, length
            self.dec_fc = nn.Linear(hidden, 64 * 40)
            self.dec_up = nn.ConvTranspose1d(64, c_in, 8, stride=8)

    def forward(self, x):
        z = self.kan1(self.enc(x))
        if self.task == "cls":
            return self.kan2(z)
        if self.task == "rul":
            return self.kan2(z).squeeze(-1)
        h = F.gelu(self.dec_fc(z)).view(-1, 64, 40)
        return self.dec_up(h)[..., : self.length]

    def kan_names(self):
        return [n for n in ("kan1", "kan2") if getattr(self, n) is not None]

    @torch.no_grad()
    def update_grids(self, x):
        was = self.training
        self.eval()
        h = self.enc(x)
        self.kan1.update_grid(h)
        if self.kan2 is not None:
            self.kan2.update_grid(self.kan1(h))
        self.train(was)
