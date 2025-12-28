import os
import random
import numpy as np
import tifffile
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim


# ---------------- Model (Spatial Encoder + ALNN + classifier) ----------------
def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class GroupedSpatialEncoder(nn.Module):
    """
    Input: patches [B, C, H, W]  (C = bands)
    Group bands into groups of size g, produce feature sequence [B, L, out_dim].
    """
    def __init__(self, patch_h=16, patch_w=16, n_bands=94, group_size=3, out_dim=64):
        super().__init__()
        self.H = patch_h
        self.W = patch_w
        self.B = n_bands
        self.g = group_size
        self.L = (n_bands + group_size - 1) // group_size
        self.out_dim = out_dim

        # 2D branch
        self.branch2d = nn.Sequential(
            nn.Conv2d(self.g, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 3D branch
        self.branch3d = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.fusion_conv = nn.Conv2d(96, out_dim, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(out_dim)
        self.res_proj = nn.Conv2d(self.g, out_dim, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

        # init
        self.apply(weight_init)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        device = x.device
        total_needed = self.L * self.g
        if C < total_needed:
            pad = torch.zeros(B, total_needed - C, H, W, device=device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
            C = total_needed
        # reshape into groups: [B, L, g, H, W]
        x_groups = x.view(B, self.L, self.g, H, W)
        # merge B and L
        merged = x_groups.view(B * self.L, self.g, H, W)  # for 2D branch
        out2d = self.branch2d(merged)  # [B*L, 64, H, W]
        # 3D branch expects [B*L, 1, g, H, W]
        merged3d = merged.unsqueeze(1)
        out3d = self.branch3d(merged3d)  # [B*L, 32, g, H, W]
        out3d = out3d.mean(dim=2)  # [B*L, 32, H, W]
        fused = torch.cat([out2d, out3d], dim=1)  # [B*L, 96, H, W]
        fused = self.fusion_conv(fused)
        fused = self.fusion_bn(fused)
        res = self.res_proj(merged)  # [B*L, out_dim, H, W]
        fused = fused + res
        fused = self.act(fused)
        gap = F.adaptive_avg_pool2d(fused, 1).view(B * self.L, self.out_dim)
        feat_seq = gap.view(B, self.L, self.out_dim)
        return feat_seq  # [B, L, out_dim]


class TauModule(nn.Module):
    def __init__(self, in_dim, state_dim, out_dim=None, tau_min=0.5, tau_max=5.0):
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else state_dim
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.Wi = nn.Linear(in_dim, self.out_dim, bias=False)
        self.Wx = nn.Linear(state_dim, self.out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.out_dim))
        self.apply(weight_init)

    def forward(self, x, I):
        g = self.Wi(I) + self.Wx(x) + self.bias
        s = torch.sigmoid(g)
        tau = self.tau_min + (self.tau_max - self.tau_min) * s
        return tau


class NCP(nn.Module):
    def __init__(self, n_sensory, n_inter, n_command, n_motor, input_dim, state_dim, sparsity=0.25):
        super().__init__()
        self.n_s = n_sensory
        self.n_h = n_inter
        self.n_c = n_command
        self.n_m = n_motor
        self.N = self.n_s + self.n_h + self.n_c + self.n_m
        # small initialization to avoid large recurrent gains
        W = torch.randn(self.N, self.N) * 0.01
        self.W = nn.Parameter(W)
        mask = (torch.rand(self.N, self.N) < sparsity).float()
        self.register_buffer('mask', mask)
        self.B = nn.Linear(input_dim, self.N, bias=False)
        self.C = nn.Linear(state_dim, self.N, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.N))
        self.Wm = nn.Linear(self.n_m, state_dim, bias=True)
        self.activation = torch.tanh
        self.apply(weight_init)

    def forward(self, x, I):
        # x: [B, state_dim], I: [B, input_dim]
        pre = self.B(I) + self.C(x) + self.bias
        z = self.activation(pre)
        Wmasked = self.W * self.mask
        recurrent = torch.matmul(z, Wmasked.t())
        z = self.activation(pre + recurrent)
        z_m = z[:, (self.N - self.n_m):]
        h_out = self.Wm(z_m)
        return h_out


class ALNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, g_l=0.1, tau_min=0.5, tau_max=5.0,
                 n_sensory=None, n_inter=16, n_command=8, n_motor=16, ncp_sparsity=0.25,
                 n_substeps=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.g_l = g_l
        self.n_substeps = n_substeps
        self.tau_module = TauModule(input_dim, hidden_dim, out_dim=hidden_dim, tau_min=tau_min, tau_max=tau_max)
        if n_sensory is None:
            n_sensory = max(4, input_dim // 2)
        self.ncp = NCP(n_sensory=n_sensory, n_inter=n_inter, n_command=n_command, n_motor=n_motor,
                       input_dim=input_dim, state_dim=hidden_dim, sparsity=ncp_sparsity)
        self.apply(weight_init)

    def vector_field(self, x, I_t):
        # x: [B, hidden_dim], I_t: [B, input_dim]
        h_val = self.ncp(x, I_t)
        leak = - x - self.g_l * x
        num = leak + h_val
        tau = self.tau_module(x, I_t)
        dxdt = num / (tau + 1e-6)
        return dxdt

    def rk4_small(self, x, I_t, dt):
        # perform RK4 step with step dt
        k1 = self.vector_field(x, I_t)
        k2 = self.vector_field(x + 0.5 * dt * k1, I_t)
        k3 = self.vector_field(x + 0.5 * dt * k2, I_t)
        k4 = self.vector_field(x + dt * k3, I_t)
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next

    def forward(self, I_seq, t_seq=None, x0=None, return_traj=False):
        # I_seq: [B, L, input_dim]
        B, L, D = I_seq.shape
        device = I_seq.device
        if x0 is None:
            x = torch.zeros(B, self.hidden_dim, device=device, dtype=I_seq.dtype)
        else:
            x = x0
        traj = []
        # iterate over discrete time steps; for stability we split dt into n_substeps
        for k in range(L):
            I_k = I_seq[:, k, :]
            # default dt = 1.0 for each discrete band-step; split into n_substeps
            dt = 1.0 / float(self.n_substeps)
            for _ in range(self.n_substeps):
                x = self.rk4_small(x, I_k, dt)
            traj.append(x.unsqueeze(1))
            # quick sanity clamp to avoid inf/nan explosion:
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                # if exploded, clip to large values to avoid NaN in subsequent ops
                x = torch.clamp(x, min=-1e6, max=1e6)
        traj = torch.cat(traj, dim=1)  # [B, L, hidden_dim]
        if return_traj:
            return traj[:, -1, :], traj
        else:
            return traj[:, -1, :], None


class HyperspectralModel(nn.Module):
    def __init__(self, patch_h=16, patch_w=16, bands=94, group_size=3, encoder_out=64,
                 hidden_dim=64, num_classes=2, **alnn_kwargs):
        super().__init__()
        self.encoder = GroupedSpatialEncoder(patch_h, patch_w, n_bands=bands, group_size=group_size, out_dim=encoder_out)
        self.alnn = ALNN(input_dim=encoder_out, hidden_dim=hidden_dim, **alnn_kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim//2, num_classes)
        )
        self.apply(weight_init)

    def forward(self, patches, return_traj=False):
        # patches: [B, C, H, W]
        feat_seq = self.encoder(patches)  # [B, L, d]
        xT, traj = self.alnn(feat_seq, return_traj=return_traj)
        logits = self.classifier(xT)
        return logits, traj



