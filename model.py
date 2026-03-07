import os
import warnings
import time
import math
import numpy as np
import pandas as pd
import torch
from torch import nn, transpose
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn
import logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
LOG = logging.getLogger(__name__)


# ==============================================================================
# Basic building blocks
# ==============================================================================

def elu():
    return nn.ELU(inplace=True)


def instance_norm(filters, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(filters, affine=True, eps=eps, **kwargs)


def conv2d(in_chan, out_chan, kernel_size, dilation=1, **kwargs):
    padding = dilation * (kernel_size - 1) // 2
    return nn.Conv2d(in_chan, out_chan, kernel_size, padding=padding,
                     dilation=dilation, **kwargs)


# ==============================================================================
# AxialAttention2D - replaces broken AttentionLayer from yylfinal
# ==============================================================================

class AxialAttention2D(nn.Module):
    """Axial attention: row attention followed by column attention.

    This replaces the broken AttentionLayer from yylfinal which produced
    a 6D tensor via incorrect einsum ('bhidw,bhjdu->bhijwu').

    The axial decomposition is efficient: O(N * W) + O(N * H) instead of O(N^2).

    When sequence length exceeds window_size, attention is computed within
    fixed-size windows to bound memory at O(N * w) instead of O(N^2).
    CNN layers between attention blocks provide cross-window information flow.
    """

    def __init__(self, channels, heads=4, window_size=256):
        super().__init__()
        self.window_size = window_size
        self.row_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)

    def _windowed_attention(self, x_seq, attn_module):
        """Compute attention within fixed-size windows to avoid O(N^2) memory.

        Args:
            x_seq: (batch, seq_len, channels)
            attn_module: nn.MultiheadAttention
        Returns:
            out: (batch, seq_len, channels)
        """
        B_seq, L, C = x_seq.shape
        w = self.window_size

        if L <= w:
            out, _ = attn_module(x_seq, x_seq, x_seq)
            return out

        # Pad to multiple of window_size
        pad_len = (w - L % w) % w
        if pad_len > 0:
            x_seq = F.pad(x_seq, (0, 0, 0, pad_len))  # pad seq dim
        L_padded = x_seq.shape[1]
        num_windows = L_padded // w

        # Reshape to windows: (B_seq * num_windows, w, C)
        x_win = x_seq.reshape(B_seq * num_windows, w, C)
        out_win, _ = attn_module(x_win, x_win, x_win)
        # Reshape back and remove padding
        out = out_win.reshape(B_seq, L_padded, C)[:, :L, :]
        return out

    def forward(self, x):
        B, C, H, W = x.shape

        # Row attention: each row is a sequence of length W
        x_row = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        attn_row = self._windowed_attention(x_row, self.row_attn)
        x = x + attn_row.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.norm1(x)

        # Column attention: each column is a sequence of length H
        x_col = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        attn_col = self._windowed_attention(x_col, self.col_attn)
        x = x + attn_col.reshape(B, W, H, C).permute(0, 3, 2, 1)
        x = self.norm2(x)

        return x


# ==============================================================================
# ChiselBackbone - dilated residual CNN with optional axial attention
# ==============================================================================

class ChiselBackbone(nn.Module):
    """Dilated residual CNN backbone with optional axial attention.

    Returns both the 2-channel output (adj + dist) AND the intermediate
    feature map (for boundary detection in the downstream RefinedModel).

    Supports gradient checkpointing (use_checkpoint=True) to trade compute
    for memory. When enabled, intermediate activations are recomputed during
    backward instead of stored, reducing memory from O(num_layers) to O(1)
    per checkpoint segment.

    Args:
        include_attention: If True (default), create axial attention layers
            every 4 CNN layers. Set to False for lightweight backbone-only
            inference (fewer parameters, faster, compatible with pretrained
            backbone checkpoints).
        use_checkpoint: If True, use gradient checkpointing during training.
    """

    def __init__(self, filters=64, kernel=3, num_layers=61,
                 in_channels=10, symmetrise_output=False,
                 include_attention=True, use_checkpoint=False):
        super().__init__()
        self.filters = filters
        self.kernel = kernel
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.symmetrise_output = symmetrise_output
        self.use_checkpoint = use_checkpoint

        self.first_block = nn.Sequential(
            conv2d(self.in_channels, filters, 1),
            instance_norm(filters),
            elu()
        )

        self.output_layer_1 = nn.Sequential(
            conv2d(filters, 1, kernel, dilation=1),
            nn.Sigmoid(),
        )

        self.output_layer_2 = nn.Sequential(
            conv2d(filters, 1, kernel, dilation=1),
            nn.Sigmoid(),
        )

        # Dilated residual blocks
        cycle_dilations = [1, 2, 4, 8, 16]
        dilations = [cycle_dilations[i % len(cycle_dilations)]
                     for i in range(num_layers)]

        self.layers = nn.ModuleList([nn.Sequential(
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters),
            elu(),
            nn.Dropout(p=0.15),
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters)
        ) for dilation in dilations])

        # Axial attention every 4 layers
        # When include_attention=False, no attention layers are created,
        # producing a lightweight backbone for fast inference.
        if include_attention:
            self.attention_layers = nn.ModuleList([
                AxialAttention2D(filters, heads=4) if i % 4 == 0 else None
                for i in range(num_layers)
            ])
        else:
            self.attention_layers = None

        self.activate = elu()

    def _block_forward(self, x, layer, attention):
        """Single residual block + optional attention. Used as checkpoint unit."""
        x = self.activate(x + layer(x))
        if attention is not None:
            x = attention(x)
        return x

    def forward(self, x):
        x = self.first_block(x)

        if self.attention_layers is not None:
            for layer, attention in zip(self.layers, self.attention_layers):
                if self.use_checkpoint and self.training:
                    x = ckpt_fn(self._block_forward, x, layer, attention,
                                use_reentrant=False)
                else:
                    x = self._block_forward(x, layer, attention)
        else:
            # No attention layers (lightweight backbone mode)
            for layer in self.layers:
                x = self.activate(x + layer(x))

        # feat: intermediate feature map (B, filters, N, N) for boundary head
        feat = x

        # 2-channel output: channel 0 = adjacency, channel 1 = distance
        y_hat = torch.cat([self.output_layer_1(x),
                           self.output_layer_2(x)], dim=1)

        if self.symmetrise_output:
            y_hat = (y_hat + transpose(y_hat, -1, -2)) * 0.5

        return feat, y_hat


# ==============================================================================
# pLDDT confidence prediction head (from yylfinal, self-supervised)
# ==============================================================================

class PLDDTPredictionHead(nn.Module):
    """Per-residue confidence prediction, similar to AlphaFold's pLDDT head.

    Takes the 2D feature map from ChiselBackbone and predicts a (B, 1, N, N)
    confidence map. The diagonal values represent per-residue confidence.

    Self-supervised: no external pLDDT labels needed. Target is derived from
    prediction accuracy (1 - row_error) during training.
    """

    def __init__(self, input_channels, filters=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, filters, 1),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# ==============================================================================
# VGAE: proper variational graph auto-encoder
# ==============================================================================
# Key differences from the old rank-2 VGAE:
#   - 32-dim latent space (vs 2), enough for 10+ domain block-diagonal structures
#   - 3-layer GCN encoder with degree-normalized message passing
#   - Bilinear decoder (z @ W @ z^T) instead of simple inner product
#   - KL divergence regularization (proper VAE objective)
#   - Operates on per-node features (diagonal of CNN feat map)

class GCNNodeLayer(nn.Module):
    """Graph convolution on per-node features with degree normalization.

    Standard GCN: H' = D^{-1}(A + I) H W
    Input: x (B, N, C_in), A (B, N, N)
    Output: (B, N, C_out)
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, A):
        # Add self-loops: A_hat = A + I
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).unsqueeze(0)
        A_hat = A + I
        # Degree normalization: D^{-1} A_hat
        D_inv = A_hat.sum(dim=-1, keepdim=True).clamp(min=1).reciprocal()
        A_norm = A_hat * D_inv
        # Message passing + linear transform
        x = torch.bmm(A_norm, x)  # (B, N, C_in)
        x = self.linear(x)  # (B, N, C_out)
        return x


class VGAEEncoder(nn.Module):
    """3-layer GCN encoder producing mean and log-variance for each node.

    Architecture: GCN(in→hidden) → GCN(hidden→hidden) → split to mu, logvar
    """

    def __init__(self, in_features, hidden_dim=32, latent_dim=32):
        super().__init__()
        self.gcn1 = GCNNodeLayer(in_features, hidden_dim)
        self.gcn2 = GCNNodeLayer(hidden_dim, hidden_dim)
        self.mu_layer = GCNNodeLayer(hidden_dim, latent_dim)
        self.logvar_layer = GCNNodeLayer(hidden_dim, latent_dim)

    def forward(self, x, A):
        """
        Args:
            x: (B, N, C_in) per-node features
            A: (B, N, N) adjacency matrix
        Returns:
            mu: (B, N, latent_dim)
            logvar: (B, N, latent_dim)
        """
        h = F.relu(self.gcn1(x, A))
        h = F.relu(self.gcn2(h, A))
        mu = self.mu_layer(h, A)
        logvar = self.logvar_layer(h, A)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar, training=True):
        """VAE reparameterization trick. Use mu only at inference."""
        if training:
            # Clamp logvar to prevent exp overflow in fp16 (max ~11 → exp(11)≈60000)
            logvar = logvar.clamp(-20, 20)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


class BilinearDecoder(nn.Module):
    """Learned bilinear decoder: A_ij = sigmoid(z_i^T W z_j).

    Much more expressive than simple inner product (z @ z^T).
    With latent_dim=32, can represent rank-32 block-diagonal structures.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z):
        """
        Args:
            z: (B, N, D) node latent vectors
        Returns:
            A: (B, N, N) reconstructed adjacency
        """
        # z @ W → (B, N, D), then (B, N, D) @ (B, D, N) → (B, N, N)
        Wz = torch.matmul(z, self.weight)
        A = torch.sigmoid(torch.bmm(Wz, z.transpose(1, 2)))
        return A


# ==============================================================================
# RefinedModel - ChiselBackbone + VGAE + pLDDT + boundary head
# ==============================================================================

class RefinedModel(nn.Module):
    """Protein domain prediction model with VGAE refinement.

    Stage 1: ChiselBackbone → A_hat_raw (B,2,N,N) + feat (B,64,N,N)
    Stage 2: VGAE refines CNN adjacency using graph structural constraints
    Aux 1:   pLDDT confidence from feat
    Aux 2:   Boundary detection from feat diagonal

    Outputs:
      - A_hat_raw (B, 2, N, N): CNN adjacency + distance predictions
      - A_refined (B, N, N): VGAE-refined adjacency
      - vgae_mu (B, N, latent_dim): VGAE latent mean (for KL loss)
      - vgae_logvar (B, N, latent_dim): VGAE latent log-variance (for KL loss)
      - plddt (B, 1, N, N): per-residue confidence map
      - boundary_pred (B, 1, N): domain boundary probabilities
    """

    def __init__(self, filters=64, num_layers=61, in_channels=10,
                 vgae_hidden=32, vgae_latent=32, use_checkpoint=False):
        super().__init__()
        self.cnn = ChiselBackbone(filters, 3, num_layers, in_channels,
                                     use_checkpoint=use_checkpoint)

        # VGAE: extract per-node features from CNN feat diagonal
        # Node features = filters-dim, refined through 3-layer GCN
        self.vgae_encoder = VGAEEncoder(filters, vgae_hidden, vgae_latent)
        self.vgae_decoder = BilinearDecoder(vgae_latent)

        # pLDDT confidence head
        self.plddt_head = PLDDTPredictionHead(filters, filters)

        # Boundary detection auxiliary head
        self.boundary_head = nn.Sequential(
            nn.Conv1d(filters, filters // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(filters // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Stage 1: CNN backbone
        feat, A_hat_raw = self.cnn(x)
        # feat: (B, filters, N, N), A_hat_raw: (B, 2, N, N)

        # Stage 2: VGAE refinement
        # Per-node features from feat diagonal: (B, N, filters)
        node_feat = torch.diagonal(feat, dim1=-2, dim2=-1)  # (B, filters, N)
        node_feat = node_feat.permute(0, 2, 1)  # (B, N, filters)
        # CNN predicted adjacency as graph structure
        A_adj = A_hat_raw[:, 0]  # (B, N, N)
        # Encode → latent → decode
        vgae_mu, vgae_logvar = self.vgae_encoder(node_feat, A_adj)
        z = self.vgae_encoder.reparameterize(vgae_mu, vgae_logvar, self.training)
        A_refined = self.vgae_decoder(z)  # (B, N, N)

        # pLDDT confidence
        plddt = self.plddt_head(feat)  # (B, 1, N, N)

        # Boundary detection from diagonal features
        diag_feat = torch.diagonal(feat, dim1=-2, dim2=-1)  # (B, filters, N)
        boundary_pred = self.boundary_head(diag_feat)  # (B, 1, N)

        return A_hat_raw, A_refined, vgae_mu, vgae_logvar, plddt, boundary_pred


# ==============================================================================
# Self-optimizing training (SOT) - L_c loss
# ==============================================================================

class self_optimizing_training(nn.Module):
    def __init__(self, num_of_features, K=10):
        super().__init__()
        self.K = K
        self.mu = nn.Parameter(
            torch.randn((1, 1, K, num_of_features)), requires_grad=True)

    def forward(self, z, update_p):
        z = z.unsqueeze(2).repeat(1, 1, self.K, 1)
        mu = self.mu.repeat(z.shape[0], z.shape[1], 1, 1)

        q_iu_up = 1. / (1 + (z - mu) ** 2)
        q_iu_down = torch.sum(1. / q_iu_up, dim=2).unsqueeze(2).repeat(
            1, 1, self.K, 1)
        q_iu = q_iu_up / q_iu_down

        p_iu_up = q_iu ** 2 / torch.sum(q_iu, dim=1).unsqueeze(1)
        p_iu_down = q_iu ** 2 / torch.sum(q_iu, dim=1).unsqueeze(1)
        p_iu_down = torch.sum(p_iu_down, dim=2).unsqueeze(2)
        p_iu = p_iu_up / p_iu_down

        # KL divergence: p * log(p/q). Add eps to prevent 0*log(0)=NaN
        eps = 1e-10
        L_c = p_iu * torch.log((p_iu + eps) / (q_iu + eps))
        L_c = torch.mean(L_c, dim=1)
        L_c = torch.mean(L_c, dim=2)
        return L_c


# ==============================================================================
# Positional encoding helper
# ==============================================================================

def positional_encoding(dim, sentence_length, device):
    pe = torch.zeros(sentence_length, dim).to(device)
    for pos in range(sentence_length):
        for i in range(0, dim, 2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / dim)))
            if i + 1 < dim:
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / dim)))
    return pe


# ==============================================================================
# MyLoss - 6-term loss function
# ==============================================================================

class MyLoss(nn.Module):
    """Loss function with 6 terms:

    L = adj_ce   * BCE(pred_adj, true_adj, pos_weight)     # main task
      + lr_mse   * MSE(pred_dist, true_dist)                # distance reconstruction
      + L_c      * SOT KL divergence                        # clustering constraint
      + vgae     * [BCE(A_refined, true_adj) + beta*KL]     # VGAE refinement
      + boundary * BCE(boundary_pred, boundary_true)         # boundary detection
      + plddt    * MSE(plddt_diag, plddt_target)             # self-supervised confidence
    """

    def __init__(self, positional_encoding='linear', device='cpu',
                 loss_weight=None, log=True):
        super().__init__()
        self.device = device
        self.mse = nn.MSELoss()

        if loss_weight is None:
            self.loss_weight = {
                'adj_ce_weight': 1.0,
                'lr_mse_weight': 1.0,
                'Lc_weight': 1.0,
                'vgae_weight': 1.0,
                'boundary_weight': 0.5,
                'plddt_weight': 0.3,
            }
        else:
            self.loss_weight = loss_weight

        # KL divergence weight (beta-VAE style)
        self.vgae_kl_beta = 0.01

        self.positional_encoding = positional_encoding
        if log:
            print('loss_weight:', self.loss_weight)

        # Boundary sharpness convolution (fixed weights, not used in loss
        # computation but required for pretrained checkpoint compatibility)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1,
                              kernel_size=2, stride=1, padding=0, bias=False)
        with torch.no_grad():
            self.conv.weight.data[0] = torch.tensor([0.5, -0.5])
        self.conv.weight.requires_grad = False

        # SOT module
        self.sot = self_optimizing_training(
            num_of_features=4, K=10).to(device)
        self.sot_mu = 0.
        self.i = 0

        # Feature projection for SOT
        self.linear = nn.Sequential(
            nn.Linear(4, 16),
            nn.Sigmoid(),
            nn.Linear(16, 4),
            nn.LeakyReLU()
        ).to(device)

        # Logging lists (all batches across all epochs, for CSV export)
        self.ce_list = []
        self.lr_list = []
        self.Lc_list = []
        self.vgae_list = []
        self.boundary_list = []
        self.plddt_list = []
        self.loss_list = []
        self.cost_list = []

        # Epoch-level accumulators (reset each epoch)
        self._epoch_sums = {}
        self._epoch_count = 0
        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        """Reset epoch-level accumulators. Call before each train/val phase."""
        self._epoch_sums = {
            'adj_ce': 0.0, 'lr_mse': 0.0, 'L_c': 0.0,
            'vgae': 0.0, 'boundary': 0.0, 'plddt': 0.0, 'total': 0.0,
        }
        self._epoch_count = 0
        self._epoch_nan_count = 0

    def get_epoch_summary(self):
        """Return dict of epoch-averaged loss components."""
        n = max(self._epoch_count, 1)
        summary = {k: v / n for k, v in self._epoch_sums.items()}
        summary['nan_count'] = self._epoch_nan_count
        summary['batch_count'] = self._epoch_count
        return summary

    def result_to_csv(self, path):
        """Append current logged data to CSV and clear lists to free memory."""
        n = len(self.loss_list)
        if n == 0:
            return
        df = pd.DataFrame({
            'ce_list': self.ce_list[:n],
            'lr_list': self.lr_list[:n],
            'Lc_list': self.Lc_list[:n],
            'vgae_list': self.vgae_list[:n],
            'boundary_list': self.boundary_list[:n],
            'plddt_list': self.plddt_list[:n],
            'loss_list': self.loss_list[:n],
            'cost_time_on_Lc': self.cost_list[:n],
        })
        # Append mode: first call writes header, subsequent appends
        write_header = not os.path.exists(path)
        df.to_csv(path, mode='a', header=write_header)

        # Clear lists to prevent unbounded memory growth
        self.ce_list.clear()
        self.lr_list.clear()
        self.Lc_list.clear()
        self.vgae_list.clear()
        self.boundary_list.clear()
        self.plddt_list.clear()
        self.loss_list.clear()
        self.cost_list.clear()

    def norm(self, x):
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-8:
            return torch.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    def forward(self, A_hat_raw, A_refined, vgae_mu, vgae_logvar,
                plddt_pred, boundary_pred, true_adj_matrix):
        """
        Args:
            A_hat_raw: (B, 2, N, N) - CNN output [adj, dist]
            A_refined: (B, N, N) - VGAE-refined adjacency
            vgae_mu: (B, N, D) - VGAE latent mean
            vgae_logvar: (B, N, D) - VGAE latent log-variance
            plddt_pred: (B, 1, N, N) - confidence prediction
            boundary_pred: (B, 1, N) - boundary detection prediction
            true_adj_matrix: (B, 2, N, N) - ground truth [adj, dist]
        """
        # Split channels
        pred_adj = A_hat_raw[:, 0:1]  # (B, 1, N, N)
        pred_dist = A_hat_raw[:, 1:2]  # (B, 1, N, N)
        true_adj = true_adj_matrix[:, 0:1]  # (B, 1, N, N)
        true_dist = true_adj_matrix[:, 1:2]  # (B, 1, N, N)

        # ---- 1. Adjacency BCE with dynamic pos_weight ----
        pred_adj_f = torch.nan_to_num(pred_adj.float(), nan=0.5).clamp(1e-6, 1 - 1e-6)
        true_adj_f = true_adj.float()
        pos_count = true_adj_f.sum().clamp(min=1)
        neg_count = (true_adj_f.numel() - pos_count).clamp(min=1)
        pos_weight = (neg_count / pos_count).clamp(max=10.0)
        ce_loss = F.binary_cross_entropy(
            pred_adj_f, true_adj_f,
            weight=(true_adj_f * (pos_weight - 1) + 1)
        ) * float(self.loss_weight.get('adj_ce_weight', 1.0))

        # ---- 2. Distance reconstruction MSE ----
        lr_mse_loss = self.mse(pred_dist, true_dist) * float(
            self.loss_weight.get('lr_mse_weight', 1.0))

        # ---- 3. SOT / L_c loss ----
        A_bar_squeeze = A_hat_raw[:, 0]  # (B, N, N)
        A_mean_0 = torch.sum(A_bar_squeeze, dim=1)  # (B, N)
        A_mean_1 = torch.sum(A_bar_squeeze, dim=2)  # (B, N)
        dim_2 = torch.sqrt(
            A_mean_0.clamp(min=0) * A_mean_1.clamp(min=0))

        if self.positional_encoding == 'sinusoidal':
            pe = torch.arange(
                A_mean_0.shape[1], dtype=torch.float32,
                device=self.device).unsqueeze(0) / A_mean_0.shape[1]
            dim_size = pe.shape[0]
            sentence_length = pe.shape[1]
            pe = positional_encoding(dim_size, sentence_length, self.device).T
        else:  # linear
            pe = torch.arange(
                A_mean_0.shape[1], dtype=torch.float32,
                device=self.device).unsqueeze(0) / A_mean_0.shape[1]

        A_mean_0 = self.norm(A_mean_0)
        A_mean_1 = self.norm(A_mean_1)
        dim_2 = self.norm(dim_2)
        pe = self.norm(pe)

        new_features = torch.cat([
            A_mean_0.unsqueeze(2), A_mean_1.unsqueeze(2),
            dim_2.unsqueeze(2), pe.unsqueeze(2)
        ], dim=2)
        new_features = self.linear(new_features)

        start_time = time.time()
        L_c = self.sot(new_features, self.i % 5 == 0)
        self.sot_mu = self.sot.mu.detach()
        self.i += 1
        cost_on_Lc = time.time() - start_time

        L_c_loss = torch.mean(L_c) * float(
            self.loss_weight.get('Lc_weight', 1.0))

        # ---- 4. VGAE refinement loss (reconstruction + KL) ----
        true_adj_2d = true_adj_matrix[:, 0]  # (B, N, N)
        # Use same pos_weight as adj_ce so VGAE and CNN optimize consistently
        vgae_recon = F.binary_cross_entropy(
            torch.nan_to_num(A_refined.float(), nan=0.5).clamp(1e-6, 1 - 1e-6),
            true_adj_2d.float(),
            weight=(true_adj_2d.float() * (pos_weight - 1) + 1))
        # KL divergence: KL(q(z|X,A) || N(0,I))
        vgae_kl = -0.5 * torch.mean(
            1 + vgae_logvar - vgae_mu.pow(2) - vgae_logvar.exp())
        vgae_loss = (vgae_recon + self.vgae_kl_beta * vgae_kl) * float(
            self.loss_weight.get('vgae_weight', 1.0))

        # ---- 5. Boundary detection loss ----
        diff = (true_adj_2d[:, :-1, :] - true_adj_2d[:, 1:, :]).abs().sum(-1)
        boundary_true = (diff > 0).float()  # (B, N-1)
        boundary_true = F.pad(boundary_true, (0, 1))  # (B, N)
        boundary_true = boundary_true.unsqueeze(1)  # (B, 1, N)

        b_pos = boundary_true.sum().clamp(min=1)
        b_neg = (boundary_true.numel() - b_pos).clamp(min=1)
        b_weight = (b_neg / b_pos).clamp(max=20.0)
        boundary_loss = F.binary_cross_entropy(
            torch.nan_to_num(boundary_pred.float(), nan=0.5).clamp(1e-6, 1 - 1e-6),
            boundary_true.float(),
            weight=(boundary_true.float() * (b_weight - 1) + 1)
        ) * float(self.loss_weight.get('boundary_weight', 0.5))

        # ---- 6. pLDDT self-supervised confidence loss ----
        # Target: per-residue prediction quality = 1 - mean_abs_error(row)
        pred_adj_2d = A_hat_raw[:, 0]  # (B, N, N)
        true_adj_2d = true_adj_matrix[:, 0]  # (B, N, N)
        row_error = (pred_adj_2d - true_adj_2d).abs().mean(dim=-1)  # (B, N)
        plddt_target = torch.nan_to_num(1 - row_error, nan=0.5).clamp(0, 1).unsqueeze(1)

        # plddt_pred is (B, 1, N, N), take diagonal → (B, 1, N)
        plddt_diag = torch.diagonal(
            plddt_pred[:, 0], dim1=-2, dim2=-1).unsqueeze(1)
        plddt_loss = self.mse(plddt_diag, plddt_target.detach()) * float(
            self.loss_weight.get('plddt_weight', 0.3))

        # ---- Total loss ----
        loss = (ce_loss + lr_mse_loss + L_c_loss + vgae_loss
                + boundary_loss + plddt_loss)

        # Guard against NaN loss (from numerical instability)
        is_nan = torch.isnan(loss)
        if is_nan:
            LOG.warning("NaN loss detected, replacing with zero to skip update")
            loss = torch.zeros_like(loss, requires_grad=True)

        # Logging (per-batch lists for CSV)
        self.ce_list.append(ce_loss.item())
        self.lr_list.append(lr_mse_loss.item())
        self.Lc_list.append(L_c_loss.item())
        self.vgae_list.append(vgae_loss.item())
        self.boundary_list.append(boundary_loss.item())
        self.plddt_list.append(plddt_loss.item())
        self.loss_list.append(loss.item())
        self.cost_list.append(cost_on_Lc)

        # Epoch-level accumulation
        if is_nan:
            self._epoch_nan_count += 1
        else:
            self._epoch_sums['adj_ce'] += ce_loss.item()
            self._epoch_sums['lr_mse'] += lr_mse_loss.item()
            self._epoch_sums['L_c'] += L_c_loss.item()
            self._epoch_sums['vgae'] += vgae_loss.item()
            self._epoch_sums['boundary'] += boundary_loss.item()
            self._epoch_sums['plddt'] += plddt_loss.item()
            self._epoch_sums['total'] += loss.item()
            self._epoch_count += 1

        return loss


# ==============================================================================
# ModelAndLoss wrapper
# ==============================================================================

class ModelAndLoss(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def model_forward(self, x):
        return self.model(x)

    def compute_loss(self, A_hat_raw, A_refined, vgae_mu, vgae_logvar,
                     plddt, boundary_pred, target):
        return self.loss_fn(A_hat_raw, A_refined, vgae_mu, vgae_logvar,
                            plddt, boundary_pred, target)

    def forward(self):
        return


# ==============================================================================
# Domain assignment utilities
# ==============================================================================

def mse_loss_at_residue(V, A_hat, j):
    d = j
    return 2 * ((V[d] @ V.T - A_hat[d]) ** 2).sum() - \
           (V[d] @ V[d] - A_hat[d, d]) ** 2


def null_mse_loss_at_residue(Y, residue_index):
    d = residue_index
    return 2 * (Y[d] ** 2).sum() - Y[d, d] ** 2


def mse_loss(V_hat, Y):
    return ((V_hat @ V_hat.T - Y) ** 2).sum()


def greedy_domain_assignment(A_hat, K_init, device='cpu', N_iter=3,
                              cost_type='mse', linker_threshold=8):
    """Greedy domain assignment algorithm.

    Parameters:
        A_hat: (Batchsize, L, L) predicted adjacency probability matrix
        K_init: Initial number of domains

    Returns:
        V: (Batchsize, L, K) domain assignment matrix
        A_prime: (Batchsize, L, L) hard adjacency matrix
    """
    K_max = K_init
    A = A_hat
    V_final = None
    A_prime_final = None
    batchsize = A_hat.shape[0]

    for n in range(batchsize):
        A_hat_n = A[n]
        A_hat_n = (A_hat_n + A_hat_n.T) / 2
        L = A_hat_n.shape[0]
        V = torch.zeros((L, K_init)).to(device)

        loss_val = mse_loss(V, A_hat_n)

        for _ in range(N_iter):
            for j in range(L):
                loss_minus_d = loss_val - mse_loss_at_residue(V, A_hat_n, j)
                V[j] *= 0

                L0 = loss_minus_d + null_mse_loss_at_residue(A_hat_n, j)
                L_opt = torch.zeros(K_max)

                for k in range(K_max):
                    V[j, k] = 1
                    L_opt[k] = loss_minus_d + mse_loss_at_residue(
                        V, A_hat_n, j)
                    V[j, k] = 0

                z = torch.argmin(L_opt)
                if L_opt[z] < L0:
                    V[j, z] = 1

                loss_val = loss_minus_d + mse_loss_at_residue(V, A_hat_n, j)

                if z == K_max - 1:
                    V = torch.cat((V, torch.zeros_like(V)), -1)
                    K_max = V.shape[1]

        # Remove empty and small clusters
        empty = V.sum(0) == 0
        V = V[:, ~empty]

        cluster_sizes = V.sum(0)
        large_clusters = cluster_sizes >= linker_threshold
        V = V[:, large_clusters]

        A_prime = V @ V.T

        if V_final is None:
            V_final = V.unsqueeze(0)
            A_prime_final = A_prime.unsqueeze(0)
        else:
            V_final = torch.cat([V_final, V.unsqueeze(0)], dim=0)
            A_prime_final = torch.cat(
                [A_prime_final, A_prime.unsqueeze(0)], dim=0)

    return V_final, A_prime_final


def labels_to_adj(labels, device):
    """Convert domain ID label vector to binary adjacency matrix.

    Args:
        labels: (B, N) long, 0=unassigned, 1..K=domain ID
    Returns:
        adj: (B, 1, N, N) float32, same-domain pairs = 1
    """
    B, N = labels.shape
    labels = labels.to(device)
    li = labels.unsqueeze(2)   # (B, N, 1)
    lj = labels.unsqueeze(1)   # (B, 1, N)
    adj = ((li == lj) & (li > 0)).float()  # (B, N, N)
    return adj.unsqueeze(1)    # (B, 1, N, N)


# ==============================================================================
# Self-conditioned training (adapted for RefinedModel with 3 outputs)
# ==============================================================================

def self_conditioned_training(model_and_loss, data, true_adj_matrix,
                               optimizer, device='cpu', R=2, K=4,
                               optimize=True):
    """Self-conditioned training loop for RefinedModel.

    The model outputs (A_hat_raw, A_refined, vgae_mu, vgae_logvar,
                       plddt, boundary_pred).
    During self-conditioning rounds, A_hat_raw[:,0] is used as A_bar
    for the recycling channels.

    Args:
        model_and_loss: ModelAndLoss wrapper
        data: (B, 8, N, N) input features (8 channels from featurisers)
        true_adj_matrix: (B, 2, N, N) ground truth [adj, dist]
        optimizer: torch optimizer
        device: computation device
        R: max recycling rounds
        K: initial domain count for greedy assignment
        optimize: whether to perform backward + step

    Returns:
        (loss_value, (A_hat_raw, A_refined, vgae_mu, vgae_logvar,
                      plddt, boundary_pred))
    """
    model_and_loss.train()
    label = true_adj_matrix

    # Ensure 4D
    if true_adj_matrix.dim() < 4:
        while true_adj_matrix.dim() < 4:
            true_adj_matrix = true_adj_matrix.unsqueeze(0)

    # Align spatial dims
    N_label = true_adj_matrix.shape[-1]
    if data.shape[-1] != N_label:
        data = data[:, :, :N_label, :N_label]

    true_adj_first_channel = true_adj_matrix[:, 0, :, :]

    # Initialize recycling channels
    A_prime = torch.zeros_like(true_adj_first_channel).to(device).unsqueeze(1)
    A_bar = torch.zeros_like(true_adj_first_channel).to(device).unsqueeze(1)

    r = torch.randint(0, R + 1, (1,)).item()

    if optimize:
        optimizer.zero_grad()

    for _ in range(r):
        with torch.no_grad():
            new_data = torch.cat([data, A_prime, A_bar], dim=1)
            A_hat_raw, A_refined, vgae_mu, vgae_logvar, \
                plddt, boundary_pred = model_and_loss.model_forward(new_data)
            A_bar = A_hat_raw[:, 0:1]

        _, A_prime = greedy_domain_assignment(
            A_bar.squeeze(1), device=device, K_init=K, N_iter=1)
        A_prime = A_prime.unsqueeze(1)

    # Final forward with gradients
    new_data = torch.cat([data, A_prime, A_bar], dim=1)
    A_hat_raw, A_refined, vgae_mu, vgae_logvar, \
        plddt, boundary_pred = model_and_loss.model_forward(new_data)

    # Compute loss
    loss = model_and_loss.compute_loss(
        A_hat_raw, A_refined, vgae_mu, vgae_logvar,
        plddt, boundary_pred, label)

    if optimize:
        loss.backward()

        # Check for NaN/Inf gradients — skip update to protect weights
        params = [p for p in model_and_loss.model.parameters()
                  if p.requires_grad and p.grad is not None]
        has_bad_grad = any(
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            for p in params)

        if has_bad_grad:
            optimizer.zero_grad()  # discard corrupted gradients
            LOG.warning("NaN/Inf gradients detected, skipping optimizer step")
        else:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

    return loss.item(), (A_hat_raw, A_refined, vgae_mu, vgae_logvar,
                         plddt, boundary_pred)

