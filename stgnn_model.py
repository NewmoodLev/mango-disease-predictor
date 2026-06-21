"""
Spatial-Temporal GNN  —  GAT + BiLSTM + Temporal Attention
===========================================================
Architecture overview
─────────────────────
Input (B, T, N, F)
  │
  ├─► SpatialBlock  : 3-layer GATConv with residuals         → (B·T·N, H)
  │     Multi-head attention learns per-edge importance
  │     instead of the fixed aggregation used by GCN.
  │
  ├─► TemporalBlock : 2-layer BiLSTM + self-attention         → (B·N, H)
  │     BiLSTM captures both past and future context.
  │     Attention pools all T steps (not just the last).
  │
  └─► PredHead      : MLP + BN + skip connection → sigmoid   → (B, N)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ─────────────────────────────────────────────────────────
# Temporal Self-Attention
# ─────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """
    Single-head scaled dot-product attention over the time axis.
    Input  : (BN, T, H)
    Output : (BN, H)   — weighted context vector
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.Wq    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5
        self.ln    = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q   = self.Wq(x)
        K   = self.Wk(x)
        V   = self.Wv(x)
        w   = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1
        )                                            # (BN, T, T)
        out = torch.bmm(w, V)                       # (BN, T, H)
        out = self.ln(out + x)                      # residual + LN
        return out[:, -1, :]                        # (BN, H)


# ─────────────────────────────────────────────────────────
# Spatial Block — 3-layer GAT
# ─────────────────────────────────────────────────────────
class SpatialBlock(nn.Module):
    """
    Three GATConv layers with layer-wise and long-range residuals.

    Why GAT over GCN:
      - Learns per-edge attention (important neighbours contribute more).
      - Multi-head averaging stabilises gradients.
      - edge_weight passed as edge_attr so graph structure is respected.
    """
    def __init__(self, in_dim: int, hidden_dim: int,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        head_dim = hidden_dim // heads

        # Layer 1 — expand to hidden_dim via multi-head concat
        # add_self_loops=False: self-loops are already in the graph (added in
        # create_graph). Keeping GAT's internal add_self_loops enabled would
        # call scatter over B*T*N nodes using the original edge_weight tensor,
        # causing an out-of-bounds index error.
        self.gat1 = GATConv(in_dim,     head_dim, heads=heads,
                            dropout=dropout, concat=True,
                            add_self_loops=False)
        # Layer 2 — stay at hidden_dim
        self.gat2 = GATConv(hidden_dim, head_dim, heads=heads,
                            dropout=dropout, concat=True,
                            add_self_loops=False)
        # Layer 3 — single head, stable output
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=1,
                            dropout=dropout, concat=False,
                            add_self_loops=False)

        self.ln1  = nn.LayerNorm(hidden_dim)
        self.ln2  = nn.LayerNorm(hidden_dim)
        self.ln3  = nn.LayerNorm(hidden_dim)

        self.proj = nn.Linear(in_dim, hidden_dim, bias=False)  # long-range residual
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        # --- Layer 1 ---
        h = F.elu(self.ln1(self.gat1(x, edge_index, edge_attr)))
        h = self.drop(h)

        # --- Layer 2 (layer residual) ---
        h2 = self.ln2(self.gat2(h, edge_index, edge_attr) + h)
        h2 = F.elu(h2)
        h2 = self.drop(h2)

        # --- Layer 3 (long-range residual from input) ---
        h3 = self.ln3(self.gat3(h2, edge_index, edge_attr) + self.proj(x))
        h3 = F.elu(h3)
        h3 = self.drop(h3)

        return h3   # (B*T*N, hidden_dim)


# ─────────────────────────────────────────────────────────
# Temporal Block — BiLSTM + Attention
# ─────────────────────────────────────────────────────────
class TemporalBlock(nn.Module):
    """
    2-layer bidirectional LSTM followed by temporal self-attention.
    """
    def __init__(self, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)  # 2H → H
        self.attn = TemporalAttention(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, T, H)
        out, _ = self.lstm(x)              # (BN, T, 2H)
        out    = F.elu(self.proj(out))     # (BN, T, H)
        out    = self.drop(out)
        return self.attn(out)              # (BN, H)


# ─────────────────────────────────────────────────────────
# Prediction Head
# ─────────────────────────────────────────────────────────
class PredHead(nn.Module):
    """MLP with BN, ELU, dropout, and a skip connection to output."""
    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        mid = max(in_dim, 64)
        self.fc1  = nn.Linear(in_dim, mid)
        self.bn1  = nn.BatchNorm1d(mid)
        self.fc2  = nn.Linear(mid, mid // 2)
        self.bn2  = nn.BatchNorm1d(mid // 2)
        self.fc3  = nn.Linear(mid // 2, 1)
        self.skip = nn.Linear(in_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.bn1(self.fc1(x)))
        h = self.drop(h)
        h = F.elu(self.bn2(self.fc2(h)))
        h = self.drop(h)
        return torch.sigmoid(self.fc3(h) + self.skip(x))


# ─────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────
class SpatialTemporalGNN(nn.Module):
    """
    GAT + BiLSTM Spatial-Temporal GNN for per-node infection probability.

    Parameters
    ----------
    node_features : int   — F, input features per node per timestep
    hidden_dim    : int   — H, internal size (must be divisible by gat_heads)
    window_size   : int   — T, number of input timesteps (informational)
    gat_heads     : int   — number of GAT attention heads  (default 4)
    dropout       : float — dropout rate throughout         (default 0.15)
    lstm_layers   : int   — BiLSTM depth                   (default 2)
    """
    def __init__(
        self,
        node_features: int   = 4,
        hidden_dim:    int   = 64,
        window_size:   int   = 7,
        gat_heads:     int   = 4,
        dropout:       float = 0.15,
        lstm_layers:   int   = 2,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.window_size = window_size

        self.spatial  = SpatialBlock(node_features, hidden_dim, gat_heads, dropout)
        self.temporal = TemporalBlock(hidden_dim, lstm_layers, dropout)
        self.head     = PredHead(hidden_dim + 1, dropout)   # +1: last_inf feature

    # ── internal: tile graph B times (NOT B*T) ───────────
    def _expand_graph_B(self, B, N, edge_index, edge_weight, device):
        """Expand graph for B batches only — called per timestep loop."""
        if hasattr(self, "_cached_ei_B") and self._cached_params_B == (B, N, device):
            return self._cached_ei_B, self._cached_ew_B
        E       = edge_index.shape[1]
        offsets = torch.arange(B, device=device).repeat_interleave(E) * N
        ei_b    = edge_index.repeat(1, B) + offsets
        ew_b    = edge_weight.repeat(B) if edge_weight is not None else None
        self._cached_ei_B     = ei_b
        self._cached_ew_B     = ew_b
        self._cached_params_B = (B, N, device)
        return ei_b, ew_b

    # ── forward ──────────────────────────────────────────
    def forward(self, x, edge_index, edge_weight=None):
        """
        x           : (B, T, N, F)
        edge_index  : (2, E)
        edge_weight : (E,) or None
        Returns     : (B, N)  ∈ [0, 1]

        Spatial encoding is done per-timestep (loop T) over B graphs at a time.
        This keeps edge count at B*E instead of B*T*E, which is T× faster on CPU.
        """
        B, T, N, F = x.shape
        last_inf   = x[:, -1, :, 0:1]                   # (B, N, 1)

        # 1. GAT spatial encoding — loop over T timesteps
        ei_b, ew_b = self._expand_graph_B(B, N, edge_index, edge_weight, x.device)
        h_list = []
        for t in range(T):
            x_t   = x[:, t, :, :].reshape(B * N, F)     # (B*N, F)
            h_t   = self.spatial(x_t, ei_b, ew_b)        # (B*N, H)
            h_list.append(h_t)
        h = torch.stack(h_list, dim=1)                   # (B*N, T, H)

        # 2. BiLSTM + attention temporal encoding
        ctx = self.temporal(h)                            # (B*N, H)

        # 3. Prediction
        ctx      = ctx.view(B, N, self.hidden_dim)
        combined = torch.cat([ctx, last_inf], dim=-1)    # (B, N, H+1)
        out      = self.head(combined.view(B * N, -1))   # (B*N, 1)
        return out.view(B, N)