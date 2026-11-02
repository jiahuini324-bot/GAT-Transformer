import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch


class GraphTransformer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.position = nn.Linear(3, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x, positions, batch):
        x, mask = to_dense_batch(x + self.position(positions), batch)
        attended, _ = self.attention(
            x, x, x, key_padding_mask=~mask, need_weights=False
        )
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.feedforward(x)))
        return x[mask]


class GATTransformer(nn.Module):
    def __init__(
        self,
        node_dim=15,
        edge_dim=8,
        global_dim=13,
        hidden_dim=128,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.node_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.edge_projection = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU())
        self.gat_layers = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    concat=False,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.transformer = GraphTransformer(hidden_dim, num_heads, dropout)
        self.head_xy = self.make_head(hidden_dim + global_dim, hidden_dim, 2, dropout)
        self.head_z = self.make_head(hidden_dim + global_dim, hidden_dim, 1, dropout)

    @staticmethod
    def make_head(input_dim, hidden_dim, output_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = data.x.new_zeros(data.num_nodes, dtype=torch.long)
        h = self.node_projection(data.x)
        edges = self.edge_projection(data.edge_attr)
        for layer in self.gat_layers:
            h = self.dropout(F.leaky_relu(layer(h, data.edge_index, edges), 0.2))
        h = self.transformer(h, data.x[:, :3], batch)
        h = torch.cat((h, data.global_attr[batch]), dim=-1)
        return torch.cat((self.head_xy(h), self.head_z(h)), dim=-1)


class DecoupledPhysicalMSE(nn.Module):
    def __init__(self, scale=(1.0, 1.0, 1.0)):
        super().__init__()
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if scale.shape != (3,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("scale must contain three positive finite values")
        self.register_buffer("scale", scale.clone())

    def forward(self, prediction, target):
        squared = ((prediction.float() - target.float()) * self.scale).square()
        loss_xy = squared[:, :2].sum(dim=-1).mean()
        loss_z = squared[:, 2].mean()
        return loss_xy + loss_z
