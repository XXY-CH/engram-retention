"""Explicit milestone snapshot cache and readout.

This is the time-axis sparse anchor mechanism restored after high-entropy needle
diagnostics showed retention gates alone are not enough for exact recall.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MilestoneSnapshotReadout(nn.Module):
    """Read from token-time hidden snapshots marked by milestone tokens."""

    def __init__(
        self,
        d_model: int,
        max_snapshots: int = 8,
        init_scale: float = 1e-4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_snapshots = max_snapshots
        self.query_norm = nn.RMSNorm(d_model)
        self.key_norm = nn.RMSNorm(d_model)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def collect(
        self,
        hidden: torch.Tensor,
        milestone_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Collect up to max_snapshots hidden vectors per batch item."""
        if self.max_snapshots <= 0:
            return None

        batch, _, d_model = hidden.shape
        device = hidden.device
        snapshots = hidden.new_zeros(batch, self.max_snapshots, d_model)
        valid = torch.zeros(batch, self.max_snapshots, device=device, dtype=torch.bool)

        any_snapshot = False
        for batch_idx in range(batch):
            positions = torch.nonzero(milestone_mask[batch_idx], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            positions = positions[-self.max_snapshots :]
            count = positions.numel()
            snapshots[batch_idx, :count] = hidden[batch_idx, positions]
            valid[batch_idx, :count] = True
            any_snapshot = True

        if not any_snapshot:
            return None
        return snapshots, valid

    def forward(
        self,
        hidden: torch.Tensor,
        snapshot_cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return scaled residual branch and snapshot attention weights."""
        if snapshot_cache is None:
            return torch.zeros_like(hidden), None

        snapshots, valid = snapshot_cache
        q = self.query_norm(hidden)
        k = self.key_proj(self.key_norm(snapshots))
        v = self.value_proj(snapshots)
        scores = torch.einsum("bsd,bmd->bsm", q, k) / (self.d_model**0.5)
        scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        readout = torch.einsum("bsm,bmd->bsd", weights, v)
        return self.residual_scale * self.out_proj(readout), weights
