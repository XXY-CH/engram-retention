"""Token Copy Buffer — direct token-embedding storage for exact recall.

Unlike MilestoneSnapshotReadout which stores compressed hidden states,
this module stores raw token embeddings at marked positions and reads
them out via attention, projecting directly to logits through the
token embedding weight matrix.

This bypasses RetNet's recurrent state compression, enabling exact
long-range token copying.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenCopyBuffer(nn.Module):
    """Store and read out token embeddings for exact recall."""

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
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def collect(
        self,
        token_embeddings: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Collect token embeddings at source positions."""
        if self.max_snapshots <= 0:
            return None

        batch, _, d_model = token_embeddings.shape
        device = token_embeddings.device
        stored = token_embeddings.new_zeros(batch, self.max_snapshots, d_model)
        valid = torch.zeros(batch, self.max_snapshots, device=device, dtype=torch.bool)

        any_stored = False
        for batch_idx in range(batch):
            positions = torch.nonzero(source_mask[batch_idx], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            positions = positions[-self.max_snapshots:]
            count = positions.numel()
            stored[batch_idx, :count] = token_embeddings[batch_idx, positions]
            valid[batch_idx, :count] = True
            any_stored = True

        if not any_stored:
            return None
        return stored, valid

    def forward(
        self,
        hidden: torch.Tensor,
        buffer_cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attend over stored token embeddings.

        Returns:
            Scaled residual (in embedding space) and attention weights.
        """
        if buffer_cache is None:
            return torch.zeros_like(hidden), None

        stored, valid = buffer_cache
        q = self.query_norm(hidden)
        k = self.key_proj(stored)
        scores = torch.einsum("bsd,bmd->bsm", q, k) / (self.d_model**0.5)
        scores = scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        readout = torch.einsum("bsm,bmd->bsd", weights, stored)
        return self.residual_scale.abs() * readout, weights
