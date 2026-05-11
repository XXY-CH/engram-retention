"""Recurrent state container for O(1) per-step inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RecurrentState:
    """State carried across recurrent inference steps.

    All tensors are constant-size — memory does not grow with sequence length.
    """

    position: int = 0

    # Per-layer retention recurrent state: [batch, n_heads, head_dim, head_dim]
    retention_states: list[torch.Tensor] = field(default_factory=list)

    # TokenCopyBuffer state
    copy_stored: torch.Tensor | None = None  # [batch, max_snapshots, d_model]
    copy_valid: torch.Tensor | None = None  # [batch, max_snapshots]
    copy_pos_ids: torch.Tensor | None = None  # [batch, max_snapshots]
    copy_frozen: bool = False
    copy_write_idx: int = 0

    # SnapshotReadout state
    snap_stored: torch.Tensor | None = None  # [batch, max_snapshots, d_model]
    snap_valid: torch.Tensor | None = None  # [batch, max_snapshots]
    snap_write_idx: int = 0

    # Snapshot timing: post-layer hidden from previous step, for collecting
    # the hidden state immediately before a milestone.
    prev_hidden: torch.Tensor | None = None  # [batch, d_model]

    # Milestone gate: steps since last milestone token (for TTL window).
    steps_since_milestone: int = 0

    @classmethod
    def init_empty(
        cls,
        batch_size: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        max_snapshots: int,
        d_model: int,
        device: torch.device,
        *,
        with_copy_buffer: bool = False,
        with_snapshots: bool = False,
    ) -> RecurrentState:
        return cls(
            retention_states=[
                torch.zeros(batch_size, n_heads, head_dim, head_dim, device=device)
                for _ in range(n_layers)
            ],
            copy_stored=(
                torch.zeros(batch_size, max_snapshots, d_model, device=device)
                if with_copy_buffer
                else None
            ),
            copy_valid=(
                torch.zeros(batch_size, max_snapshots, device=device, dtype=torch.bool)
                if with_copy_buffer
                else None
            ),
            copy_pos_ids=(
                torch.zeros(batch_size, max_snapshots, device=device, dtype=torch.long)
                if with_copy_buffer
                else None
            ),
            snap_stored=(
                torch.zeros(batch_size, max_snapshots, d_model, device=device)
                if with_snapshots
                else None
            ),
            snap_valid=(
                torch.zeros(batch_size, max_snapshots, device=device, dtype=torch.bool)
                if with_snapshots
                else None
            ),
        )
