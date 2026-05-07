"""Milestone-triggered retention gate helpers."""

from __future__ import annotations

import torch
import torch.nn as nn


class MilestoneRetentionGate(nn.Module):
    """Build per-step retention gates from explicit milestone token IDs.

    This is an optional project extension. The gate is intentionally simple and
    measurable: a milestone opens a protected window for a bounded number of
    future steps, with a gate value close to one.
    """

    def __init__(
        self,
        n_heads: int,
        milestone_token_ids: tuple[int, ...] = (),
        protected_gamma: float = 0.999,
        ttl: int = 256,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.milestone_token_ids = tuple(milestone_token_ids)
        self.protected_gamma = protected_gamma
        self.ttl = ttl

    def forward(
        self,
        input_ids: torch.Tensor,
        base_gamma: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return [batch, seq_len, n_heads] gate values or None if disabled."""
        if not self.milestone_token_ids:
            return None

        batch, seq_len = input_ids.shape
        device = input_ids.device
        protected = torch.zeros(batch, seq_len, device=device, dtype=torch.bool)
        marks = torch.zeros_like(protected)
        for token_id in self.milestone_token_ids:
            marks |= input_ids == token_id

        for offset in range(self.ttl):
            if offset == 0:
                protected |= marks
            else:
                protected[:, offset:] |= marks[:, :-offset]

        gamma = base_gamma.to(device=device).view(1, 1, self.n_heads)
        gate = gamma.expand(batch, seq_len, self.n_heads).clone()
        gate = torch.where(
            protected.unsqueeze(-1),
            torch.maximum(gate, torch.full_like(gate, self.protected_gamma)),
            gate,
        )
        return gate
