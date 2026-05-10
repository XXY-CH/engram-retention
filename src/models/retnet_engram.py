"""Dense RetNet + Block AttnRes + hashed Engram language model.

This is the small practice architecture aligned with the current proof stack:
- RetNet: small-state sequence mixing.
- Dense FFN: phase-1 channel-mixing control baseline; MoE is deferred.
- Block AttnRes: depth-axis residual readout.
- Engram: deterministic N-gram hash lookup with gated residual injection.
- Milestone gate: optional project extension for time-axis preservation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..layers.attention_residual import BlockAttentionResidual
from ..layers.engram import HashedNgramEngram
from ..layers.milestone_gate import MilestoneRetentionGate
from ..layers.milestone_snapshot import MilestoneSnapshotReadout
from ..layers.retention import RetentionLayer
from ..layers.token_copy_buffer import TokenCopyBuffer


@dataclass
class RetNetEngramConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 8
    d_ff: int | None = None
    max_seq_len: int = 2048
    dropout: float = 0.0
    engram_layers: tuple[int, ...] = (2,)
    engram_num_slots: int = 4096
    engram_max_ngram: int = 3
    engram_hash_heads: int = 4
    attnres_every: int = 4
    attnres_max_sources: int = 8
    attnres_distance_penalty: float = 0.0
    branch_init_scale: float = 1e-4
    milestone_token_ids: tuple[int, ...] = field(default_factory=tuple)
    milestone_ttl: int = 256
    milestone_gamma: float = 0.999
    use_milestone_snapshots: bool = False
    max_milestone_snapshots: int = 8
    use_snapshot_logit_bias: bool = False
    snapshot_logit_scale: float = 1.0
    use_token_copy_buffer: bool = False


class DenseRetNetEngramLayer(nn.Module):
    """One Dense RetNet block with optional Engram and Block AttnRes branches."""

    def __init__(
        self,
        config: RetNetEngramConfig,
        layer_idx: int,
        use_engram: bool,
        use_attnres: bool,
    ) -> None:
        super().__init__()
        d_ff = config.d_ff or config.d_model * 4
        self.layer_idx = layer_idx
        self.use_engram = use_engram
        self.use_attnres = use_attnres

        self.retention_norm = nn.RMSNorm(config.d_model)
        self.retention = RetentionLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        self.ffn_norm = nn.RMSNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_ff, config.d_model),
        )

        self.engram = (
            HashedNgramEngram(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                num_slots=config.engram_num_slots,
                max_ngram=config.engram_max_ngram,
                num_hash_heads=config.engram_hash_heads,
                init_scale=config.branch_init_scale,
                dropout=config.dropout,
            )
            if use_engram
            else None
        )
        self.attnres = (
            BlockAttentionResidual(
                d_model=config.d_model,
                max_sources=config.attnres_max_sources,
                init_scale=config.branch_init_scale,
                distance_penalty=config.attnres_distance_penalty,
                dropout=config.dropout,
            )
            if use_attnres
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        depth_sources: list[torch.Tensor],
        retention_gate: torch.Tensor | None,
        disable_engram: bool = False,
        disable_attnres: bool = False,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None], dict[str, torch.Tensor]]:
        metrics: dict[str, torch.Tensor | None] = {}
        diagnostics: dict[str, torch.Tensor] = {}

        u = self.retention_norm(x)
        retention_out = self.retention(u, retention_gate=retention_gate)
        if not isinstance(retention_out, torch.Tensor):
            retention_out = retention_out[0]

        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + retention_out + ffn_out

        if self.engram is not None and not disable_engram:
            engram_residual, engram_gate = self.engram(self.ffn_norm(x), input_ids)
            x = x + engram_residual
            metrics[f"layer_{self.layer_idx}_engram_gate_mean"] = engram_gate.mean()
            metrics[f"layer_{self.layer_idx}_engram_scale"] = self.engram.residual_scale.detach()
            if return_diagnostics:
                diagnostics[f"layer_{self.layer_idx}_engram_gate"] = engram_gate.detach()

        if self.attnres is not None and not disable_attnres:
            attnres_residual, attn_weights = self.attnres(self.ffn_norm(x), depth_sources)
            x = x + attnres_residual
            metrics[f"layer_{self.layer_idx}_attnres_scale"] = self.attnres.residual_scale.detach()
            if attn_weights is not None:
                metrics[f"layer_{self.layer_idx}_attnres_entropy"] = (
                    -(attn_weights.clamp_min(1e-9).log() * attn_weights).sum(dim=-1).mean()
                )
                if return_diagnostics:
                    diagnostics[f"layer_{self.layer_idx}_attnres_weights"] = attn_weights.detach()

        return x, metrics, diagnostics


class RetNetEngramModel(nn.Module):
    """Small language model aligned to the Dense-first phase-1 proof architecture."""

    def __init__(self, config: RetNetEngramConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        engram_layers = set(config.engram_layers)
        self.layers = nn.ModuleList(
            [
                DenseRetNetEngramLayer(
                    config=config,
                    layer_idx=i,
                    use_engram=i in engram_layers,
                    use_attnres=config.attnres_every > 0 and (i + 1) % config.attnres_every == 0,
                )
                for i in range(config.n_layers)
            ]
        )
        self.milestone_gate = MilestoneRetentionGate(
            n_heads=config.n_heads,
            milestone_token_ids=config.milestone_token_ids,
            protected_gamma=config.milestone_gamma,
            ttl=config.milestone_ttl,
        )
        self.snapshot_readout = (
            MilestoneSnapshotReadout(
                d_model=config.d_model,
                max_snapshots=config.max_milestone_snapshots,
                init_scale=config.branch_init_scale,
                dropout=config.dropout,
            )
            if config.use_milestone_snapshots
            else None
        )
        self.token_copy_buffer = (
            TokenCopyBuffer(
                d_model=config.d_model,
                max_snapshots=config.max_milestone_snapshots,
                init_scale=config.branch_init_scale,
                dropout=config.dropout,
            )
            if config.use_token_copy_buffer
            else None
        )
        self.final_norm = nn.RMSNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_metrics: bool = False,
        disable_engram: bool = False,
        disable_attnres: bool = False,
        disable_snapshots: bool = False,
        return_diagnostics: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, dict[str, torch.Tensor | None]]
        | tuple[torch.Tensor, dict[str, torch.Tensor | None], dict[str, torch.Tensor]]
    ):
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))

        retention_gate = self.milestone_gate(
            input_ids,
            base_gamma=self.layers[0].retention.gamma,
        )
        milestone_mask = self._milestone_mask(input_ids)
        snapshot_source_mask = self._pre_milestone_mask(milestone_mask)

        token_copy_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.token_copy_buffer is not None and not disable_snapshots:
            token_emb = self.token_embedding(input_ids)
            copy_source_mask = self._content_before_milestone_mask(milestone_mask)
            collected = self.token_copy_buffer.collect(token_emb, copy_source_mask)
            if collected is not None:
                token_copy_cache = collected

        depth_sources: list[torch.Tensor] = []
        metrics: dict[str, torch.Tensor | None] = {}
        diagnostics: dict[str, torch.Tensor] = {}
        snapshot_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        for layer in self.layers:
            x, layer_metrics, layer_diagnostics = layer(
                x=x,
                input_ids=input_ids,
                depth_sources=depth_sources,
                retention_gate=retention_gate,
                disable_engram=disable_engram,
                disable_attnres=disable_attnres,
                return_diagnostics=return_diagnostics,
            )
            metrics.update(layer_metrics)
            diagnostics.update(layer_diagnostics)
            if self.snapshot_readout is not None and not disable_snapshots:
                collected = self.snapshot_readout.collect(x, snapshot_source_mask)
                if collected is not None:
                    snapshot_cache = collected
            depth_sources.append(x.detach() if not self.training else x)

        if self.snapshot_readout is not None and not disable_snapshots:
            snapshot_residual, snapshot_weights = self.snapshot_readout(
                self.final_norm(x),
                snapshot_cache,
            )
            x = x + snapshot_residual
            metrics["snapshot_scale"] = self.snapshot_readout.residual_scale.detach()
            if snapshot_weights is not None:
                metrics["snapshot_entropy"] = (
                    -(snapshot_weights.clamp_min(1e-9).log() * snapshot_weights).sum(dim=-1).mean()
                )
                metrics["snapshot_valid_count"] = snapshot_cache[1].sum(dim=-1).float().mean()
                if return_diagnostics:
                    diagnostics["snapshot_weights"] = snapshot_weights.detach()
                    diagnostics["snapshot_valid"] = snapshot_cache[1].detach()

        final_hidden = self.final_norm(x)
        logits = self.output_head(final_hidden)
        if self.token_copy_buffer is not None and not disable_snapshots and token_copy_cache is not None:
            copy_readout, copy_weights = self.token_copy_buffer(final_hidden, token_copy_cache)
            copy_logits = torch.matmul(copy_readout, self.token_embedding.weight.t())
            logits = logits + copy_logits
            if return_metrics:
                metrics["token_copy_scale"] = self.token_copy_buffer.residual_scale.abs().detach()
                if copy_weights is not None:
                    metrics["token_copy_entropy"] = (
                        -(copy_weights.clamp_min(1e-9).log() * copy_weights).sum(dim=-1).mean()
                    )
        if (
            self.snapshot_readout is not None
            and self.config.use_snapshot_logit_bias
            and not disable_snapshots
            and snapshot_cache is not None
        ):
            snapshot_logits, _ = self.snapshot_readout(final_hidden, snapshot_cache)
            logits = logits + self.config.snapshot_logit_scale * torch.matmul(
                snapshot_logits,
                self.token_embedding.weight.t(),
            )
        if return_metrics:
            if retention_gate is not None:
                metrics["milestone_gate_mean"] = retention_gate.mean()
                if return_diagnostics:
                    diagnostics["milestone_gate"] = retention_gate.detach()
            if return_diagnostics:
                return logits, metrics, diagnostics
            return logits, metrics
        if return_diagnostics:
            return logits, metrics, diagnostics
        return logits

    def _milestone_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.config.milestone_token_ids:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.config.milestone_token_ids:
            mask |= input_ids == token_id
        return mask

    @staticmethod
    def _pre_milestone_mask(milestone_mask: torch.Tensor) -> torch.Tensor:
        source_mask = torch.zeros_like(milestone_mask)
        source_mask[:, :-1] = milestone_mask[:, 1:]
        return source_mask

    @staticmethod
    def _content_before_milestone_mask(milestone_mask: torch.Tensor) -> torch.Tensor:
        """Mark all positions before any milestone for token copy buffer."""
        has_future = milestone_mask.flip(dims=[1]).cummax(dim=1).values.flip(dims=[1])
        source_mask = has_future.roll(-1, dims=1)
        source_mask[:, -1] = False
        return source_mask & ~milestone_mask
