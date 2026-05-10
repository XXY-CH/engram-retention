"""Paper-faithful hashed N-gram Engram lookup.

Engram here is static conditional memory: deterministic N-gram hashes retrieve
embedding rows, then a context-aware gate injects a tiny residual branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HashedNgramEngram(nn.Module):
    """Deterministic multi-head N-gram hash lookup with gated residual fusion."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_slots: int = 4096,
        max_ngram: int = 3,
        num_hash_heads: int = 4,
        init_scale: float = 1e-4,
        gate_bias: float = -3.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_slots = num_slots
        self.max_ngram = max_ngram
        self.num_hash_heads = num_hash_heads

        self.tables = nn.ModuleList(
            [nn.Embedding(num_slots, d_model) for _ in range(max_ngram * num_hash_heads)]
        )
        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=d_model**-0.5)

        self.hidden_norm = nn.RMSNorm(d_model)
        self.memory_norm = nn.RMSNorm(d_model)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        nn.init.constant_(self.gate_proj.bias, gate_bias)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

        salts = torch.arange(1, max_ngram * num_hash_heads + 1, dtype=torch.long)
        self.register_buffer("salts", salts * 0x9E3779B1, persistent=False)

    def _hash_suffixes(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """Return hash indices for all n-gram orders and heads."""
        batch, seq_len = input_ids.shape
        device = input_ids.device
        ids = input_ids.to(torch.long)
        hashes: list[torch.Tensor] = []
        table_idx = 0

        for order in range(1, self.max_ngram + 1):
            h = torch.zeros(batch, seq_len, device=device, dtype=torch.long)
            for offset in range(order):
                shifted = torch.zeros_like(ids)
                if offset == 0:
                    shifted = ids
                else:
                    shifted[:, offset:] = ids[:, :-offset]
                multiplier = 0x85EBCA77 + 0xC2B2AE3D * (offset + 1)
                h = h ^ ((shifted + 1) * multiplier)

            # Avoid pretending incomplete prefixes are full n-grams.
            if order > 1:
                prefix = torch.arange(seq_len, device=device) < (order - 1)
                h[:, prefix] = ids[:, prefix]

            for _ in range(self.num_hash_heads):
                salt = self.salts[table_idx].to(device=device)
                hashes.append((h ^ salt) % self.num_slots)
                table_idx += 1

        return hashes

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return scaled Engram residual and gate values.

        Args:
            hidden: Current hidden states [batch, seq_len, d_model].
            input_ids: Token IDs [batch, seq_len].
        """
        hashes = self._hash_suffixes(input_ids)
        retrieved = [table(index) for table, index in zip(self.tables, hashes)]
        memory = torch.stack(retrieved, dim=0).mean(dim=0)

        norm_hidden = self.hidden_norm(hidden)
        norm_memory = self.memory_norm(memory)
        gate = torch.sigmoid(self.gate_proj(torch.cat([norm_hidden, norm_memory], dim=-1)))
        value = self.value_proj(norm_memory)
        residual = self.residual_scale.abs() * self.dropout(gate * value)
        return residual, gate
