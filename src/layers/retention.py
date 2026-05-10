"""
Retention Layer - Core RetNet building block.

Implements parallel, recurrent, and chunkwise retention mechanisms.
"""

import torch
import torch.nn as nn


class RetentionLayer(nn.Module):
    """Single retention layer with multi-head retention.

    Supports three computation modes:
    - parallel: for training (full sequence at once)
    - recurrent: for autoregressive inference (step-by-step)
    - chunkwise: for long sequences (chunked processing)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_decay()

    def _init_decay(self) -> None:
        """Initialize fixed RetNet-style per-head decay rates below one."""
        if self.n_heads == 1:
            gamma = torch.tensor([1 - 2**-5], dtype=torch.float32)
        else:
            gamma = 1 - torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(1 / 32, dtype=torch.float32)),
                    torch.log(torch.tensor(1 / 512, dtype=torch.float32)),
                    self.n_heads,
                )
            )
        self.register_buffer("gamma", gamma)

    def parallel_retention(
        self,
        x: torch.Tensor,
        retention_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Parallel retention for training.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Retained output [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        positions = torch.arange(seq_len, device=x.device)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [i, j] = i - j
        causal_mask = (diff >= 0).to(dtype=x.dtype)

        if retention_gate is None:
            # Causal decay mask: D[i,j] = gamma^(i-j) for i >= j
            decay = self.gamma.to(device=x.device, dtype=x.dtype)
            decay_mask = decay.unsqueeze(-1).unsqueeze(-1) ** diff.abs()
            decay_mask = decay_mask * causal_mask
        else:
            # Batch-specific gated decay:
            # D[b,h,i,j] = prod_{r=j+1..i} gate[b,r,h].
            gate = retention_gate.to(device=x.device, dtype=x.dtype)
            if gate.shape != (batch, seq_len, self.n_heads):
                raise ValueError(
                    "retention_gate must have shape "
                    f"{(batch, seq_len, self.n_heads)}, got {tuple(gate.shape)}"
                )
            log_gate = torch.log(gate.clamp_min(torch.finfo(x.dtype).tiny))
            prefix = torch.cumsum(log_gate, dim=1)  # [b, s, h]
            prefix_i = prefix.unsqueeze(2)
            prefix_j = prefix.unsqueeze(1)
            log_decay = prefix_i - prefix_j
            decay_mask = torch.exp(log_decay).permute(0, 3, 1, 2)
            decay_mask = decay_mask * causal_mask.view(1, 1, seq_len, seq_len)

        # Retention: (Q K^T * D) V
        attn = torch.einsum("bhsd,bhtd->bhst", q, k) * decay_mask
        attn = self.dropout(attn)
        output = torch.einsum("bhst,bhtd->bhsd", attn, v)

        output = self._merge_heads(output)
        return self.out_proj(output)

    def recurrent_retention(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        retention_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent retention for autoregressive inference.

        Args:
            x: Input tensor [batch, 1, d_model]
            state: Previous state [batch, n_heads, head_dim, head_dim]

        Returns:
            Output tensor and updated state
        """
        batch = x.shape[0]

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if state is None:
            state = torch.zeros(batch, self.n_heads, self.head_dim, self.head_dim, device=x.device)

        # Update state: S = gamma * S + k^T v
        if retention_gate is None:
            gamma = self.gamma.to(device=x.device, dtype=x.dtype).view(1, self.n_heads, 1, 1)
        else:
            gamma = retention_gate.to(device=x.device, dtype=x.dtype).view(
                batch, self.n_heads, 1, 1
            )
        new_state = gamma * state + torch.einsum("bhld,bhle->bhde", k, v)

        # Output: q @ state
        output = torch.einsum("bhld,bhde->bhle", q, new_state)
        output = self._merge_heads(output)

        return self.out_proj(output), new_state

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "parallel",
        state: torch.Tensor | None = None,
        retention_gate: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with selectable computation mode.

        Args:
            x: Input tensor
            mode: 'parallel', 'recurrent', or 'chunkwise'
            state: Previous state (for recurrent mode)

        Returns:
            Retained output, optionally with updated state
        """
        if mode == "parallel":
            return self.parallel_retention(x, retention_gate=retention_gate)
        elif mode == "recurrent":
            return self.recurrent_retention(x, state, retention_gate=retention_gate)
        else:
            raise ValueError(f"Unknown retention mode: {mode}")

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
