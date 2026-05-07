import torch

from src.layers.retention import RetentionLayer


def test_retention_decay_is_contracting() -> None:
    layer = RetentionLayer(d_model=8, n_heads=4)

    assert not isinstance(layer.gamma, torch.nn.Parameter)
    assert torch.all(layer.gamma > 0)
    assert torch.all(layer.gamma < 1)


def test_recurrent_retention_keeps_state_shape() -> None:
    layer = RetentionLayer(d_model=8, n_heads=4)
    x = torch.randn(2, 1, 8)

    output, state = layer(x, mode="recurrent")

    assert output.shape == (2, 1, 8)
    assert state.shape == (2, 4, 2, 2)
