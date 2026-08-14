"""Tests to verify that core hlforge package operates cleanly without requiring CUDA wheels or optional dependencies."""

import torch

import hlforge


def test_core_imports_without_cuda():
    """Verify that all public exports in hlforge package can be imported and inspected."""
    for symbol in hlforge.__all__:
        assert hasattr(hlforge, symbol), f"Symbol {symbol} missing from hlforge package"


def test_core_modules_cpu_compatibility():
    """Verify core classes function on CPU without requiring CUDA tensors or CUDA wheels."""
    backbone = torch.nn.Linear(10, 5)
    model = hlforge.CompositeModel(backbone=backbone)
    x = torch.randn(2, 10)
    out = model(x)
    assert out.device.type == "cpu"
    assert out.shape == (2, 5)


def test_no_forced_cuda_device_in_core():
    """Verify that default device usage in core hlforge modules defaults to CPU when CUDA is absent."""
    module = hlforge.BaseLightningModule()
    assert module.device.type == "cpu"
