"""Hydra-Lightning Forge package."""

import functools

import torch

# Register functools.partial as safe global for PyTorch >= 2.6 weights_only checkpoint loading
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([functools.partial])

from hlforge.core.datamodule import BaseLightningDataModule, RandomSplitter, TransformedDataset
from hlforge.core.experiment import Experiment
from hlforge.core.forge import ExperimentForge
from hlforge.core.module import BaseLightningModule, CompositeModel
from hlforge.core.runner import ExperimentRunner

__all__ = [
    "BaseLightningDataModule",
    "BaseLightningModule",
    "CompositeModel",
    "Experiment",
    "ExperimentForge",
    "ExperimentRunner",
    "RandomSplitter",
    "TransformedDataset",
]
