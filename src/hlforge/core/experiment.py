"""Runtime container model for assembled Hydra-Lightning experiments."""

from dataclasses import dataclass, field
from typing import Any

import pytorch_lightning as pl


@dataclass
class Experiment:
    """Container holding the assembled assets for a PyTorch Lightning experiment.

    This class serves as a structured payload encapsulating the core objects
    required to run training, validation, testing, or prediction stages.

    Attributes:
        module: The configured LightningModule.
        datamodule: The configured LightningDataModule.
        trainer: The PyTorch Lightning Trainer instance.
        callbacks: List of callback instances attached to the trainer.
        logger: List of logger instances configured for tracking experiment metrics.
        seed: The global seed value used to enforce reproducibility.
    """

    module: pl.LightningModule
    datamodule: pl.LightningDataModule
    trainer: pl.Trainer
    callbacks: list[pl.Callback] = field(default_factory=list)
    logger: list[Any] = field(default_factory=list)
    seed: int | None = None
