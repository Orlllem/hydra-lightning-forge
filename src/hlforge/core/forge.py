"""Hydra-driven assembly helpers for constructing experiment objects."""

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from hlforge.core.experiment import Experiment


class ExperimentForge:
    """Factory class to assemble PyTorch Lightning components from a Hydra configuration.

    The forge orchestrates the instantiation of the model, datamodule, and trainer,
    and configures the seeding for reproducibility.
    """

    @staticmethod
    def assemble(cfg: DictConfig) -> Experiment:
        """Create an Experiment container from a Hydra config object.

        This method handles:
        1. Seeding the global state (if `seed` is specified in config).
        2. Instantiating the model, datamodule, and trainer from config.
        3. Packaging the results into an Experiment payload.

        Args:
            cfg: Hydra configuration object describing the experiment assembly.

        Returns:
            Experiment: An initialized runtime payload containing the trainer,
            model, datamodule, callbacks, logger, and seed.
        """
        if cfg.get("seed") is not None:
            pl.seed_everything(cfg.seed, workers=True)

        datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
        module: pl.LightningModule = instantiate(cfg.module)
        trainer: pl.Trainer = instantiate(cfg.trainer)

        return Experiment(
            module=module,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=trainer.callbacks,
            logger=trainer.loggers,
            seed=cfg.get("seed"),
        )
