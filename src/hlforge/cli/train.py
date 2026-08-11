"""Command-line entrypoint for Hydra-Lightning Forge experiments."""

import logging

import hydra
import torch
from omegaconf import DictConfig

from hlforge.core.forge import ExperimentForge
from hlforge.core.runner import ExperimentRunner

# Patch torch.load to disable weights_only validation globally for local experiment execution
# PyTorch 2.6 defaults weights_only to True, which breaks unpickling of custom configuration/structures in PyTorch Lightning checkpoints.
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

logger = logging.getLogger(__name__)


def run_experiment(cfg: DictConfig) -> None:
    """Execute a Hydra-configured training and optional test pipeline.

    Args:
        cfg: Hydra configuration object describing the experiment assembly.

    Returns:
        None: The runner logs the outcome through the package logger.
    """
    # 1. ASSEMBLY PHASE
    experiment = ExperimentForge.assemble(cfg=cfg)

    # 2. EXECUTION PHASE
    runner = ExperimentRunner(experiment=experiment)
    results: dict[str, object] = runner.run(
        train=cfg.get("train", True),
        test=cfg.get("test", True),
    )
    logger.info("Execution completed successfully.")
    logger.info("Final metrics: %s", results)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra CLI wrapper that delegates to the experiment executor.

    The CLI accepts config selection at runtime through Hydra's
    ``--config-path`` / ``--config-name`` overrides.  We intentionally do not
    hard-code a default config location here because the repository ships the
    example configuration tree outside the installed ``hlforge`` package.

    Args:
        cfg: Hydra runtime configuration for the experiment.

    Returns:
        None
    """
    run_experiment(cfg=cfg)


if __name__ == "__main__":
    main()
