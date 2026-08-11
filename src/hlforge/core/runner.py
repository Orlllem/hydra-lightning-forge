"""Execution lifecycle helpers for an assembled experiment object."""

from typing import Any

from hlforge.core.experiment import Experiment


class ExperimentRunner:
    """Orchestrator to run training and/or testing lifecycle phases of an Experiment.

    Delegates model execution and data orchestration directly to PyTorch Lightning's
    Trainer interface and aggregates performance metrics.
    """

    def __init__(self, experiment: Experiment) -> None:
        """Initialize the runner with an Experiment payload.

        Args:
            experiment: Experiment payload containing the model, datamodule, and trainer.
        """
        self.exp: Experiment = experiment

    def run(self, train: bool = True, test: bool = True) -> dict[str, Any]:
        """Execute the requested stages (training and/or test) in the experiment lifecycle.

        If training is enabled, executes `trainer.fit`.
        If testing is enabled, executes `trainer.test`. If a checkpoint callback is present
        and training ran, automatically loads the "best" model checkpoint path.

        Args:
            train: Whether to run model training and validation (fit stage).
            test: Whether to run model testing (test stage).

        Returns:
            dict[str, Any]: A dictionary capturing final metric results, containing keys
            "train_metrics" (mapping of final metric values) and "test_metrics" (test outputs).
        """
        results: dict[str, Any] = {}

        if train:
            self.exp.trainer.fit(
                model=self.exp.module,
                datamodule=self.exp.datamodule,
            )
            results["train_metrics"] = self.exp.trainer.callback_metrics

        if test:
            trainer = self.exp.trainer
            has_checkpoint = (
                train
                and not getattr(trainer, "fast_dev_run", False)
                and getattr(trainer, "checkpoint_callback", None) is not None
            )
            test_results = self.exp.trainer.test(
                model=self.exp.module,
                datamodule=self.exp.datamodule,
                ckpt_path="best" if has_checkpoint else None,
            )
            results["test_metrics"] = test_results

        return results
