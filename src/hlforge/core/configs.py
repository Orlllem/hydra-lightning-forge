"""Configuration structures and dataclasses for Hydra-Lightning Forge."""

from dataclasses import dataclass


@dataclass
class LRSchedulerConfig:
    """Configuration container for PyTorch learning rate schedulers.

    Wraps parameters required to configure a PyTorch learning rate scheduler within
    PyTorch Lightning's optimizer configuration dictionary.

    Attributes:
        scheduler: The instantiated scheduler or a factory callable that accepts an optimizer.
        name: Optional name for logging/tracking the learning rate.
        interval: Unit of time to update the scheduler ("epoch" or "step").
        frequency: Number of intervals between scheduler updates.
        reduce_on_plateau: Flag indicating if the scheduler is ReduceLROnPlateau.
        monitor: Metric name to monitor for ReduceLROnPlateau.
        strict: Whether to crash if the monitor metric is not found.
    """

    scheduler: object
    name: str | None = None
    interval: str = "epoch"
    frequency: int = 1
    reduce_on_plateau: bool = False
    monitor: str | None = None
    strict: bool = True
