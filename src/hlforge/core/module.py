from collections.abc import Callable, Mapping
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import Optimizer
from torchmetrics import Metric, MetricCollection

from hlforge.core.configs import LRSchedulerConfig


class CompositeModel(nn.Module):
    """Modular neural network composed of backbone, optional neck, and optional head."""

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module | None = None,
        head: nn.Module | None = None,
    ) -> None:
        """Initialize composite architecture sub-modules.

        Args:
            backbone: Core feature extractor network.
            neck: Optional intermediate feature transformation layer (e.g. FPN, pooling).
            head: Optional task-specific output projection layer (e.g. classification head).
        """
        super().__init__()
        self.backbone: nn.Module = backbone
        self.neck: nn.Module | None = neck
        self.head: nn.Module | None = head

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Sequential forward pass through backbone -> neck -> head.

        Args:
            x: Input tensor.
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            The output tensor.
        """
        out = self.backbone(x, *args, **kwargs)
        if self.neck is not None:
            out = self.neck(out)
        if self.head is not None:
            out = self.head(out)
        return out


class BaseLightningModule(pl.LightningModule):
    """Hydra-instantiable base LightningModule with modular sub-components.

    Supports decoupled models (or backbone/neck/head composition), loss functions
    with regularizers, partially instantiated optimizers and learning rate schedulers,
    TorchMetrics collections, and post-processors.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        criterion: nn.Module | Callable[..., Tensor] | None = None,
        optimizer_config: Optimizer | None = None,
        optimizer: Optimizer | None = None,
        lr_scheduler: LRSchedulerConfig | None = None,
        log_config: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the BaseLightningModule and configure all sub-components.

        Args:
            model: The neural network model instance.
            criterion: The loss function or criterion.
            optimizer_config: List or dict configuring optimizers and schedulers.
            optimizer: Optional direct optimizer factory (single).
            lr_scheduler: Optional direct scheduler factory (single).
            log_config: Configuration for logging parameters.
            *args: Additional positional arguments for pl.LightningModule.
            **kwargs: Additional keyword arguments for pl.LightningModule. These arguments will be stored in `hparams`. Supported keys:
                post_processor (Optional[Callable[[Tensor], Any]]): Callable to transform model outputs before metrics.
                loss_wrapper (Optional[Callable[..., Tensor]]): Callable to wrap the loss calculation.
                regularizer (Optional[Callable[..., Tensor]]): Callable to compute regularization losses.
                train_metrics (Optional[Union[MetricCollection, Metric, dict[str, Metric]]]): Metrics calculated during training.
                val_metrics (Optional[Union[MetricCollection, Metric, dict[str, Metric]]]): Metrics calculated during validation.
                test_metrics (Optional[Union[MetricCollection, Metric, dict[str, Metric]]]): Metrics calculated during testing.
        """
        super().__init__()

        logger = False
        if log_config is not None:
            logger = log_config.get("save_hyperparameters_logger", False)

        self.log_config = log_config

        self.save_hyperparameters(
            logger=logger,
            ignore=[
                "model",
                "criterion",
                "optimizer_config",
                "optimizer",
                "lr_scheduler",
                "log_config",
            ],
        )

        # Model and Criterion assembly
        self.model: nn.Module | None = model
        self.criterion: nn.Module | Callable[..., Tensor] | None = criterion

        # Resolve optimizer configuration
        if optimizer_config is None and (optimizer is not None or lr_scheduler is not None):
            self.optimizer_config = [{"optimizer": optimizer, "lr_scheduler": lr_scheduler}]
        else:
            self.optimizer_config = optimizer_config or []

        self.post_processor: Callable[[Tensor], Any] | None = self.hparams.get("post_processor")
        self.loss_wrapper: Callable[..., Tensor] | None = self.hparams.get("loss_wrapper")
        self.regularizer: Callable[..., Tensor] | None = self.hparams.get("regularizer")

        # Metric collections registration
        self.train_metrics: MetricCollection | None = self._prepare_metrics(
            metrics=self.hparams.get("train_metrics"), prefix="train/"
        )
        self.val_metrics: MetricCollection | None = self._prepare_metrics(
            metrics=self.hparams.get("val_metrics"), prefix="val/"
        )
        self.test_metrics: MetricCollection | None = self._prepare_metrics(
            metrics=self.hparams.get("test_metrics"), prefix="test/"
        )

    @staticmethod
    def _prepare_metrics(
        metrics: MetricCollection | Metric | Mapping[str, Metric] | None,
        prefix: str,
    ) -> MetricCollection | None:
        """Wrap input metrics into a MetricCollection with an optional prefix.

        Args:
            metrics: Metrics to be collected.
            prefix: String prefix to prepend to metric names.

        Returns:
            A MetricCollection if metrics are provided, else None.
        """
        if metrics is None:
            return None
        if isinstance(metrics, MetricCollection):
            return metrics
        if isinstance(metrics, Metric):
            return MetricCollection([metrics], prefix=prefix)
        if isinstance(metrics, Mapping):
            return MetricCollection(dict(metrics), prefix=prefix)
        try:
            return MetricCollection(dict(metrics), prefix=prefix)
        except (TypeError, ValueError):
            return None

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Any:
        """Forward pass delegated to the underlying model.

        Args:
            x: Input tensor.
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            The model output.
        """
        return self.model(x, *args, **kwargs)

    def _unpack_batch(self, batch: tuple | list | dict) -> tuple[Any, Any | None]:
        """Extract input features (x) and targets (y) from standard batch formats.

        Args:
            batch: The input batch, expected to be tuple, list, or dict.

        Returns:
            A tuple of (features, targets). Targets may be None.
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                return batch[0], None
            return batch[0], batch[1]
        if isinstance(batch, dict):
            if "x" in batch and "y" in batch:
                return batch["x"], batch["y"]
            if "inputs" in batch and "targets" in batch:
                return batch["inputs"], batch["targets"]
            if "data" in batch and "target" in batch:
                return batch["data"], batch["target"]
            if "image" in batch and "label" in batch:
                return batch["image"], batch["label"]
            if "labels" in batch:
                return batch, batch["labels"]
        return batch, None

    def _shared_step(self, batch: tuple | list | dict, stage: str) -> Tensor:
        """Shared step for training, validation, and testing.

        Args:
            batch: The input batch.
            stage: The current training stage ('train', 'val', 'test').

        Returns:
            The computed loss tensor.
        """

        if self.model is None:
            raise ValueError("A model must be configured to use _shared_step.")

        x, y = self._unpack_batch(batch=batch)
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y, batch=batch) if self.criterion is not None else None

        # Log loss
        if loss is not None:
            log_config: dict[str, Any] = self.log_config or {}
            log_loss_config: dict[str, Any] = log_config.get("loss", {})
            self.log(
                f"{stage}_loss",
                loss,
                on_step=log_loss_config.get("on_step", stage == "train"),
                on_epoch=log_loss_config.get("on_epoch", True),
                prog_bar=log_loss_config.get("prog_bar", True),
                sync_dist=log_loss_config.get("sync_dist", True),
            )

        # Update metrics
        metrics = getattr(self, f"{stage}_metrics")
        if metrics is not None and y is not None:
            preds = self.post_processor(y_hat) if self.post_processor is not None else y_hat
            metrics.update(preds, y)
            log_metrics_config: dict[str, Any] = (self.log_config or {}).get("metrics", {})
            self.log_dict(
                metrics,
                on_step=log_metrics_config.get("on_step", False),
                on_epoch=log_metrics_config.get("on_epoch", True),
                prog_bar=log_metrics_config.get("prog_bar", True),
                sync_dist=log_metrics_config.get("sync_dist", True),
            )

        return loss

    def compute_loss(self, y_hat: Any, y: Any, batch: Any = None) -> Tensor:
        """Compute training/validation loss using criterion, loss_wrapper, and regularizers.

        Args:
            y_hat: Model output predictions.
            y: Target values.
            batch: The input batch (optional, used by loss_wrapper/regularizers).

        Returns:
            The computed loss tensor.
        """
        if self.criterion is None:
            raise ValueError("A criterion must be configured to compute loss, or override compute_loss().")

        if self.loss_wrapper is not None:
            loss = self.loss_wrapper(self.criterion, y_hat, y, batch=batch)
        else:
            loss = self.criterion(y_hat, y)

        if self.regularizer is not None:
            reg_loss = self.regularizer(self.model, batch=batch)
            loss = loss + reg_loss

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        """Standard training step.

        Args:
            batch: The input batch.
            batch_idx: Index of the current batch.

        Returns:
            The loss tensor.
        """
        loss = self._shared_step(batch=batch, stage="train")
        if not isinstance(loss, Tensor):
            raise TypeError("Training step must return a tensor loss.")
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        """Validation step.

        Args:
            batch: The input batch.
            batch_idx: Index of the current batch.

        Returns:
            None.
        """
        _ = self._shared_step(batch=batch, stage="val")

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        """Test step.

        Args:
            batch: The input batch.
            batch_idx: Index of the current batch.

        Returns:
            None.
        """
        _ = self._shared_step(batch=batch, stage="test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int | None = None) -> Any:
        """Prediction step.

        Args:
            batch: The input batch.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the dataloader.

        Returns:
            The model predictions.
        """
        x, _ = self._unpack_batch(batch=batch)
        y_hat = self(x)

        if self.post_processor is not None:
            return self.post_processor(y_hat)
        return y_hat

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Instantiate optimizers and learning rate schedulers from config.

        Returns:
            The configured optimizers and schedulers.
        """
        _optimizer_lrscheduler = []
        for item in self.optimizer_config:
            opt_factory = item["optimizer"]
            sched_factory = item.get("lr_scheduler")

            if callable(opt_factory):
                optimizer = opt_factory([p for p in self.parameters() if p.requires_grad])
            else:
                optimizer = opt_factory

            if sched_factory is None:
                _optimizer_lrscheduler.append({"optimizer": optimizer})
                continue

            if isinstance(sched_factory, LRSchedulerConfig):
                lr_scheduler = {k: v for k, v in sched_factory.__dict__.items() if v is not None and k != "scheduler"}
                sched_obj = sched_factory.scheduler
                lr_scheduler["scheduler"] = sched_obj(optimizer=optimizer) if callable(sched_obj) else sched_obj
            elif callable(sched_factory):
                lr_scheduler = {"scheduler": sched_factory(optimizer=optimizer)}
            elif isinstance(sched_factory, dict):
                lr_scheduler = dict(sched_factory)
                if callable(lr_scheduler.get("scheduler")):
                    lr_scheduler["scheduler"] = lr_scheduler["scheduler"](optimizer=optimizer)
            else:
                lr_scheduler = {"scheduler": sched_factory}

            _optimizer_lrscheduler.append(
                {
                    "optimizer": optimizer,
                    "lr_scheduler": lr_scheduler,
                }
            )

        if len(_optimizer_lrscheduler) == 1:
            return _optimizer_lrscheduler[0]
        return _optimizer_lrscheduler
