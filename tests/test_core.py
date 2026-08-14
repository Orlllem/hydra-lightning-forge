"""Tests for BaseLightningModule, BaseLightningDataModule, and core abstractions."""

from functools import partial

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset
from torchmetrics.classification import MulticlassAccuracy

from hlforge import (
    BaseLightningDataModule,
    BaseLightningModule,
    CompositeModel,
    ExperimentForge,
    ExperimentRunner,
    RandomSplitter,
    TransformedDataset,
)


class DummyDataset(Dataset):
    def __init__(self, size: int = 100, num_classes: int = 2):
        self.x = torch.randn(size, 8)
        self.y = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def test_composite_model():
    backbone = nn.Linear(8, 16)
    neck = nn.ReLU()
    head = nn.Linear(16, 2)
    model = CompositeModel(backbone=backbone, neck=neck, head=head)

    x = torch.randn(4, 8)
    out = model(x)
    assert out.shape == (4, 2)


def test_transformed_dataset():
    base_ds = DummyDataset(size=10)

    def double_transform(t: torch.Tensor) -> torch.Tensor:
        return t * 2.0

    t_ds = TransformedDataset(base_ds, double_transform)
    assert len(t_ds) == 10
    x, _ = t_ds[0]
    orig_x, _ = base_ds[0]
    assert torch.allclose(x, orig_x * 2.0)


def test_base_lightning_module_init_and_forward():
    backbone = nn.Linear(8, 4)
    head = nn.Linear(4, 2)
    model = CompositeModel(backbone=backbone, head=head)
    criterion = nn.CrossEntropyLoss()
    optimizer_fn = partial(torch.optim.Adam, lr=1e-3)
    scheduler_fn = partial(torch.optim.lr_scheduler.StepLR, step_size=1)
    acc = MulticlassAccuracy(num_classes=2)

    module = BaseLightningModule(
        model=model,
        criterion=criterion,
        optimizer=optimizer_fn,
        lr_scheduler=scheduler_fn,
        train_metrics=acc,
        val_metrics=acc,
    )
    x = torch.randn(4, 8)
    out = module(x)
    assert out.shape == (4, 2)

    opt_conf = module.configure_optimizers()
    assert isinstance(opt_conf, dict)
    assert "optimizer" in opt_conf
    assert "lr_scheduler" in opt_conf


def test_base_lightning_data_module():
    dataset = DummyDataset(size=100)
    dm = BaseLightningDataModule(
        dataset=dataset,
        data_splitter=RandomSplitter(val_fraction=0.2, seed=42),
        batch_size=16,
    )
    dm.setup("fit")
    assert dm.data_train is not None
    assert dm.data_val is not None
    assert len(dm.data_train) == 80
    assert len(dm.data_val) == 20

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert len(batch) == 2
    assert batch[0].shape == (16, 8)


def test_end_to_end_lightning_training(tmp_path):
    backbone = nn.Linear(8, 4)
    head = nn.Linear(4, 2)
    model = CompositeModel(backbone=backbone, head=head)
    criterion = nn.CrossEntropyLoss()
    optimizer_fn = partial(torch.optim.Adam, lr=1e-3)

    module = BaseLightningModule(
        model=model,
        criterion=criterion,
        optimizer=optimizer_fn,
        train_metrics=MulticlassAccuracy(num_classes=2),
        val_metrics=MulticlassAccuracy(num_classes=2),
        test_metrics=MulticlassAccuracy(num_classes=2),
    )

    train_ds = DummyDataset(size=40)
    val_ds = DummyDataset(size=20)
    test_ds = DummyDataset(size=20)

    dm = BaseLightningDataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=8,
    )

    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(module, datamodule=dm)
    test_res = trainer.test(module, datamodule=dm)
    assert len(test_res) > 0


def test_hydra_instantiation_with_forge_and_runner(tmp_path):
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "module": {
                "_target_": "hlforge.BaseLightningModule",
                "model": {
                    "_target_": "hlforge.CompositeModel",
                    "backbone": {
                        "_target_": "torch.nn.Linear",
                        "in_features": 8,
                        "out_features": 2,
                    },
                },
                "criterion": {
                    "_target_": "torch.nn.CrossEntropyLoss",
                },
                "optimizer": {
                    "_target_": "torch.optim.Adam",
                    "_partial_": True,
                    "lr": 0.001,
                },
            },
            "datamodule": {
                "_target_": "hlforge.BaseLightningDataModule",
                "dataset": {
                    "_target_": "torch.utils.data.TensorDataset",
                    "_args_": [
                        torch.randn(60, 8),
                        torch.randint(0, 2, (60,)),
                    ],
                },
                "data_splitter": {
                    "_target_": "hlforge.RandomSplitter",
                    "val_fraction": 0.2,
                    "seed": 42,
                },
                "batch_size": 10,
            },
            "trainer": {
                "_target_": "pytorch_lightning.Trainer",
                "default_root_dir": str(tmp_path),
                "max_epochs": 1,
                "fast_dev_run": True,
                "logger": False,
                "enable_checkpointing": False,
            },
        },
        flags={"allow_objects": True},
    )

    experiment = ExperimentForge.assemble(cfg)
    runner = ExperimentRunner(experiment)
    results = runner.run(train=True, test=False)
    assert "train_metrics" in results


def test_base_lightning_module_log_config():
    log_config = {
        "loss": {"on_step": True, "on_epoch": False, "prog_bar": False, "sync_dist": False},
        "metrics": {"on_step": True, "on_epoch": False, "prog_bar": False, "sync_dist": False},
    }

    module = BaseLightningModule(log_config=log_config)
    assert module.log_config == log_config


def test_base_lightning_module_save_hyperparameters_logger():
    log_config = {"save_hyperparameters_logger": True}
    module = BaseLightningModule(log_config=log_config)
    assert module.log_config["save_hyperparameters_logger"] is True
