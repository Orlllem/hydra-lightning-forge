from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning.demos import Transformer
from torch.utils.data import Dataset, random_split

from hlforge.core.datamodule import BaseLightningDataModule
from hlforge.core.module import BaseLightningModule


class AutoTransformerModule(BaseLightningModule):
    """Transformer implementation with dynamic model instantiation."""

    def setup(self, stage: str | None = None) -> None:
        if self.model is None:
            # Dynamically instantiate model from vocab_size from datamodule
            vocab_size = self.trainer.datamodule.data_train.dataset.vocab_size
            self.model = Transformer(
                vocab_size=vocab_size, nhead=self.hparams.nhead, nhid=self.hparams.nhid, nlayers=self.hparams.nlayers
            )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("test_loss", loss, prog_bar=True)
        return loss


class RandomSplitter:
    """Configurable train/validation dataset random splitter."""

    def __init__(self, val_size: int = 2000, test_size: int = 2000) -> None:
        """Initialize splitting.

        Args:
            val_size: Validation size.
            test_size: Test size.

        Returns:
            None
        """
        self.val_size = val_size
        self.test_size = test_size

    def __call__(self, dataset: Dataset) -> tuple[Dataset, Dataset, Dataset]:
        """Split dataset into train and validation subsets.

        Args:
            dataset: The input dataset to split.

        Returns:
            tuple[Dataset, Dataset, Dataset]: The training, validation, and test subsets.
        """
        n = len(dataset)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n - (self.val_size + self.test_size), self.val_size, self.test_size]
        )
        return train_dataset, val_dataset, test_dataset


class AutoTransformerDataModule(BaseLightningDataModule):
    def setup(self, stage: str | None = None) -> None:
        """Build and assign datasets for fit, test, and predict stages."""

        base_ds = self._resolve_dataset(self._raw_dataset)
        # Note: data_splitter now returns 3 items
        train_ds, val_ds, test_ds = self.data_splitter(base_ds)

        if stage in ("fit", None):
            self.data_train = self._apply_transform(train_ds, self.train_transform)
            self.data_val = self._apply_transform(val_ds, self.val_transform)

        if stage in ("test", None):
            self.data_test = self._apply_transform(test_ds, self.test_transform)
