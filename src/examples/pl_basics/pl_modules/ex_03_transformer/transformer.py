# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer language model example based on PyTorch Lightning's basics example.

Based on official PyTorch Lightning example:
https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/transformer.py
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.demos import Transformer, WikiText2
from torch.utils.data import DataLoader, random_split


class LanguageModel(pl.LightningModule):
    """Sequence-to-sequence Transformer Language Model.

    Args:
        vocab_size: Size of the vocabulary. If None, initialized dynamically during setup.
    """

    def __init__(self, vocab_size: int | None = None) -> None:
        super().__init__()
        if vocab_size is None:
            self.model = None
        else:
            self.model = Transformer(vocab_size=vocab_size)

    def setup(self, stage: str) -> None:
        if self.model is None:
            # Dynamically instantiate model using vocab_size from datamodule
            self.model = Transformer(vocab_size=self.trainer.datamodule.vocab_size)

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class MyDataModule(pl.LightningDataModule):
    """WikiText2 LightningDataModule.

    Args:
        batch_size: Batch size for dataloaders.
        num_workers: Number of worker processes for dataloaders.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        self.dataset = WikiText2()

    def setup(self, stage: str | None = None):
        n = len(self.dataset)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [n - 4000, 2000, 2000])
        # to inject in module
        self.vocab_size = self.dataset.vocab_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
