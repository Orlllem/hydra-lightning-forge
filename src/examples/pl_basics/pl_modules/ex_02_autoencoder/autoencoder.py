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
"""Autoencoder example based on PyTorch Lightning's basics example.

Based on official PyTorch Lightning example:
https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/autoencoder.py
"""

from typing import Any

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class Encoder(nn.Module):
    """Encoder module mapping flattened input images to a low-dimensional latent space.

    Args:
        in_features: Number of input features (defaults to 28 * 28 = 784 for MNIST).
        hidden_dim: Dimension of the intermediate hidden layer.
        latent_dim: Dimension of the latent space representation.
    """

    def __init__(
        self,
        in_features: int = 28 * 28,
        hidden_dim: int = 64,
        latent_dim: int = 3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


class Decoder(nn.Module):
    """Decoder module reconstructing flattened images from a low-dimensional latent space.

    Args:
        latent_dim: Dimension of the latent space representation.
        hidden_dim: Dimension of the intermediate hidden layer.
        out_features: Number of reconstructed output features (defaults to 28 * 28 = 784 for MNIST).
    """

    def __init__(
        self,
        latent_dim: int = 3,
        hidden_dim: int = 64,
        out_features: int = 28 * 28,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LitAutoEncoder(pl.LightningModule):
    """MNIST AutoEncoder LightningModule.

    Args:
        encoder: Optional custom encoder module. If None, default Encoder() is used.
        decoder: Optional custom decoder module. If None, default Decoder() is used.
        hidden_dim: Intermediate hidden dimension used when constructing default encoder/decoder.
        latent_dim: Latent space dimension used when constructing default encoder/decoder.
        learning_rate: Learning rate for Adam optimizer.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        hidden_dim: int = 64,
        latent_dim: int = 3,
        learning_rate: float = 0.001,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        if encoder is None:
            encoder = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        if decoder is None:
            decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def _prepare_batch(self, batch: Any) -> torch.Tensor:
        x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
        return x.view(x.size(0), -1)

    def _common_step(self, batch: Any, batch_idx: int, stage: str) -> torch.Tensor:
        x = self._prepare_batch(batch)
        loss = F.mse_loss(self(x), x)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "valid")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int | None = None) -> torch.Tensor:
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class MyDataModule(pl.LightningDataModule):
    """MNIST LightningDataModule.

    Args:
        data_dir: Path to directory where MNIST dataset will be stored.
        batch_size: Batch size for dataloaders.
        num_workers: Number of worker processes for dataloaders.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None
        self.mnist_predict = None

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            dataset = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class ImageSampler(Callback):
    """Callback for visual sample generation during training epochs.

    Reconstructs a fixed number of validation sample images and saves grid visualizations.

    Args:
        num_samples: Number of images displayed in the grid.
        nrow: Number of images displayed in each row of the grid.
        padding: Amount of padding.
        normalize: If True, shift the image to the range (0, 1).
        value_range: Optional tuple (min, max) for normalizing the image.
        scale_each: If True, scale each image in the batch separately.
        pad_value: Value for the padded pixels.
    """

    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        value_range: tuple[int, int] | None = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.value_range = value_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def _to_grid(self, images: torch.Tensor) -> torch.Tensor:
        return torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            value_range=self.value_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        dataloader = None
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            if getattr(trainer.datamodule, "mnist_val", None) is not None:
                dataloader = DataLoader(trainer.datamodule.mnist_val, batch_size=self.num_samples)
            elif hasattr(trainer.datamodule, "val_dataloader"):
                dataloader = trainer.datamodule.val_dataloader()
        elif trainer.val_dataloaders is not None:
            dataloader = trainer.val_dataloaders
            if isinstance(dataloader, list):
                dataloader = dataloader[0]

        if dataloader is None:
            return

        batch = next(iter(dataloader))
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images[: self.num_samples]
        images_flattened = images.view(images.size(0), -1)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images_generated = pl_module(images_flattened.to(pl_module.device))
            pl_module.train()

        if trainer.current_epoch == 0:
            torchvision.utils.save_image(self._to_grid(images), f"grid_ori_{trainer.current_epoch}.png")
        torchvision.utils.save_image(
            self._to_grid(images_generated.reshape(images.shape)), f"grid_generated_{trainer.current_epoch}.png"
        )


__all__ = [
    "Decoder",
    "Encoder",
    "ImageSampler",
    "LitAutoEncoder",
    "MyDataModule",
]
