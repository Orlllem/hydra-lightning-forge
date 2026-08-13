import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import nn

from hlforge.core.module import BaseLightningModule


class MseLossWrapper:
    def __call__(self, criterion, y_hat, y, batch):
        image = batch[0].view(batch[0].size(0), -1)
        return criterion(y_hat, image)


class Encoder(nn.Module):
    def __init__(self, in_features: int = 28 * 28, hidden_dim: int = 64, latent_dim: int = 3) -> None:
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
    def __init__(self, latent_dim: int = 3, hidden_dim: int = 64, out_features: int = 28 * 28) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AutoEncoderModule(BaseLightningModule):
    """AutoEncoder implementation using BaseLightningModule."""


class ImageSampler(Callback):
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
        # Simplified dataloader retrieval from trainer
        if trainer.val_dataloaders is not None:
            dataloader = (
                trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
            )
        else:
            return

        batch = next(iter(dataloader))
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        images = images[: self.num_samples]
        images_flattened = images.view(images.size(0), -1)

        with torch.no_grad():
            pl_module.eval()
            # Assuming model forward gives reconstruction
            images_generated = pl_module(images_flattened.to(pl_module.device))
            pl_module.train()

        if trainer.current_epoch == 0:
            torchvision.utils.save_image(self._to_grid(images), f"grid_ori_{trainer.current_epoch}.png")
        torchvision.utils.save_image(
            self._to_grid(images_generated.reshape(images.shape)), f"grid_generated_{trainer.current_epoch}.png"
        )
