"""Tests for the Autoencoder example module and configuration."""

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from examples.pl_basics.pl_modules.ex_02_autoencoder.autoencoder import (
    Decoder,
    Encoder,
    ImageSampler,
    LitAutoEncoder,
    MyDataModule,
)
from hlforge.core.forge import ExperimentForge
from hlforge.core.runner import ExperimentRunner


def test_encoder_decoder_forward():
    encoder = Encoder(in_features=784, hidden_dim=64, latent_dim=3)
    decoder = Decoder(latent_dim=3, hidden_dim=64, out_features=784)

    # Test 2D input (B, 784)
    x = torch.randn(4, 784)
    z = encoder(x)
    assert z.shape == (4, 3)
    x_rec = decoder(z)
    assert x_rec.shape == (4, 784)

    # Test 4D image tensor input (B, 1, 28, 28)
    x_img = torch.randn(4, 1, 28, 28)
    z_img = encoder(x_img)
    assert z_img.shape == (4, 3)


def test_lit_autoencoder_module():
    module = LitAutoEncoder(hidden_dim=32, latent_dim=4, learning_rate=1e-3)
    x = torch.randn(4, 1, 28, 28)
    out = module(x)
    assert out.shape == (4, 784)

    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)

    batch = (x, torch.randint(0, 10, (4,)))
    loss = module.training_step(batch, batch_idx=0)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_autoencoder_training_fast_dev_run(tmp_path):
    module = LitAutoEncoder(hidden_dim=32, latent_dim=3)
    dm = MyDataModule(data_dir=str(tmp_path), batch_size=8)

    sampler = ImageSampler(num_samples=2, nrow=2)
    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        callbacks=[sampler],
    )

    trainer.fit(module, datamodule=dm)
    test_res = trainer.test(module, datamodule=dm)
    assert len(test_res) > 0


def test_autoencoder_hydra_config(tmp_path):
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "module": {
                "_target_": "examples.pl_basics.pl_modules.ex_02_autoencoder.autoencoder.LitAutoEncoder",
                "learning_rate": 0.001,
                "encoder": {
                    "_target_": "examples.pl_basics.pl_modules.ex_02_autoencoder.autoencoder.Encoder",
                    "in_features": 784,
                    "hidden_dim": 32,
                    "latent_dim": 3,
                },
                "decoder": {
                    "_target_": "examples.pl_basics.pl_modules.ex_02_autoencoder.autoencoder.Decoder",
                    "latent_dim": 3,
                    "hidden_dim": 32,
                    "out_features": 784,
                },
            },
            "datamodule": {
                "_target_": "examples.pl_basics.pl_modules.ex_02_autoencoder.autoencoder.MyDataModule",
                "data_dir": str(tmp_path),
                "batch_size": 8,
                "num_workers": 0,
            },
            "callbacks": {
                "sampler": {
                    "_target_": "examples.pl_basics.pl_modules.ex_02_autoencoder.autoencoder.ImageSampler",
                    "num_samples": 2,
                    "nrow": 2,
                }
            },
            "trainer": {
                "_target_": "pytorch_lightning.Trainer",
                "default_root_dir": str(tmp_path),
                "max_epochs": 1,
                "fast_dev_run": True,
                "logger": False,
                "enable_checkpointing": False,
            },
        }
    )

    experiment = ExperimentForge.assemble(cfg)
    runner = ExperimentRunner(experiment)
    results = runner.run(train=True, test=True)
    assert "train_metrics" in results
    assert "test_metrics" in results
