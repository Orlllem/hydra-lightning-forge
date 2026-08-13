# Tutorial: Config-Driven Image Reconstruction Autoencoder (`ex_02_autoencoder`)

This tutorial demonstrates how to build an Image Reconstruction Autoencoder pipeline with customized loss handling and visualization callbacks using **Hydra-Lightning Forge (`hlforge`)**.

---

## 1. Upstream & Raw PyTorch Lightning Baseline

This example is derived from the official PyTorch Lightning basics example:
* **Upstream Source**: [`autoencoder.py`](https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/autoencoder.py)
* **Raw PyTorch Lightning Version**: [`src/examples/pl_basics/pl_modules/ex_02_autoencoder`](../../pl_modules/ex_02_autoencoder)

### Differences in the Raw Approach
In standard PyTorch Lightning:
1. `LitAutoEncoder` wraps both `Encoder` and `Decoder`, hardcoding `forward`, MSE loss calculation (`F.mse_loss`), optimizer creation, and steps (`training_step`, `validation_step`, `test_step`).
2. Custom datamodule (`MyDataModule`) and callback (`ImageSampler`) are manually instantiated and tied together in python code.

---

## 2. The `hlforge` Abstraction Approach

In this `hl_forge_modules` example, components are modularized and injected dynamically via Hydra configuration:

```
ex_02_autoencoder/
├── autoencoder.py  # Encoder, Decoder, MseLossWrapper, AutoEncoderModule, ImageSampler
├── config.yaml     # Declarative configuration file
└── README.md       # Tutorial documentation
```

### Component Decoupling
1. **Encoder & Decoder Architecture ([`autoencoder.py`](autoencoder.py))**:
   - `Encoder`: Maps 784-dim flattened MNIST images to a 3-dim latent space representation (`in_features=784`, `hidden_dim=64`, `latent_dim=3`).
   - `Decoder`: Reconstructs 784-dim image vectors from the 3-dim latent space representation.
   - Combined via `hlforge.CompositeModel(backbone=Encoder, head=Decoder)`.

2. **Custom Loss Wrapper (`MseLossWrapper`)**:
   - Because autoencoders compare generated image outputs (`y_hat`) directly with flattened input images (rather than target labels `y`), `MseLossWrapper` adapts the generic `(criterion, y_hat, y, batch)` interface required by `BaseLightningModule`.

3. **Lightning Module (`AutoEncoderModule`)**:
   - Subclasses `BaseLightningModule`, inheriting standard logging, step dispatching, and optimizer setup without boilerplate code.

4. **Visualization Callback (`ImageSampler`)**:
   - PyTorch Lightning callback that reconstructs sample images at the end of training epochs and saves `grid_ori_0.png` and `grid_generated_X.png` grids to disk.

---

## 3. Configuration Walkthrough (`config.yaml`)

```yaml
seed: 42

module:
  _target_: examples.pl_basics.hl_forge_modules.ex_02_autoencoder.autoencoder.AutoEncoderModule
  model:
    _target_: hlforge.core.module.CompositeModel
    backbone:
      _target_: examples.pl_basics.hl_forge_modules.ex_02_autoencoder.autoencoder.Encoder
      in_features: 784
      hidden_dim: 64
      latent_dim: 3
    head:
      _target_: examples.pl_basics.hl_forge_modules.ex_02_autoencoder.autoencoder.Decoder
      latent_dim: 3
      hidden_dim: 64
      out_features: 784
  criterion:
    _target_: torch.nn.MSELoss
  loss_wrapper:
    _target_: examples.pl_basics.hl_forge_modules.ex_02_autoencoder.autoencoder.MseLossWrapper
  optimizer_config:
    - optimizer:
        _target_: torch.optim.Adam
        _partial_: true
        lr: 0.001

datamodule:
  _target_: hlforge.core.datamodule.BaseLightningDataModule
  dataset:
    _target_: torchvision.datasets.MNIST
    root: "./data"
    train: true
    download: true
    transform:
      _target_: torchvision.transforms.ToTensor

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1
  callbacks:
    - _target_: examples.pl_basics.hl_forge_modules.ex_02_autoencoder.autoencoder.ImageSampler
      num_samples: 3
      nrow: 8
```

---

## 4. How to Run

Execute training using the `hlforge-train` CLI entry point:

```bash
hlforge-train --config-path "$PWD/src/examples/pl_basics/hl_forge_modules/ex_02_autoencoder" --config-name config
```
