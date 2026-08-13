# Tutorial: Config-Driven MNIST Image Classification (`ex_01_mnist`)

This tutorial explains how to convert a standard PyTorch Lightning image classifier into a clean, configuration-driven pipeline using **Hydra-Lightning Forge (`hlforge`)**.

---

## 1. Upstream & Raw PyTorch Lightning Baseline

This example is directly derived from the official PyTorch Lightning basics example:
* **Upstream Source**: [`backbone_image_classifier.py`](https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/backbone_image_classifier.py)
* **Raw PyTorch Lightning Version**: [`src/examples/pl_basics/pl_modules/ex_01_backbone_image_classifier`](../../pl_modules/ex_01_backbone_image_classifier)

### Differences in the Raw Approach
In standard PyTorch Lightning:
1. `LitClassifier` subclasses `pl.LightningModule` and hardcodes training step logic, optimizer creation (`Adam`), loss calculation (`F.cross_entropy`), and target evaluation.
2. `MyDataModule` explicitly implements `prepare_data`, `setup`, and dataloader creation routines specifically for `MNIST`.

---

## 2. The `hlforge` Abstraction Approach

In this `hl_forge_modules` example, code boilerplate is removed in favor of modular components declared in [`config.yaml`](config.yaml):

```
ex_01_mnist/
├── backbone_image_classifier.py  # Minimal PyTorch nn.Module definitions
├── config.yaml                   # Complete pipeline declaration
└── README.md                     # Tutorial documentation
```

### Component Decoupling
1. **Model Architecture ([`backbone_image_classifier.py`](backbone_image_classifier.py))**:
   - `Backbone`: Pure `nn.Module` feature extractor (flattens input and applies linear transformation + ReLU).
   - `ClassificationHead`: Pure `nn.Module` linear classifier.
   - Combined seamlessly using `hlforge.CompositeModel`.

2. **Lightning Module**:
   - Uses `hlforge.BaseLightningModule` directly. No custom `LightningModule` class definition is required!
   - Loss function (`torch.nn.CrossEntropyLoss`), optimizer (`torch.optim.Adam`), LR scheduler (`StepLR`), and evaluation metrics (`torchmetrics.classification.MulticlassAccuracy`) are injected via Hydra.

3. **Data Pipeline**:
   - Uses `hlforge.BaseLightningDataModule` directly.
   - Configured with `torchvision.datasets.MNIST`, `ToTensor` transforms, and `hlforge.RandomSplitter`.

---

## 3. Configuration Walkthrough (`config.yaml`)

```yaml
seed: 42

module:
  _target_: hlforge.BaseLightningModule
  model:
    _target_: hlforge.CompositeModel
    backbone:
      _target_: examples.pl_basics.hl_forge_modules.ex_01_mnist.backbone_image_classifier.Backbone
      in_features: 784
      hidden_dim: 128
    head:
      _target_: examples.pl_basics.hl_forge_modules.ex_01_mnist.backbone_image_classifier.ClassificationHead
      in_features: 128
      num_classes: 10
  criterion:
    _target_: torch.nn.CrossEntropyLoss
  optimizer_config:
    - optimizer:
        _target_: torch.optim.Adam
        _partial_: true
        lr: 0.0001
      lr_scheduler:
        _target_: hlforge.core.configs.LRSchedulerConfig
        scheduler:
          _target_: torch.optim.lr_scheduler.StepLR
          _partial_: true
          step_size: 5
          gamma: 0.5
  train_metrics:
    acc:
      _target_: torchmetrics.classification.MulticlassAccuracy
      num_classes: 10

datamodule:
  _target_: hlforge.BaseLightningDataModule
  dataset:
    _target_: torchvision.datasets.MNIST
    root: "./data"
    train: true
    download: true
  data_splitter:
    _target_: hlforge.RandomSplitter
    val_fraction: 0.08333333333333333
```

---

## 4. How to Run

Execute training using the `hlforge-train` CLI entry point:

```bash
hlforge-train --config-path "$PWD/src/examples/pl_basics/hl_forge_modules/ex_01_mnist" --config-name config
```
