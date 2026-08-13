# Examples

This directory contains machine learning pipeline examples using PyTorch Lightning and Hydra-Lightning Forge (`hlforge`).

## Directory Structure

- `pl_modules/`: Raw PyTorch Lightning implementations, demonstrating standard boilerplate for models and data modules.
- `hl_forge_modules/`: Examples utilizing `hlforge` core abstractions (`hlforge.BaseLightningModule`, `hlforge.BaseLightningDataModule`) for configuration-driven training.

---

## 1. Raw PyTorch Lightning Examples (`pl_modules/`)

These examples show how to structure standard PyTorch Lightning components without Hydra-Lightning Forge wrappers.
They are adapted directly from official [PyTorch Lightning Basics Examples](https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples/pytorch/basics).

| Local Example Directory | PyTorch Lightning Source File | Focus / Architecture |
| :--- | :--- | :--- |
| [`ex_01_backbone_image_classifier`](pl_modules/ex_01_backbone_image_classifier) | [`backbone_image_classifier.py`](https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/backbone_image_classifier.py) | Custom MNIST backbone feature extractor & classifier |
| [`ex_02_autoencoder`](pl_modules/ex_02_autoencoder) | [`autoencoder.py`](https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/autoencoder.py) | Multi-module Encoder-Decoder architecture & custom image sampler callback |
| [`ex_03_transformer`](pl_modules/ex_03_transformer) | [`transformer.py`](https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/transformer.py) | Sequence modeling with Transformer & WikiText2 text preprocessing |

### Run Command
To run these raw PyTorch Lightning examples, use `hlforge-train` pointed to the respective example directory:
```bash
hlforge-train --config-path "$PWD/src/examples/pl_basics/pl_modules/<example_name>" --config-name config
```

---

## 2. Hydra-Lightning Forge Examples (`hl_forge_modules/`)

These examples demonstrate how to build modular, configuration-driven pipelines using `hlforge` core abstractions (`hlforge.BaseLightningModule`, `hlforge.BaseLightningDataModule`, `hlforge.CompositeModel`). Each folder includes a detailed tutorial `README.md` showing how it relates to the official PyTorch Lightning basics example.

| Local Example Directory | Tutorial | Upstream PyTorch Lightning Reference | Focus / Architecture |
| :--- | :--- | :--- | :--- |
| [`ex_01_mnist`](hl_forge_modules/ex_01_mnist) | [`ex_01_mnist/README.md`](hl_forge_modules/ex_01_mnist/README.md) | [`backbone_image_classifier.py`](https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/backbone_image_classifier.py) | Config-driven MNIST classification with `CompositeModel` |
| [`ex_02_autoencoder`](hl_forge_modules/ex_02_autoencoder) | [`ex_02_autoencoder/README.md`](hl_forge_modules/ex_02_autoencoder/README.md) | [`autoencoder.py`](https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/autoencoder.py) | Config-driven MNIST autoencoder with `MseLossWrapper` & `ImageSampler` |
| [`ex_03_transformer`](hl_forge_modules/ex_03_transformer) | [`ex_03_transformer/README.md`](hl_forge_modules/ex_03_transformer/README.md) | [`transformer.py`](https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/transformer.py) | Config-driven sequence model with dynamic vocab setup & `WikiText2` |

### Run Command
```bash
hlforge-train --config-path "$PWD/src/examples/pl_basics/hl_forge_modules/<example_name>" --config-name config
```
