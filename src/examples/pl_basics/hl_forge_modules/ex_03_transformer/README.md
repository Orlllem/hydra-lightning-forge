# Tutorial: Config-Driven Sequence Modeling Transformer (`ex_03_transformer`)

This tutorial demonstrates how to build a Transformer language model pipeline with dynamic vocabulary resolution and dataset splitting using **Hydra-Lightning Forge (`hlforge`)**.

---

## 1. Upstream & Raw PyTorch Lightning Baseline

This example is derived from the official PyTorch Lightning basics example:
* **Upstream Source**: [`transformer.py`](https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/basics/transformer.py)
* **Raw PyTorch Lightning Version**: [`src/examples/pl_basics/pl_modules/ex_03_transformer`](../../pl_modules/ex_03_transformer)

### Differences in the Raw Approach
In standard PyTorch Lightning:
1. `LanguageModel` dynamically inspects `self.trainer.datamodule.vocab_size` inside `setup()` to instantiate `pytorch_lightning.demos.Transformer`.
2. `MyDataModule` manually downloads and splits `WikiText2` into train/val/test dataloaders.

---

## 2. The `hlforge` Abstraction Approach

In this `hl_forge_modules` example, both dataset setup and model instantiation are decoupled using `hlforge` core abstractions:

```
ex_03_transformer/
├── transformer.py  # AutoTransformerModule, RandomSplitter, AutoTransformerDataModule
├── config.yaml     # Declarative configuration file
└── README.md       # Tutorial documentation
```

### Component Decoupling
1. **Dynamic Lightning Module (`AutoTransformerModule`)**:
   - Subclasses `hlforge.BaseLightningModule`.
   - In its `setup()` phase, it dynamically retrieves the dataset's vocabulary size (`self.trainer.datamodule.data_train.dataset.vocab_size`) to construct `pytorch_lightning.demos.Transformer(vocab_size=...)` using hyperparameters (`nhead`, `nhid`, `nlayers`) supplied by Hydra.

2. **Custom Data Module & Splitter**:
   - `AutoTransformerDataModule` subclasses `hlforge.BaseLightningDataModule`.
   - `RandomSplitter` splits `WikiText2` into three subsets (`train`, `val`, `test`) based on configured sizes (`val_size=2000`, `test_size=2000`).

---

## 3. Configuration Walkthrough (`config.yaml`)

```yaml
seed: 42

module:
  _target_: examples.pl_basics.hl_forge_modules.ex_03_transformer.transformer.AutoTransformerModule
  nhead: 2
  nhid: 200
  nlayers: 2
  optimizer_config:
    - optimizer:
        _target_: torch.optim.SGD
        _partial_: true
        lr: 0.1

datamodule:
  _target_: examples.pl_basics.hl_forge_modules.ex_03_transformer.transformer.AutoTransformerDataModule
  dataset:
    _target_: pytorch_lightning.demos.WikiText2
  data_splitter:
    _target_: examples.pl_basics.hl_forge_modules.ex_03_transformer.transformer.RandomSplitter
    val_size: 2000
    test_size: 2000
  batch_size: 32

trainer:
  _target_: pytorch_lightning.Trainer
  max_epochs: 1
  overfit_batches: 10
```

---

## 4. How to Run

Execute training using the `hlforge-train` CLI entry point:

```bash
hlforge-train --config-path "$PWD/src/examples/pl_basics/hl_forge_modules/ex_03_transformer" --config-name config
```
