"""Base PyTorch Lightning DataModule with Hydra-instantiable sub-components."""

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, random_split


class TransformedDataset(Dataset):
    """Wraps a PyTorch Dataset to dynamically apply a transform callable.

    Useful when datasets are instantiated without internal transforms, allowing
    post-instantiation wrapping.
    """

    def __init__(self, dataset: Dataset, transform: Callable[[Any], Any]) -> None:
        """Initialize wrapper with underlying dataset and transform.

        Args:
            dataset: Base PyTorch dataset.
            transform: Callable transformation function or pipeline.
        """
        self.dataset: Dataset = dataset
        self.transform: Callable[[Any], Any] = transform

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        """Fetch a sample at the specified index and apply the transform.

        Supports standard tuples, dicts (transforming keys under 'x'), or single items.

        Args:
            idx: The index of the sample to fetch.

        Returns:
            The transformed sample.
        """
        item = self.dataset[idx]
        if isinstance(item, tuple) and len(item) >= 2:
            x, y, *rest = item
            x_trans = self.transform(x)
            if rest:
                return (x_trans, y, *rest)
            return (x_trans, y)
        if isinstance(item, dict) and "x" in item:
            item_copy = dict(item)
            item_copy["x"] = self.transform(item["x"])
            return item_copy
        return self.transform(item)


class RandomSplitter:
    """Configurable train/validation dataset random splitter."""

    def __init__(self, val_fraction: float = 0.1, seed: int = 42) -> None:
        """Initialize splitting fraction and random seed.

        Args:
            val_fraction: Proportion of the dataset allocated to validation (0.0 to 1.0).
            seed: RNG seed for reproducible splits.
        """
        self.val_fraction: float = val_fraction
        self.seed: int = seed

    def __call__(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Split dataset into train and validation subsets.

        Args:
            dataset: The input dataset to split.

        Returns:
            A tuple containing the training and validation subsets.
        """
        val_len = int(len(dataset) * self.val_fraction)
        train_len = len(dataset) - val_len
        generator = torch.Generator().manual_seed(self.seed)
        subsets = random_split(dataset, [train_len, val_len], generator=generator)
        return subsets[0], subsets[1]


class BaseLightningDataModule(pl.LightningDataModule):
    """Hydra-instantiable Base DataModule for PyTorch Lightning.

    Encapsulates dataset construction, transforms, splitting strategies, samplers,
    custom batch collate functions, and DataLoader configurations.
    """

    def __init__(
        self,
        dataset: Dataset | Callable[[], Dataset] | None = None,
        train_dataset: Dataset | Callable[[], Dataset] | None = None,
        val_dataset: Dataset | Callable[[], Dataset] | None = None,
        test_dataset: Dataset | Callable[[], Dataset] | None = None,
        predict_dataset: Dataset | Callable[[], Dataset] | None = None,
        train_transform: Callable[[Any], Any] | None = None,
        val_transform: Callable[[Any], Any] | None = None,
        test_transform: Callable[[Any], Any] | None = None,
        predict_transform: Callable[[Any], Any] | None = None,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        train_sampler: Sampler | Callable[[Dataset], Sampler] | None = None,
        val_sampler: Sampler | Callable[[Dataset], Sampler] | None = None,
        test_sampler: Sampler | Callable[[Dataset], Sampler] | None = None,
        data_splitter: Callable[[Dataset], tuple[Dataset, Dataset]] | None = None,
        preparer: Callable[[], None] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the DataModule with all configurable data pipeline sub-components.

        Args:
            dataset: Base single dataset to be split using data_splitter.
            train_dataset: Explicit training dataset or dataset factory.
            val_dataset: Explicit validation dataset or dataset factory.
            test_dataset: Explicit test dataset or dataset factory.
            predict_dataset: Explicit prediction dataset or dataset factory.
            train_transform: Transform/augmentation pipeline applied to training data.
            val_transform: Transform pipeline applied to validation data.
            test_transform: Transform pipeline applied to test data.
            predict_transform: Transform pipeline applied to prediction data.
            collate_fn: Custom batch collation callable.
            train_sampler: Sampler or factory for training dataloader.
            val_sampler: Sampler or factory for validation dataloader.
            test_sampler: Sampler or factory for test dataloader.
            data_splitter: Splitting callable used when only a single base dataset is provided.
            preparer: Preparation/download callable invoked in prepare_data().
            *args: Additional positional arguments for pl.LightningDataModule.
            **kwargs: Additional keyword arguments for pl.LightningDataModule. These arguments will be stored in `hparams`. Supported keys:
                batch_size: Default training batch size.
                val_batch_size: Optional validation batch size override (defaults to batch_size).
                test_batch_size: Optional test batch size override (defaults to batch_size).
                num_workers: Number of DataLoader subprocesses.
                pin_memory: Whether DataLoader pins memory to CUDA page-locked memory.
                drop_last: Whether to drop the last incomplete batch in training.
                shuffle: Whether to shuffle training data (ignored if sampler is provided).
                persistent_workers: Keeps DataLoader worker processes alive between epochs.
                prefetch_factor: Number of batches loaded in advance by each worker.
                dataloader_kwargs: Extra keyword arguments forwarded to DataLoader constructors.

        Returns:
            None.
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "dataset",
                "train_dataset",
                "val_dataset",
                "test_dataset",
                "predict_dataset",
                "train_transform",
                "val_transform",
                "test_transform",
                "predict_transform",
                "collate_fn",
                "train_sampler",
                "val_sampler",
                "test_sampler",
                "data_splitter",
                "preparer",
            ],
        )

        self._raw_dataset: Dataset | Callable[[], Dataset] | None = dataset
        self._raw_train_dataset: Dataset | Callable[[], Dataset] | None = train_dataset
        self._raw_val_dataset: Dataset | Callable[[], Dataset] | None = val_dataset
        self._raw_test_dataset: Dataset | Callable[[], Dataset] | None = test_dataset
        self._raw_predict_dataset: Dataset | Callable[[], Dataset] | None = predict_dataset

        self.train_transform: Callable[[Any], Any] | None = train_transform
        self.val_transform: Callable[[Any], Any] | None = val_transform
        self.test_transform: Callable[[Any], Any] | None = test_transform
        self.predict_transform: Callable[[Any], Any] | None = predict_transform

        self.collate_fn: Callable[[list[Any]], Any] | None = collate_fn
        self.train_sampler_factory: Sampler | Callable[[Dataset], Sampler] | None = train_sampler
        self.val_sampler_factory: Sampler | Callable[[Dataset], Sampler] | None = val_sampler
        self.test_sampler_factory: Sampler | Callable[[Dataset], Sampler] | None = test_sampler
        self.data_splitter: Callable[[Dataset], tuple[Dataset, Dataset]] | None = data_splitter
        self.preparer: Callable[[], None] | None = preparer

        # Resolved datasets populated in setup()
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.data_predict: Dataset | None = None

    def prepare_data(self) -> None:
        """Execute one-time dataset preparation/download logic.

        Returns:
            None.
        """
        if self.preparer is not None:
            self.preparer()

    @staticmethod
    def _resolve_dataset(dataset: Dataset | Callable[[], Dataset] | None) -> Dataset | None:
        """Instantiate callable dataset factories if needed.

        Args:
            dataset: The dataset or dataset factory to resolve.

        Returns:
            The resolved Dataset instance, or None if the input is None.
        """
        if dataset is None:
            return None
        if callable(dataset) and not isinstance(dataset, Dataset):
            return dataset()
        return dataset

    @staticmethod
    def _apply_transform(dataset: Dataset | None, transform: Callable[[Any], Any] | None) -> Dataset | None:
        """Wrap dataset in TransformedDataset if transform is provided and dataset does not embed it.

        Args:
            dataset: The input dataset.
            transform: The transformation to apply.

        Returns:
            The dataset with the transform applied, or None if the input dataset is None.
        """
        if dataset is None or transform is None:
            return dataset
        if hasattr(dataset, "transform") and getattr(dataset, "transform", None) is None:
            dataset.transform = transform
            return dataset
        return TransformedDataset(dataset=dataset, transform=transform)

    def setup(self, stage: str | None = None) -> None:
        """Build and assign datasets for fit, test, and predict stages.

        Args:
            stage: The stage to set up (fit, test, predict, or None).

        Returns:
            None.
        """
        if stage in ("fit", None):
            if self._raw_train_dataset is not None:
                train_ds = self._resolve_dataset(dataset=self._raw_train_dataset)
                val_ds = self._resolve_dataset(dataset=self._raw_val_dataset)
            elif self._raw_dataset is not None:
                base_ds = self._resolve_dataset(dataset=self._raw_dataset)
                if base_ds is not None and self.data_splitter is not None:
                    train_ds, val_ds = self.data_splitter(base_ds)
                else:
                    train_ds, val_ds = base_ds, base_ds
            else:
                train_ds, val_ds = None, None

            self.data_train = self._apply_transform(dataset=train_ds, transform=self.train_transform)
            self.data_val = self._apply_transform(dataset=val_ds, transform=self.val_transform)

        if stage in ("test", None):
            test_ds = self._resolve_dataset(dataset=self._raw_test_dataset)
            self.data_test = self._apply_transform(dataset=test_ds, transform=self.test_transform)

        if stage in ("predict", None):
            predict_ds = self._resolve_dataset(dataset=self._raw_predict_dataset)
            self.data_predict = self._apply_transform(dataset=predict_ds, transform=self.predict_transform)

    def _build_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        sampler_factory: Sampler | Callable[[Dataset], Sampler] | None,
        shuffle: bool,
        drop_last: bool = False,
    ) -> DataLoader:
        """Construct DataLoader with configured options.

        Args:
            dataset: The dataset to use.
            batch_size: The batch size.
            sampler_factory: Sampler or factory for the dataloader.
            shuffle: Whether to shuffle the data.
            drop_last: Whether to drop the last incomplete batch.

        Returns:
            The constructed DataLoader.
        """
        sampler: Sampler | None = None
        if sampler_factory is not None:
            if callable(sampler_factory) and not isinstance(sampler_factory, Sampler):
                sampler = sampler_factory(dataset)
            else:
                sampler = sampler_factory

        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "num_workers": self.hparams.get("num_workers", 0),
            "pin_memory": self.hparams.get("pin_memory", False),
            "collate_fn": self.collate_fn,
            "drop_last": drop_last,
            **self.hparams.get("dataloader_kwargs", {}),
        }

        if sampler is not None:
            kwargs["sampler"] = sampler
            kwargs["shuffle"] = False
        else:
            kwargs["shuffle"] = shuffle

        return DataLoader(**kwargs)

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader.

        Returns:
            The training DataLoader.
        """
        if self.data_train is None:
            raise RuntimeError("Train dataset is not initialized. Ensure setup('fit') was called.")
        return self._build_dataloader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            sampler_factory=self.train_sampler_factory,
            shuffle=self.hparams.get("shuffle", True),
            drop_last=self.hparams.get("drop_last", False),
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader.

        Returns:
            The validation DataLoader.
        """
        if self.data_val is None:
            raise RuntimeError("Validation dataset is not initialized. Ensure setup('fit') was called.")
        batch_size = self.hparams.get("val_batch_size", self.hparams.batch_size)
        return self._build_dataloader(
            dataset=self.data_val,
            batch_size=batch_size,
            sampler_factory=self.val_sampler_factory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test DataLoader.

        Returns:
            The test DataLoader.
        """
        if self.data_test is None:
            raise RuntimeError("Test dataset is not initialized. Ensure setup('test') was called.")
        batch_size = self.hparams.get("test_batch_size", self.hparams.batch_size)
        return self._build_dataloader(
            dataset=self.data_test,
            batch_size=batch_size,
            sampler_factory=self.test_sampler_factory,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create the prediction DataLoader.

        Returns:
            The prediction DataLoader.
        """
        if self.data_predict is None:
            raise RuntimeError("Predict dataset is not initialized. Ensure setup('predict') was called.")
        batch_size = self.hparams.test_batch_size or self.hparams.batch_size
        return self._build_dataloader(
            dataset=self.data_predict,
            batch_size=batch_size,
            sampler_factory=None,
            shuffle=False,
            drop_last=False,
        )
