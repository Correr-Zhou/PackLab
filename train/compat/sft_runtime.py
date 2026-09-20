"""Runtime compatibility patches for vendored verl SFT trainers."""

from __future__ import annotations

from importlib import import_module


PATCH_ATTR = "_packlab_keep_last_batches_patch"


def patch_sft_trainer_keep_last_batches() -> None:
    """Keep partial train/validation batches in vendored verl SFT trainers."""
    for module_name in ("verl.trainer.sft_trainer", "verl.trainer.sft_trainer_ray"):
        try:
            trainer_module = import_module(module_name)
        except Exception:
            continue
        trainer_cls = getattr(trainer_module, "SFTTrainer", None)
        if trainer_cls is None or getattr(trainer_cls._build_dataloader, PATCH_ATTR, False):
            continue
        trainer_cls._build_dataloader = _build_dataloader_keep_last_factory(trainer_module)
        setattr(trainer_cls._build_dataloader, PATCH_ATTR, True)


def _build_dataloader_keep_last_factory(trainer_module):
    def _build_dataloader_keep_last(self):
        config = self.config
        device_name = trainer_module.get_device_name()

        if hasattr(self, "engine"):
            dp_rank = self.engine.get_data_parallel_rank()
            dp_size = self.engine.get_data_parallel_size()
            num_workers = self.config.data.num_workers
        else:
            dp_rank = 0
            dp_size = 1
            num_workers = self.config.data.get("num_workers", 8)

        self.train_sampler = trainer_module.DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=dp_size,
            rank=dp_rank,
            drop_last=False,
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = trainer_module.SFTTensorCollator(config.data.pad_mode)

        self.train_dataloader = trainer_module.StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = trainer_module.DistributedSampler(
                self.val_dataset,
                shuffle=False,
                num_replicas=dp_size,
                rank=dp_rank,
                drop_last=False,
            )
            self.val_dataloader = trainer_module.StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

        if trainer_module.__name__.endswith("sft_trainer_ray"):
            if self.config.trainer.total_training_steps is not None:
                self.total_training_steps = self.config.trainer.total_training_steps
            else:
                self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
            self.optimizer_config.total_training_steps = self.total_training_steps

            self.steps_per_epoch = len(self.train_dataloader)
            self.save_freq = self.config.trainer.save_freq
            if self.save_freq == "after_each_epoch":
                self.save_freq = self.steps_per_epoch

            self.test_freq = self.config.trainer.test_freq
            if self.test_freq == "after_each_epoch":
                self.test_freq = self.steps_per_epoch

    return _build_dataloader_keep_last
