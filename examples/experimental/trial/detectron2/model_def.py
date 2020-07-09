from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
from torch import nn

import determined as det
from determined.pytorch import DataLoader, PyTorchTrial, reset_parameters
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
from detectron2.config import get_cfg
import torch

from determined.pytorch import LRScheduler
from detectron2.utils.registry import Registry

from detectron2_files.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    AspectRatioGroupedDatasetBatchSamp,
)
from detectron2.solver import build_lr_scheduler
from fvcore.common.config import CfgNode as _CfgNode
from detectron2.utils.events import EventStorage

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.
The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""
import os
from os import listdir
from os.path import isfile, join

class PadSequence:
    def __call__(self, batch):

        return batch

class DetectronTrial(PyTorchTrial):
    def __init__(self, context: det.TrialContext) -> None:
        self.context = context

        # Create a unique download directory for each rank so they don't overwrite each other.
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        self.data_downloaded = False
        self.cfg = get_cfg()
        # print ('aspect: ', self.cfg.DATALOADER.ASPECT_RATIO_GROUPING)
        print ('cfg: ', self.cfg)
        # loaded_cfg = _CfgNode.load_yaml_with_base(self.context.get_hparam('model_yaml'), allow_unsafe=True)
        # print ('loaded cfg: ', loaded_cfg)
        # self.cfg.merge_from_other_cfg(loaded_cfg)

        self.cfg.merge_from_file(self.context.get_hparam('model_yaml'))

    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        step_mode = LRScheduler.StepMode.STEP_EVERY_BATCH
        scheduler = build_lr_scheduler(self.cfg, optimizer)

        return LRScheduler(scheduler, step_mode=step_mode)

    def build_training_data_loader(self) -> DataLoader:
        dataset = build_detection_train_loader(self.cfg, batch_size=self.context.get_per_slot_batch_size())
        return DataLoader(dataset, batch_sampler=AspectRatioGroupedDatasetBatchSamp(dataset, self.context.get_per_slot_batch_size()), collate_fn=PadSequence())

    def build_validation_data_loader(self) -> DataLoader:
        dataset = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0],batch_size=self.context.get_per_slot_batch_size())
        return DataLoader(dataset, batch_sampler=AspectRatioGroupedDatasetBatchSamp(dataset, self.context.get_per_slot_batch_size()), collate_fn=PadSequence())

    def build_model(self) -> nn.Module:
        model = build_model(self.cfg)
        return model

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:  # type: ignore
        return build_optimizer(self.cfg, model)

    def train_batch(self, batch: TorchData, model: nn.Module, epoch_idx: int, batch_idx: int):
        print ('in train batch: ', len(batch), batch[0])
        with EventStorage(1) as storage:
            loss_dict = model(batch)
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict

        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        return {"loss": losses}

    def evaluate_batch(self, batch: TorchData, model: nn.Module) -> Dict[str, Any]:
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
        return {"validation_loss": 1}
