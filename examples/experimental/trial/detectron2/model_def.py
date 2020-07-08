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
)
from detectron2.solver import build_lr_scheduler

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.
The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""
import os
from os import listdir
from os.path import isfile, join

class DetectronTrial(PyTorchTrial):
    def __init__(self, context: det.TrialContext) -> None:
        self.context = context
        for f in listdir('/mnt/dtrain-fsx'):
            print ('files: ', f)

        # Create a unique download directory for each rank so they don't overwrite each other.
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        self.data_downloaded = False
        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.context.get_hparam('model_yaml'))

    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        step_mode = LRScheduler.StepMode.STEP_EVERY_BATCH
        scheduler = build_lr_scheduler(self.cfg, optimizer)

        return LRScheduler(scheduler, step_mode=step_mode)

    def build_training_data_loader(self) -> DataLoader:
        
        return build_detection_train_loader(self.cfg, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> DataLoader:

        return build_detection_test_loader(self.cfg, batch_size=self.context.get_per_slot_batch_size())

    def build_model(self) -> nn.Module:

        model = build_model(self.cfg)

        return model

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:  # type: ignore
        return build_optimizer(self.cfg, model)

    def train_batch(self, batch: TorchData, model: nn.Module, epoch_idx: int, batch_idx: int):
            
        loss_dict = model(batch)
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict

        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        return {"loss": losses}

    def evaluate_batch(self, batch: TorchData, model: nn.Module) -> Dict[str, Any]:

        return {"validation_loss": 1}
