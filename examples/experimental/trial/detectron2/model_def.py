from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
from torch import nn

import determined as det
from determined.pytorch import DataLoader, PyTorchTrial, reset_parameters
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.config import get_cfg
import torch
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
from fvcore.common.checkpoint import Checkpointer

from determined.pytorch import LRScheduler
# from detectron2.utils.registry import Registry

from detectron2_files.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    AspectRatioGroupedDatasetBatchSamp,
)
# from detectron2_files.rcnn import GeneralizedRCNN

# from detectron2.data import (
#     build_detection_test_loader,
#     build_detection_train_loader,
#     AspectRatioGroupedDatasetBatchSamp,
# )
# from detectron2.modeling import GeneralizedRCNN
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.solver import build_lr_scheduler
from fvcore.common.config import CfgNode as _CfgNode
import detectron2
import detectron2.utils.comm as comm

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

# META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
# META_ARCH_REGISTRY.__doc__ = """
# Registry for meta-architectures, i.e. the whole model.
# The registered object will be called with `obj(cfg)`
# and expected to return a `nn.Module` object.
# """
import os
from os import listdir
from os.path import isfile, join

class _IncompatibleKeys(
    NamedTuple(
        # pyre-fixme[10]: Name `IncompatibleKeys` is used but not defined.
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple]),
        ],
    )
):
    pass

class PadSequence:
    def __call__(self, batch):
        # print ('pad sequence: batch: ', batch)
        # print ('solo batch: ', batch[0])
        # t = t
        return batch

class DetectronTrial(PyTorchTrial):
    def __init__(self, context: det.TrialContext) -> None:
        self.context = context
        # print ('VERSION: ', detectron2.__version__)

        # Create a unique download directory for each rank so they don't overwrite each other.
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        self.data_downloaded = False
        self.cfg = get_cfg()

        self.cfg.merge_from_file(self.context.get_hparam('model_yaml'))
        # print ('cfg: ', self.cfg)

        print ('batch_size: ', self.cfg.SOLVER.IMS_PER_BATCH)
        self.cfg.SOLVER.IMS_PER_BATCH = self.context.get_per_slot_batch_size()
        print ('batch_size: ', self.cfg.SOLVER.IMS_PER_BATCH)

    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        step_mode = LRScheduler.StepMode.STEP_EVERY_BATCH
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        # print ('scheduler: ', scheduler)

        return LRScheduler(scheduler, step_mode=step_mode)

    # def build_training_data_loader(self) -> DataLoader:
    #     print ('in train dataloader')
    #     dataset = build_detection_train_loader(self.cfg, batch_size=self.context.get_per_slot_batch_size())
    #     return DataLoader(dataset, batch_sampler=AspectRatioGroupedDatasetBatchSamp(dataset, self.context.get_per_slot_batch_size()), collate_fn=PadSequence())

    def build_training_data_loader(self) -> DataLoader:
        # print ('in train dataloader')
        dataset = build_detection_train_loader(self.cfg, batch_size=self.context.get_per_slot_batch_size())
        # print ('number of samples: ', len(dataset))
        return DataLoader(dataset, batch_sampler=AspectRatioGroupedDatasetBatchSamp(dataset, self.context.get_per_slot_batch_size()), collate_fn=PadSequence())

    def build_validation_data_loader(self) -> DataLoader:
        # print ('in validation dataloader')
        dataset = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0],batch_size=self.context.get_per_slot_batch_size())
        return DataLoader(dataset, batch_sampler=AspectRatioGroupedDatasetBatchSamp(dataset, self.context.get_per_slot_batch_size()), collate_fn=PadSequence())

    def build_model(self) -> nn.Module:
        model = build_model(self.cfg)
        fi = '/mnt/dtrain-fsx/detectron2/R-50.pkl'
        ch = DetectionCheckpointer(model)
        ch.resume_or_load(path = fi, resume=False)
        return ch.model

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:  # type: ignore
        optim = build_optimizer(self.cfg, model)
        return optim

    def train_batch(self, batch: TorchData, model: nn.Module, epoch_idx: int, batch_idx: int):
        loss_dict = model(batch)
        losses = sum(loss_dict.values())
        # assert torch.isfinite(losses).all(), loss_dict
        # print (batch_idx,losses, loss_dict)
        loss_dict['loss'] = losses

        return loss_dict
        # print ('loss_dict: ', loss_dict)

        # loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # return_dict = loss_dict_reduced
        # return_dict['loss'] = losses
        # return_dict['total_loss'] = losses_reduced

        # print ('return dict: ', return_dict)
        # t = t
        # return return_dict

    def evaluate_batch(self, batch: TorchData, model: nn.Module) -> Dict[str, Any]:
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
        return {"validation_loss": 1}
