from typing import Any, Dict, Sequence, Tuple, Union, cast
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
import time
import os
import operator

from collections import OrderedDict
import torch
from torch import nn

import detectron2
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.config import get_cfg
from determined.pytorch import samplers
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.samplers import TrainingSampler
from detectron2.data.dataset_mapper import DatasetMapper

from data import *
from detectron2.data.common import MapDataset



TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

class DetectronTrial(PyTorchTrial):
    def __init__(self, context):
        self.context = context
        self.cfg = self.setup_cfg()

        # model
        model = build_model(self.cfg)
        fi = self.context.get_hparam('data_loc')
        ch = DetectionCheckpointer(model)
        ch.resume_or_load(path = fi, resume=False)
        self.model = self.context.wrap_model(ch.model)

        # optimizer
        self.optimizer = build_optimizer(self.cfg, model)
        self.optimizer = self.context.wrap_optimizer(self.optimizer)

        # scheduler
        self.scheduler = build_lr_scheduler(self.cfg, self.optimizer)
        self.scheduler = self.context.wrap_lr_scheduler(self.scheduler, step_mode=LRScheduler.StepMode.STEP_EVERY_BATCH)

    def setup_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(self.context.get_hparam('model_yaml'))
        cfg.SOLVER.IMS_PER_BATCH = self.context.get_per_slot_batch_size()
        # cfg.freeze()
        return cfg# if you don't like any of the default setup, write your own setup code

    def build_training_data_loader(self):
        dataset_dicts = get_detection_dataset_dicts(
            self.cfg.DATASETS.TRAIN,
            filter_empty=self.cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=self.cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if self.cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=self.cfg.DATASETS.PROPOSAL_FILES_TRAIN if self.cfg.MODEL.LOAD_PROPOSALS else None,
        )
        dataset = DatasetFromList(dataset_dicts, copy=False)

        mapper = None
        if mapper is None:
            print ('MAPPER IS NONE')
            mapper = DatasetMapper(self.cfg, True)
        dataset = MapDataset(dataset, mapper)

        seed = self.context.get_trial_seed()
        rank = self.context.distributed.get_rank()
        size = self.context.distributed.get_size()
        skip = self.context.get_initial_batch()
        batch_size = self.context.get_per_slot_batch_size()

        # Shuffle-repeat-shard-batch-skip, just like Determined told me to.
        sampler = torch.utils.data.SequentialSampler(dataset)
        sampler = samplers.ReproducibleShuffleSampler(sampler, seed)
        sampler = samplers.RepeatSampler(sampler)
        sampler = samplers.DistributedSampler(sampler, num_workers=size, rank=rank)
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
        batch_sampler = samplers.SkipBatchSampler(batch_sampler, skip)


        data_loader = torch.utils.data.DataLoader(
            dataset,
            # sampler=train_sampler,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        self.data_loader = AspectRatioGroupedDataset(data_loader, batch_size)


        return self.data_loader

    def build_validation_data_loader(self):
        return self.data_loader


    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int):
        # print (len(batch))
        # print (batch[0])
        loss_dict = self.model(batch)
        losses = sum(loss_dict.values())

        loss_dict['loss'] = losses

        # loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # if comm.is_main_process():
        #     storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

        self.context.backward(losses)
        self.context.step_optimizer(self.optimizer)
        
        return loss_dict


    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        return {"validation_loss": 1}
    