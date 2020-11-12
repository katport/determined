from typing import Any, Dict, Sequence, Tuple, Union, cast
import os
import argparse
import time
import yaml
import logging
# from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torchvision.utils

from effdet import create_model, unwrap_bench, create_loader, create_dataset, create_evaluator
from effdet.data.loader import DetectionFastCollate
from effdet.data import resolve_input_config, SkipSubset
from effdet.data.transforms import *
from effdet.anchors import Anchors, AnchorLabeler
from timm.models import resume_checkpoint, load_checkpoint
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

# torch.backends.cudnn.benchmark = True
MAX_NUM_INSTANCES = 100

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

class DotDict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if value == 'None':
                value = None
            self[key] = value
    def __getattr__(self, name):
        try:
            t = self[name]
            return t
        except: 
            return None

class MNistTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.hparam = self.context.get_hparam
        # Create a unique download directory for each rank so they don't overwrite each other.
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"

        self.args = DotDict(self.context.get_hparams())

        self.args.pretrained_backbone = not self.args.no_pretrained_backbone
        self.args.prefetcher = not self.args.no_prefetcher

        print ('num_classes: ', self.args.num_classes, type(self.args.num_classes))
        print ('workers: ', self.args.workers, type(self.args.workers))

        print (self.args)
        self.model = create_model(
            self.args.model,
            bench_task='train',
            num_classes=self.args.num_classes,
            pretrained=self.args.pretrained,
            pretrained_backbone=self.args.pretrained_backbone,
            redundant_bias=self.args.redundant_bias,
            label_smoothing=self.args.smoothing,
            new_focal=self.args.new_focal,
            jit_loss=self.args.jit_loss,
            bench_labeler=self.args.bench_labeler,
            checkpoint_path=self.args.initial_checkpoint,
        )   
        self.model_config = self.model.config 
        self.input_config = resolve_input_config(self.args, model_config=self.model_config)

        self.model = self.context.wrap_model(self.model)
        print ('Model created, param count:' , self.args.model, sum([m.numel() for m in self.model.parameters()]))

        self.optimizer = self.context.wrap_optimizer(create_optimizer(self.args, self.model))

        self.model_ema = None
        if self.args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.model_ema = ModelEma(self.model, decay=self.args.model_ema_decay)

        self.lr_scheduler, num_epochs = create_scheduler(self.args, self.optimizer)
        self.lr_scheduler = self.context.wrap_lr_scheduler(self.lr_scheduler, LRScheduler.StepMode.MANUAL_STEP)
        self.lr_scheduler.step(0) # 0 start_epoch
        self.last_epoch = 0
        self.num_updates  = 0 * self.last_epoch

        if self.args.prefetcher:
            self.train_mean, self.train_std, self.train_random_erasing = self.calculate_means(self.input_config['mean'], self.input_config['std'], self.args.reprob, self.args.remode, self.args.recount)

            self.val_mean, self.val_std, self.val_random_erasing = self.calculate_means(self.input_config['mean'], self.input_config['std'])
        
        self.amp_autocast = suppress


    def calculate_means(self,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            re_prob=0.,
            re_mode='pixel',
            re_count=1):
        mean = torch.tensor([x * 255 for x in mean]).cuda().view(1, 3, 1, 1)
        std = torch.tensor([x * 255 for x in std]).cuda().view(1, 3, 1, 1)
        
        if re_prob > 0.:
            random_erasing = RandomErasing(probability=re_prob, mode=re_mode, max_count=re_count)
        else:
            random_erasing = None
        
        return mean, std, random_erasing

    def _create_loader(self,
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        re_prob=0.,
        re_mode='pixel',
        re_count=1,
        interpolation='bilinear',
        fill_color='mean',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        pin_mem=False,
        anchor_labeler=None,
    ):

        if isinstance(input_size, tuple):
            img_size = input_size[-2:]
        else:
            img_size = input_size

        if is_training:
            transform = transforms_coco_train(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                fill_color=fill_color,
                mean=mean,
                std=std)
        else:
            transform = transforms_coco_eval(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                fill_color=fill_color,
                mean=mean,
                std=std)

        dataset.transform = transform

        sampler = None

        collate_fn = DetectionFastCollate(anchor_labeler=anchor_labeler)
        loader = DataLoader(
            dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=sampler is None and is_training,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_mem,
            collate_fn=collate_fn,
        )


        return loader

    def build_training_data_loader(self):
        dataset_train, self.dataset_eval = create_dataset(self.args.dataset, self.args.root)

        self.labeler = None
        if not self.args.bench_labeler:
            self.labeler = AnchorLabeler(
                Anchors.from_config(self.model_config), self.model_config.num_classes, match_threshold=0.5)

        loader_train = self._create_loader(
            dataset_train,
            input_size=self.input_config['input_size'],
            batch_size=self.context.get_per_slot_batch_size(),
            is_training=True,
            use_prefetcher=self.args.prefetcher,
            re_prob=self.args.reprob,
            re_mode=self.args.remode,
            re_count=self.args.recount,
            # color_jitter=self.args.color_jitter,
            # auto_augment=self.args.aa,
            interpolation=self.args.train_interpolation or self.input_config['interpolation'],
            fill_color=self.input_config['fill_color'],
            mean=self.input_config['mean'],
            std=self.input_config['std'],
            num_workers=self.args.workers,
            distributed=self.args.distributed,
            pin_mem=self.args.pin_mem,
            anchor_labeler=self.labeler,
        )
        if self.model_config.num_classes < loader_train.dataset.parser.max_label:
            logging.error(
                f'Model {self.model_config.num_classes} has fewer classes than dataset {loader_train.dataset.parser.max_label}.')
            exit(1)
        if self.model_config.num_classes > loader_train.dataset.parser.max_label:
            logging.warning(
                f'Model {self.model_config.num_classes} has more classes than dataset {loader_train.dataset.parser.max_label}.')

        self.data_length = len(loader_train)
        return loader_train

    def build_validation_data_loader(self):
        loader_eval = self._create_loader(
            self.dataset_eval,
            input_size=self.input_config['input_size'],
            batch_size=self.context.get_per_slot_batch_size(),
            is_training=False,
            use_prefetcher=self.args.prefetcher,
            interpolation=self.input_config['interpolation'],
            fill_color=self.input_config['fill_color'],
            mean=self.input_config['mean'],
            std=self.input_config['std'],
            num_workers=self.args.workers,
            distributed=self.args.distributed,
            pin_mem=self.args.pin_mem,
            anchor_labeler=self.labeler,
        )
        return loader_eval
    
    def clip_grads(self,params):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)


    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int):
        # last_batch = batch_idx == last_idx
        input, target = batch

        if self.args.prefetcher:
            input = input.float().sub_(self.train_mean).div_(self.train_std)
            if self.train_random_erasing is not None:
                input = self.train_random_erasing(input, target)

        if self.args.channels_last:
            print ('should skip')
            input = input.contiguous(memory_format=torch.channels_last)

        print ('sum: input0: ', torch.sum(input[0]))
        with self.amp_autocast():
            output = self.model(input, target)
        # input=input.float()
        # output = self.model(input, target)    
        loss = output['loss']

        # self.context.backward(loss)
        # self.context.step_optimizer(self.optimizer, self.clip_grads)

        # self.num_updates += 1

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step_update(num_updates=self.num_updates, metric=loss) # metric 
        #     #is not used
        #     lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
        #     # print ('lrl: ', lrl)
        #     if epoch_idx != self.last_epoch:
        #         print ('new epoch: ', epoch_idx, self.last_epoch)
        #         # step LR for next epoch
        #         self.lr_scheduler.step(epoch_idx + 1, loss)
        #         self.last_epoch = epoch_idx
        #         self.num_updates = epoch_idx * self.data_length

        print ('loss: ', loss.item())

        return {'loss': loss.item(), 'test': 1}

    def evaluate_batch(self, batch: TorchData):
        # the overhead of evaluating with coco style datasets is fairly high, so just ema or non, not both
        if model_ema is not None:
            if self.args.distributed and self.args.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model_ema, self.args.world_size, self.args.dist_bn == 'reduce')

            eval_metrics = validate(model_ema.ema, loader_eval, self.args, evaluator, log_suffix=' (EMA)')
        else:
            eval_metrics = validate(model, loader_eval, self.args, evaluator)
        return {"validation_loss": validation_loss, "accuracy": accuracy}
