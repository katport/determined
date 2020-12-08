from typing import Any, Dict, Sequence, Tuple, Union, cast
import os
import argparse
import time
import yaml
import logging
import abc
import json
import random
# from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from torch import nn
from torchvision import transforms
import pickle as pkl
import torchvision
import math
from contextlib import suppress
import apex
from apex import amp
import horovod

from pycocotools.cocoeval import COCOeval
import effdet.evaluation.detection_evaluator as tfm_eval

import torch
import torchvision.utils

from effdet import create_model, unwrap_bench, create_loader, create_dataset#, create_evaluator
from effdet.data.loader import DetectionFastCollate
from effdet.data import resolve_input_config, SkipSubset
from effdet.data.transforms import *
from effdet.anchors import Anchors, AnchorLabeler
from timm.models import resume_checkpoint, load_checkpoint
from timm.utils import *
from timm.utils.model import unwrap_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
import numpy as np
import pycocotools

import logging
from collections import OrderedDict
from copy import deepcopy

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler, PyTorchCallback
from horovod.torch.sync_batch_norm import SyncBatchNorm
import sys
from determined.experimental import Determined

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

MAX_NUM_INSTANCES = 100


class EffDetTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.hparam = self.context.get_hparam
        self.args = DotDict(self.context.get_hparams())
        # Create a unique download directory for each rank so they don't overwrite each other.
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        self.num_slots = int(self.context.get_experiment_config()['resources']['slots_per_trial'])

        # slots = int(self.context.get_experiment_config()['resources']['slots_per_trial'])
        # if slots > 1:
        #     self.args.distributed = slots
        # print ('dtrian: ', self.args.distributed)

        self.args.pretrained_backbone = not self.args.no_pretrained_backbone
        self.args.prefetcher = not self.args.no_prefetcher

        tmp = []
        for arg in self.args.lr_noise.split(' '):
            tmp.append(float(arg))
        self.args.lr_noise = tmp

        # print (self.args)
        # print (self.args.model,self.args.num_classes, self.args.pretrained, self.args.pretrained_backbone, self.args.redundant_bias, self.args.smoothing, self.args.new_focal, self.args.jit_loss, self.args.bench_labeler, self.args.initial_checkpoint)
        print ('seed: ', self.args.seed + self.context.distributed.get_rank())
        # torch.manual_seed(self.args.seed + self.context.distributed.get_rank())
        # random.seed(42)
        # np.random.seed(42)
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
 
        if self.args.sync_bn:
            print ('creating batch sync model')
            if self.args.model_ema:
                print ('creating batch sync ema model')

                self.model_ema = self.context.wrap_model(deepcopy(self.model))
            self.model = self.convert_syncbn_model(self.model)

        self.model = self.context.wrap_model(self.model)
        print ('Model created, param count:' , self.args.model, sum([m.numel() for m in self.model.parameters()]))

        self.optimizer = self.context.wrap_optimizer(create_optimizer(self.args, self.model))
        print ('Created optimizer: ', self.optimizer)

        if self.args.amp:
            print ('using amp')
            if self.args.sync_bn and self.args.model_ema:
                print ('using sync_bn and model_ema when creating apex_amp')
                (self.model, self.model_ema), self.optimizer = self.context.configure_apex_amp([self.model, self.model_ema], self.optimizer, min_loss_scale=self.hparam("min_loss_scale"))
            else:
                self.model, self.optimizer = self.context.configure_apex_amp(self.model, self.optimizer, min_loss_scale=self.hparam("min_loss_scale"))

        # self.amp_autocast = suppress
        if self.args.model_ema:
            print ('using model ema')
            if self.args.sync_bn:
                print ('using model ema batch syn')
                self.model_ema = ModelEma(self.model_ema, context=self.context,decay=self.args.model_ema_decay, rank=self.context.distributed.get_rank())
            else:
                # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
                self.model_ema = ModelEma(self.model, context=self.context,decay=self.args.model_ema_decay, rank=self.context.distributed.get_rank())

        self.lr_scheduler, self.num_epochs = create_scheduler(self.args, self.optimizer)
        self.lr_scheduler = self.context.wrap_lr_scheduler(self.lr_scheduler, LRScheduler.StepMode.MANUAL_STEP)

        self.cur_epoch = 0
        self.num_updates  = 0 * self.cur_epoch

        if self.args.prefetcher:
            self.train_mean, self.train_std, self.train_random_erasing = self.calculate_means(mean=self.input_config['mean'], std=self.input_config['std'], re_prob=self.args.reprob, re_mode=self.args.remode, re_count=self.args.recount)
            
            self.val_mean, self.val_std, self.val_random_erasing = self.calculate_means(self.input_config['mean'], self.input_config['std'])
        

    def calculate_means(self,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            re_prob=0.,
            re_mode='pixel',
            re_count=1):
        # We need to precalculate the prefetcher. 
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

            print ('input_size', input_size)
            print ('batch_size', batch_size )
            print ('is_training', is_training)
            print ('use_prefetcher', use_prefetcher)
            print ('re_prob', re_prob)
            print ('re_mode', re_mode)
            print ('re_count', re_count)
            print ('interpolation', interpolation)
            print ('fill_color', fill_color)
            print ('mean', mean)
            print ('std', std)
            print ('num_workers', num_workers)
            print ('distributed', distributed)
            print ('pin_mem', pin_mem)
            print ('anchor_labeler', anchor_labeler)

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
                shuffle=False, #sampler is None and is_training,
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
            self.labeler = AnchorLabeler(Anchors.from_config(self.model_config), self.model_config.num_classes, match_threshold=0.5)

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
        print ('train_dataset_length: ', len(dataset_train))
        print ('loader_train: ', self.data_length)
        print ('num_classes: ', self.model_config.num_classes)

        return loader_train


    def build_validation_data_loader(self):
        if self.args.val_skip > 1:
            self.dataset_eval = SkipSubset(self.dataset_eval, self.args.val_skip)
        self.loader_eval = self._create_loader(
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
        
        self.evaluator = self.create_evaluator(self.args.dataset, self.loader_eval.dataset, pred_yxyx=False)

        self.eval_counter = 0
        print ('val_dataset_length: ', len(self.dataset_eval))
        print ('loader_val: ', len(self.loader_eval))
        return self.loader_eval

    def clip_grads(self,params):
        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.clip_grad)

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int):
        if epoch_idx != self.cur_epoch and self.lr_scheduler is not None:
            self.cur_epoch = epoch_idx
            self.num_updates = epoch_idx * self.data_length
            
            self.lr_scheduler.step(self.cur_epoch)
            lrl2 = [param_group['lr'] for param_group in self.optimizer.param_groups]
            print (self.cur_epoch, 'new lr: ', lrl2, batch_idx)

        input, target = batch

        if self.args.prefetcher:
            input = input.float().sub_(self.train_mean).div_(self.train_std)
            if self.train_random_erasing is not None:
                input = self.train_random_erasing(input, target)

        if self.args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # with self.amp_autocast():
        output = self.model(input, target)
        loss = output['loss']

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer, self.clip_grads)
        
        if self.model_ema is not None:
            self.model_ema.update(self.model)

        self.num_updates += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step_update(num_updates=self.num_updates) 
            lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]

        return {"loss": loss}

    def evaluate_batch(self, batch: TorchData):
        input, target = batch
        if self.args.prefetcher:
            input = input.float().sub_(self.val_mean).div_(self.val_std)

        if self.val_random_erasing is not None:
            input = self.val_random_erasing(input, target)

        input = self.context.to_device(input)
        target = self.context.to_device(target)

        output = self.model_ema.ema(input, target)
        loss = output['loss']

        if self.evaluator is not None:
            self.evaluator.add_predictions(output['detections'], target)

        reduced_loss = loss.data.item()
        
        self.eval_counter += 1

        if reduced_loss is np.nan or math.isnan(reduced_loss):
            print ('nan',reduced_loss, self.context.distributed.get_rank())
            for name, p in self.model.named_parameters():
                print(batch_idx, name,'norm: ', p.grad.norm().item(), 'sum: ', p.grad.sum().item(), 'max: ', p.grad.min().item(), 'min: ', p.grad.min().item())
            reduced_loss = 0
        if self.eval_counter == len(self.loader_eval)/self.num_slots and self.evaluator is not None:

            local_imgs = torch.Tensor(self.evaluator.img_indices).type(torch.IntTensor)
            local_predictions = torch.Tensor(self.evaluator.predictions)
            self.evaluator.reset()
            self.eval_counter = 0

            # print ('rank: ', self.context.distributed.get_rank(), 'localimgs: ', len(local_imgs))
            concat_imgs = horovod.torch.allgather(local_imgs)
            concat_pred = horovod.torch.allgather(local_predictions)

            # print ('rank: ', self.context.distributed.get_rank(), len(concat_imgs))
            if self.context.distributed.get_rank() == 0:
                concat_imgs = concat_imgs.numpy()
                concat_pred = concat_pred.numpy()

                self.evaluator.img_indices = concat_imgs
                self.evaluator.predictions = concat_pred
                
                metrics_map = float(self.evaluator.evaluate())
                print ('ran the metrics: ', metrics_map)
                self.evaluator.reset()

                return {'val_loss':reduced_loss, 'map' : metrics_map*len(self.loader_eval)}

        return {'val_loss': reduced_loss, 'map': 0}
    
    def create_evaluator(self,name, dataset, pred_yxyx=False):
        if 'coco' in name:
            return CocoEvaluator(dataset, pred_yxyx=pred_yxyx)

    def convert_syncbn_model(self, module, process_group=None, channel_last=False):
        '''
        Recursively traverse module and its children to replace all instances of
        ``torch.nn.modules.batchnorm._BatchNorm`` with :class:`apex.parallel.SyncBatchNorm`.

        All ``torch.nn.BatchNorm*N*d`` wrap around
        ``torch.nn.modules.batchnorm._BatchNorm``, so this function lets you easily switch
        to use sync BN.

        Args:
            module (torch.nn.Module): input module

        Example::

            >>> # model is an instance of torch.nn.Module
            >>> import apex
            >>> sync_bn_model = apex.parallel.convert_syncbn_model(model)
        '''
        mod = module
        if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
            return module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            mod = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats) #, process_group, channel_last=channel_last
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()
        for name, child in module.named_children():
            mod.add_module(name, self.convert_syncbn_model(child,
                                                    process_group=process_group,
                                                    channel_last=channel_last))
        # TODO(jie) should I delete model explicitly?
        del module
        return mod

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


class Evaluator:

    def __init__(self, pred_yxyx=False):
        self.pred_yxyx = pred_yxyx
        self.img_indices = []
        self.predictions = []

    def add_predictions(self, detections, target):
        img_indices = target['img_idx']

        detections = detections.cpu().numpy()
        img_indices = img_indices.cpu().numpy()
        for img_idx, img_dets in zip(img_indices, detections):
            self.img_indices.append(img_idx)
            self.predictions.append(img_dets)

    def _coco_predictions(self):
        # generate coco-style predictions
        coco_predictions = []
        coco_ids = []
        for img_idx, img_dets in zip(self.img_indices, self.predictions):
            img_id = self._dataset.img_ids[img_idx]
            coco_ids.append(img_id)
            if self.pred_yxyx:
                # to xyxy
                img_dets[:, 0:4] = img_dets[:, [1, 0, 3, 2]]
            # to xywh
            img_dets[:, 2] -= img_dets[:, 0]
            img_dets[:, 3] -= img_dets[:, 1]
            for det in img_dets:
                score = float(det[4])
                if score < .001:  # stop when below this threshold, scores in descending order
                    break
                coco_det = dict(
                    image_id=int(img_id),
                    bbox=det[0:4].tolist(),
                    score=score,
                    category_id=int(det[5]))
                coco_predictions.append(coco_det)
        return coco_predictions, coco_ids

    @abc.abstractmethod
    def evaluate(self):
        pass

class CocoEvaluator(Evaluator):
    def __init__(self, dataset, pred_yxyx=False):
        super().__init__(pred_yxyx=pred_yxyx)
        self._dataset = dataset.parser
        self.coco_api = dataset.parser.coco

    def reset(self):
        self.img_indices = []
        self.predictions = []

    def evaluate(self):
        coco_predictions, coco_ids = self._coco_predictions()
        json.dump(coco_predictions, open('./temp.json', 'w'), indent=4)
        results = self.coco_api.loadRes('./temp.json')
        coco_eval = COCOeval(self.coco_api, results, 'bbox')
        coco_eval.params.imgIds = coco_ids  # score only ids we've used
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metric = coco_eval.stats[0]  # mAP 0.5-0.95
        return metric



class ModelEma:
    """ Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """
    def __init__(self, model, decay=0.9999, context='', resume='', rank=0):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.context = context
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            print ('resuming')
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.rank = rank
        
    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                # ema model may have been wrapped by DataParallel, and need module prefix
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            _logger.info("Loaded state_dict_ema")
        else:
            _logger.warning("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                model_v = self.context.to_device(model_v)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)


    # def hold_evaluate_full_dataset(self, data_loader):
    #     losses_m = AverageMeter()
    #     print ('val rank: ', self.context.distributed.get_rank())
    #     print ('val ema rank: ', self.model_ema.rank)
    #     print ('val length: ', len(data_loader))

    #     with torch.no_grad():
    #         for batch_idx, (input, target) in enumerate(data_loader):
    #             if batch_idx % 300 == 0:
    #                 print ('batch: ', batch_idx, "/", len(data_loader), input.shape)
    #             try:
    #                 if self.args.prefetcher:
    #                     input = input.float().sub_(self.val_mean.cpu()).div_(self.val_std.cpu())

    #                     if self.val_random_erasing is not None:
    #                         input = self.val_random_erasing(input, target)

    #                 input = self.context.to_device(input)
    #                 target = self.context.to_device(target)

    #                 output = self.model_ema.ema(input, target)
    #                 loss = output['loss']

    #                 if self.evaluator is not None:
    #                     self.evaluator.add_predictions(output['detections'], target)

    #                 if loss is np.nan or math.isnan(loss):
    #                     print ('nan batch_idx: ', batch_idx)
    #                     for name, p in self.model.named_parameters():
    #                         print(batch_idx, name,'norm: ', p.grad.norm().item(), 'sum: ', p.grad.sum().item(), 'max: ', p.grad.min().item(), 'min: ', p.grad.min().item())
    #                     continue
    #                 reduced_loss = loss.data
    #                 torch.cuda.synchronize() # horovod.torch.synchronize?
    #                 losses_m.update(reduced_loss.item(), input.size(0))
    #             except Exception as e:
    #                 print ('failed batch_idx: ', batch_idx)
    #                 raise e
                    

    #     metrics = {'val_loss': losses_m.avg}
    #     if self.evaluator is not None:
    #         metrics['map'] = float(self.evaluator.evaluate())
        
    #     return metrics

        
        # gradient= None,
        # retain_graph = False
        # create_graph = False

        # if (self.context._last_backward_batch_idx is None
        #         or self.context._current_batch_idx is None
        #         or self.context._last_backward_batch_idx < self.context._current_batch_idx
        #     ):
        #         self.context._last_backward_batch_idx = self.context._current_batch_idx
        # else:
        #     raise det.errors.InvalidExperimentException(
        #         "Calling context.backward(loss) multiple times is not supported "
        #         "while using apex.amp and parallel/distributed training"
        #         )

        # if loss not in self.context._loss_ids:
        #     self.context._loss_ids[loss] = len(self.context._loss_ids)
        
        # with apex.amp.scale_loss(loss, self.context.optimizers, loss_id=self.context._loss_ids[loss]) as scaled_loss:
        #     print ('before',scaled_loss)
            
        #     scaled_loss.backward(gradient=gradient, retain_graph=retain_graph, create_graph=create_graph)
        #     print ('after: ', scaled_loss)
        #     # print ('optimizers: ', self.context.optimizers)
        #     for optimizer in self.context.optimizers:
        #         # optimizer.synchronize()
        #         print ('opt: ', optimizer)
        #         missing_p = optimizer._requires_update - set(optimizer._handles.keys())
        #         print ('missing_p: ', missing_p)
        #         for p in missing_p:
        #             print ('p: ', p.grad)

        # t = t

        # fi = open('/mnt/data/det_input.pkl', 'wb+')
        # pkl.dump(input, fi)
        # fi.close()
        # fi = open('/mnt/data/det_target.pkl', 'wb+')
        # pkl.dump(target, fi)
        # fi.close()
        # t = t
        # fi = open('/mnt/data/input_data.pkl','rb')
        # input = pkl.load(fi)
        # fi.close()

        # fi = open('/mnt/data/target_data.pkl','rb')
        # target = pkl.load(fi)
        # fi.close()
        # print ('loss: ', loss)
        # print ('output: ', output)
        # print(list(self.model.parameters())[-1].grad)
        # t = t