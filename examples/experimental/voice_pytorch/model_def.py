"""
This example is to show how to use an existing PyTorch model with Determined.
The flags and configurations can be found under const.yaml. For more information
regarding the optional flas view the original script linked below.

This implementation is based on:
https://github.com/huggingface/transformers/blob/v2.2.0/examples/run_glue.py

"""
from typing import Dict, Sequence, Union

import numpy as np
import torch
from torch import nn

import determined as det
from determined.pytorch import DataLoader, LRScheduler, PyTorchTrial

from data import *
from model.model import VoiceFilter

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class VoicePyTorch(PyTorchTrial):
    def __init__(self, context: det.TrialContext) -> None:
        self.context = context

        # Create a unique download directory for each rank so they don't overwrite each other.
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        self.data_downloaded = False

    def build_training_data_loader(self):
        train_dataset = VFDataset(self.context.get_data_config(), train=True)
        print ('LENGTH OF TRAIN DATASET: ', len(train_dataset))
        return DataLoader(train_dataset, batch_size=self.context.get_per_slot_batch_size(), collate_fn=train_collate_fn())

    def build_validation_data_loader(self):
        valid_dataset = VFDataset(self.context.get_data_config(), train=False)

        return DataLoader(valid_dataset, batch_size=self.context.get_per_slot_batch_size(), collate_fn=train_collate_fn())

    def build_model(self):
        model = VoiceFilter(self.context.get_data_config())

        self.embedder = SpeechEmbedder(self.context.get_data_config())#.cuda()
        self.embedder.load_state_dict(embedder_pt)
        self.embedder.eval()
        return model

    def optimizer(self, model: nn.Module):
        optimizer = torch.optim.Adam(model.parameters(),lr=self.context.get_data_config()['train']['adam'])
        return optimizer

    def train_batch(self, batch: TorchData, model: nn.Module, epoch_idx: int, batch_idx: int):
        dvec_mels, target_mag, mixed_mag = batch
        
        target_mag = target_mag#.cuda()
        mixed_mag = mixed_mag#.cuda()

        dvec_list = list()
        for mel in dvec_mels:
            mel = mel.cuda()
            dvec = self.embedder(mel)
            dvec_list.append(dvec)
        dvec = torch.stack(dvec_list, dim=0)
        dvec = dvec.detach()

        mask = model(mixed_mag, dvec)
        output = mixed_mag * mask

        loss = nn.MSELoss(output, target_mag)
        return results

    def evaluate_batch(self, batch: TorchData, model: nn.Module):

        return {'loss': 1}
