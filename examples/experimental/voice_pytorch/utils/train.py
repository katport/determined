import os
import math
import torch
import torch.nn as nn
import traceback

from .adabound import AdaBound
from .audio import Audio
from .evaluation import validate
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder


def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    # load embedder
    embedder_pt = torch.load(args.embedder_path)
    embedder = SpeechEmbedder(hp).cuda()
    embedder.load_state_dict(embedder_pt)
    embedder.eval()

    audio = Audio(hp)
    model = VoiceFilter(hp).cuda()
    step = 0

    try:
        criterion = nn.MSELoss()
        while True:
            model.train()
            for dvec_mels, target_mag, mixed_mag in trainloader:
                target_mag = target_mag.cuda()
                mixed_mag = mixed_mag.cuda()

                dvec_list = list()
                for mel in dvec_mels:
                    mel = mel.cuda()
                    dvec = embedder(mel)
                    dvec_list.append(dvec)
                dvec = torch.stack(dvec_list, dim=0)
                dvec = dvec.detach()

                mask = model(mixed_mag, dvec)
                output = mixed_mag * mask

                # output = torch.pow(torch.clamp(output, min=0.0), hp.audio.power)
                # target_mag = torch.pow(torch.clamp(target_mag, min=0.0), hp.audio.power)
                loss = criterion(output, target_mag)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                loss = loss.item()
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
