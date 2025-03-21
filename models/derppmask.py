# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Derppmask(ContinualModel):
    NAME = 'derppmask'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derppmask, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.LongTensor(size=(0,)).to(self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        pre_masked_logits = logits

        mask  = torch.zeros_like(logits)
        # unmask classes present in incoming batch
        mask[:, present] = 1
        # also unmask classes which we have not seen before
        mask[:, self.seen_so_far.max():] = 1

        logits = logits.masked_fill(mask == 0, -1e9)

        loss = self.loss(logits, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=pre_masked_logits.data)

        return loss.item()
