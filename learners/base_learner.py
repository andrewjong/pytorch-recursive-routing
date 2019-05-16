import os
from collections import OrderedDict

import torch
from abc import ABC, abstractmethod
import argparse

from . import networks


class BaseLearner(ABC):
    def __init__(self, opt: argparse.Namespace):
        self.opt = opt
        self.is_train = opt.is_train
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # speed up train time with benchmark
        torch.backends.cudnn.benchmark = True

        self.model_names = []
        self.loss_names = []
        self.schedulers = []
        self.metric = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing
        options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this
                                flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing
        steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and
        <test>. """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every
        training iteration """
        pass

    def setup(self):
        """
        Load and print networks; create schedulers
        Args:
            opt:

        Returns:

        """
        if self.is_train:
            self.schedulers = [
                networks.get_scheduler(optimizer, self.opt)
                for optimizer in self.optimizers
            ]

        if not self.is_train or self.opt.continue_train:
            self.load_networks(self.opt.load_epoch, self.opt.load_iter)
        self.print_networks()

    def get_current_losses(self):
        loss_values = OrderedDict()
        for name in self.loss_names:
            loss_values[name] = getattr(self, "loss_" + name)
        return loss_values

    def update_learning_rates(self):
        """
        Update the learning rate schedulers for all networks.
        Returns:

        """
        for scheduler in self.schedulers:
            if self.opt.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

    def save_networks(self, epoch):
        for name in self.model_names:
            save_file = f"net_{name}_epoch_{epoch}.pth"
            save_path = os.path.join(self.save_dir, save_file)
            net = getattr(self, "net" + name)

    def load_networks(self, load_epoch, load_iter):
        pass

    def print_networks(self):
        pass

    def test(self):
        with torch.no_grad():
            self.forward()
