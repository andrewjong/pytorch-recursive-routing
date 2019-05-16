import argparse
from functools import reduce
from torch import nn
from torch.optim import lr_scheduler


class NLayerFunction(nn.Module):
    """
    A simple function for the recursive router to choose from
    """

    def __init__(self, input_dim, n_out, hidden_dim=256, n_hidden=3):
        """

        Args:
            input_dim: HxWxC tuple
            n_out (int): number of output classes
            hidden_dim: size of the hidden layer
            n_hidden: how many hidden layers
        """
        # get total input volume, excluding the batch size
        super().__init__()
        in_volume = reduce(lambda x, y: x * y, input_dim[1:])

        # input layer
        layers = [nn.Linear(in_volume, hidden_dim), nn.ReLU()]

        # hidden layers
        hidden_block = [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        # have n hidden layers
        layers += hidden_block * n_hidden

        # output layer
        layers += [nn.Linear(hidden_dim, n_out), nn.Softmax()]

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        self.model(input)


def get_scheduler(optimizer, opt: argparse.Namespace):
    """
    Construct a scheduler for the optimizer based on the commandline options
    Args:
        optimizer:
        opt:

    Returns:

    """
    if opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1
        )
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    else:
        return NotImplementedError(
            f"learning rate policy {opt.lr_policy} is not implemented"
        )

    return scheduler
