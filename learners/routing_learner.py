from .base_learner import BaseLearner


class RoutingLearner(BaseLearner):

    def __init__(self, opt):
        super().__init__(opt)

    def set_input(self, input):
        # unpack the input
        premise, hypothesis, meta, label = input
        self.premise = premise
        self.hypothesis = hypothesis
        self.meta = meta
        self.label = label

    def forward(self):
        pass

    def optimize_parameters(self):
        pass



