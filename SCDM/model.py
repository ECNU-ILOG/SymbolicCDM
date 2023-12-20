import numpy as np
import warnings
from torch.utils.data import random_split

from .utility import StudentDataSet, printLogs
from .interaction import GeneticInteractionFunc
from .argument import GeneticArgumentSearch


class EvolCDM:
    def __init__(self, q_matrix: np.ndarray,
                 student_number: int,
                 question_number: int,
                 knowledge_number: int,
                 response_logs: np.ndarray):
        # because training will generate warning, we want to silence then
        warnings.filterwarnings("ignore")
        # dataset split
        response_logs = StudentDataSet(response_logs)
        train_size = int(len(response_logs) * 0.75)
        valid_size = len(response_logs) - train_size
        self.train_set, self.valid_set = random_split(response_logs, [train_size, valid_size])

        # cognitive diagnosis modules: interaction function and arguments
        self.interaction = GeneticInteractionFunc(q_matrix,
                                                  boosting=5,
                                                  train_set=self.train_set)

        self.argument = GeneticArgumentSearch(q_matrix,
                                              student_number,
                                              question_number,
                                              knowledge_number,
                                              self.train_set, )
        self.logs = dict({})

    def train(self, epochs):
        # for logs
        headers = ["accuracy", "RMSE", "AUC", "f1 score", "DOA"]
        print("===============Start training===============")
        for epoch in range(0, epochs):
            print("[Epoch: {}]".format(epoch + 1))
            # Search arguments in parallel computing strategy
            # self.argument.train(init=True if epoch == 0 else False)
            self.argument.train(init=True)
            print("The {}-th epoch arguments search complete".format(epoch + 1))
            temp_argument = self.argument.unpack()
            # Evaluate argument on train set
            printLogs(list(self.argument.evaluate(self.train_set, *temp_argument)), headers, "Metric in train set")
            # Evaluate argument on valid set
            printLogs(list(self.argument.evaluate(self.valid_set, *temp_argument)), headers, "Metric in valid set")
            # Training interaction function
            # Update argument
            self.interaction.update(*temp_argument)
            # Training interaction function
            print("Training interaction function...")
            self.interaction.train()
            print("The {}-th epoch best three interaction functions:".format(epoch + 1))
            # Update interaction function
            self.argument.update(self.interaction.function(), str(self.interaction))
            metric = list(self.interaction.evaluation(self.train_set))
            printLogs(metric, headers, "Metric in train set in this epoch")
            metric = list(self.interaction.evaluation(self.valid_set))
            printLogs(metric, headers, "Metric in valid set in this epoch")
            self.logs[epoch] = dict(zip(headers, metric))


class NeuralEvolCDM:
    pass