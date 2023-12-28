import warnings
import numpy as np
import torch
import pprint

from torch.utils.data import random_split

from .utility import StudentDataSet, print_logs, transform
from .interaction import GeneticInteractionFunc
from .parameter import Parameter
from .eval import degree_of_agreement


class SymbolicCDM:
    def __init__(self,
                 q_matrix: np.ndarray,
                 student_number: int,
                 question_number: int,
                 knowledge_number: int,
                 response_logs: np.ndarray,
                 device="cpu"):
        # dataset split
        response_logs = StudentDataSet(response_logs)
        # organize dataset
        train_size = int(len(response_logs) * 0.75)
        valid_size = len(response_logs) - train_size
        train_set, valid_set = random_split(response_logs, [train_size, valid_size])
        train_set = np.array(train_set)
        valid_set = np.array(valid_set)
        self.train_set = transform(train_set[:, 0], train_set[:, 1], train_set[:, 2], torch.Tensor(q_matrix))
        self.train_size = train_size
        self.valid_set = transform(valid_set[:, 0], valid_set[:, 1], valid_set[:, 2], torch.Tensor(q_matrix))

        self.interaction = GeneticInteractionFunc(self.train_set, train_size)
        self.parameter = Parameter(student_number,
                                   question_number,
                                   knowledge_number)

        self.q_matrix = q_matrix
        self.logs = dict({})
        self.device = device

    def train(self, epochs, nn_epochs):
        # for logs
        headers = ["accuracy", "AUC", "f1 score"]
        print("===============Start training===============")
        for epoch in range(0, epochs):
            print(f"[Epoch: {epoch + 1}]")
            self.parameter.train(self.train_set, epochs=nn_epochs, device=self.device, init=(epoch == 0))
            print(f"The {epoch + 1}-th epoch parameters optimization complete")
            # update arguments
            arguments = self.parameter.unpack()
            self.interaction.update(*arguments)
            # calculate degree of agreement
            doa = degree_of_agreement(self.q_matrix, arguments[0], self.valid_set)
            print(f"DOA in this epoch: {doa}")
            # evaluate argument on valid set
            print_logs(list(self.interaction.evaluation(self.valid_set)), headers, "Metric in valid set")
            # Training interaction function
            print("Training interaction function...")
            self.interaction.train()
            print(f"The {epoch+1}-th epoch interaction function complete")
            # Update interaction function
            self.parameter.update(self.interaction.function(), str(self.interaction))
            metric = list(self.interaction.evaluation(self.valid_set))
            print_logs(metric, headers, "Metric in valid set in this epoch")

            self.logs[epoch] = dict(zip(headers, metric))
            self.logs[epoch]["doa"] = doa
        pprint.pprint(self.logs)

