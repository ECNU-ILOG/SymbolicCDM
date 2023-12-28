import torch
import torch.nn as nn
from tqdm import tqdm

from .eval import accuracy, area_under_curve, f1_score
from .utility import init_interaction_function


class ComputeIF(nn.Module):
    def __init__(self,
                 student_number,
                 question_number,
                 knowledge_number):
        super(ComputeIF, self).__init__()
        self.student_emb = nn.Embedding(student_number, knowledge_number)
        self.difficulty = nn.Embedding(question_number, knowledge_number)
        self.discrimination = nn.Embedding(question_number, 1)

        # initialize
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_id, question, q_matrix_line, interaction_func):
        proficiency_level = torch.sigmoid(self.student_emb(student_id))
        difficulty = torch.sigmoid(self.difficulty(question))
        discrimination = torch.sigmoid(self.discrimination(question))

        input_x = interaction_func(discrimination, proficiency_level - difficulty, q_matrix_line)
        output = torch.sigmoid(input_x)

        return output.view(-1)


class Parameter:
    def __init__(self,
                 student_number: int,
                 question_number: int,
                 knowledge_number: int,):
        self.net = ComputeIF(student_number, question_number, knowledge_number)
        self.student_number = student_number
        self.question_number = question_number
        self.knowledge_number = knowledge_number
        self.interaction_function = init_interaction_function
        self.interaction_function_string = "initial interaction function"

    def train(self, train_set, epochs, device="cpu", lr=0.002, init=True):
        # initialize
        if init:
            for name, param in self.net.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param)
        self.net = self.net.to(device)
        self.net.train()
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        with tqdm(total=epochs, desc="Training Process", unit="epoch") as pbar:
            for epoch in range(epochs):
                epoch_losses = []
                for batch_data in train_set:
                    student_id, question, q_matrix_line, y = batch_data
                    student_id: torch.Tensor = student_id.to(device)
                    question: torch.Tensor = question.to(device)
                    q_matrix_line: torch.Tensor = q_matrix_line.to(device)
                    y: torch.Tensor = y.to(device)
                    pred: torch.Tensor = self.net(student_id,
                                                  question,
                                                  q_matrix_line,
                                                  self.interaction_function)
                    loss = loss_function(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(loss.mean().item())
                pbar.update()

    def evaluate(self, test_set, interaction_func, device="cpu"):
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        for batch_data in test_set:
            student_id, question, q_matrix_line, y = batch_data
            student_id: torch.Tensor = student_id.to(device)
            question: torch.Tensor = question.to(device)
            q_matrix_line: torch.Tensor = q_matrix_line.to(device)
            pred: torch.Tensor = self.net(student_id,
                                          question,
                                          q_matrix_line,
                                          interaction_func)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        acc = accuracy(y_pred, y_true)
        auc = area_under_curve(y_pred, y_true)
        f1 = f1_score(y_pred, y_true)
        return acc, auc, f1,

    def unpack(self):
        proficiency_level = self.net.student_emb(torch.arange(0, self.student_number).to()).detach().cpu().numpy()
        difficulty = self.net.difficulty(torch.arange(0, self.question_number).to()).detach().cpu().numpy()
        discrimination = self.net.discrimination(torch.arange(0, self.question_number).to()).detach().cpu().numpy()
        return proficiency_level, difficulty, discrimination,

    def update(self, interaction_func, interaction_func_str):
        self.interaction_function = interaction_func
        self.interaction_function_string = interaction_func_str
