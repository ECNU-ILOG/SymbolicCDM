import torch
import numpy as np
import random
import operator

from pandas import DataFrame
from deap.gp import Primitive
from inspect import isclass
from torch.utils.data import TensorDataset, Dataset, DataLoader

from .operators import sigmoid


class StudentDataSet(Dataset):
    def __init__(self, loaded_data):
        """
        This class is designed for transforming loaded_data from np.ndarray to Dataset.
        """
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def exam(test_set, proficiency, difficulty, discrimination, interaction_func):
    """
    Simulate the interaction between students and questions.

    :param test_set: test set excluding the training data
    :param proficiency: proficiency level of each student
    :param difficulty: difficulty of each knowledge attributes in each questions
    :param discrimination: discrimination of questions
    :param interaction_func: compiled interaction function from genetic programming
    :return: prediction of response `y_pred` and true labels `y_true`
    """
    y_pred, y_true = [], []
    for batch_data in test_set:
        student_id_batch, question_batch, q_matrix_batch, y = list(map(np.array, batch_data))
        for student_id, question, q_matrix in zip(student_id_batch, question_batch, q_matrix_batch):
            p = sigmoid(proficiency[student_id])
            dk = sigmoid(difficulty[question])
            de = sigmoid(discrimination[question])
            pred = sigmoid(interaction_func(de, p - dk, q_matrix)).item()
            y_pred.append(pred)
        y_true.extend(y.tolist())
    y_pred = np.array(y_pred)
    y_pred = y_pred.tolist()
    return y_pred, y_true


def print_logs(metric, headers, title):
    print(title)
    df_string = DataFrame(data=[metric], columns=headers).to_string(index=False)
    print("-" * (len(df_string) // 2))
    print(df_string)
    print("-" * (len(df_string) // 2))


def transform(student_id, question, y, q_matrix=None):
    """
    Transform data to match the input of parameter optimization

    :return: torch.DataLoader(batch_size=32)
    """
    if q_matrix is None:
        dataset = TensorDataset(torch.tensor(student_id, dtype=torch.int64) - 1,
                                torch.tensor(question, dtype=torch.int64) - 1,
                                torch.tensor(y, dtype=torch.float32))
    else:
        q_matrix_line = q_matrix[question - 1]
        dataset = TensorDataset(torch.tensor(student_id, dtype=torch.int64) - 1,
                                torch.tensor(question, dtype=torch.int64) - 1,
                                q_matrix_line,
                                torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=32)


def mut_uniform_with_pruning(individual, pset, pruning=0.5):
    rand = np.random.uniform(0, 1)
    if rand < pruning:
        # pruning tree
        # We don't want to "shrink" the tree too much
        if len(individual) < 15 or individual.height <= 5:
            return individual,

        iprims = []
        for i, node in enumerate(individual[1:], 1):
            if isinstance(node, Primitive) and node.ret in node.args:
                iprims.append((i, node))

        if len(iprims) != 0:
            index, prim = random.choice(iprims)
            arg_idx = random.choice([i for i, type_ in enumerate(prim.args) if type_ == prim.ret])
            rindex = index + 1
            for _ in range(arg_idx + 1):
                rslice = individual.searchSubtree(rindex)
                subtree = individual[rslice]
                rindex += len(subtree)

            slice_ = individual.searchSubtree(index)
            individual[slice_] = subtree
    else:
        index = random.randrange(len(individual))
        node = individual[index]
        slice_ = individual.searchSubtree(index)
        choice = random.choice

        # As we want to keep the current node as children of the new one,
        # it must accept the return value of the current node
        primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

        if len(primitives) == 0:
            return individual,

        new_node = choice(primitives)
        new_subtree = [None] * len(new_node.args)
        position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

        for i, arg_type in enumerate(new_node.args):
            if i != position:
                term = choice(pset.terminals[arg_type])
                if isclass(term):
                    term = term()
                new_subtree[i] = term

        new_subtree[position:position + 1] = individual[slice_]
        new_subtree.insert(0, new_node)
        individual[slice_] = new_subtree

    return individual,


def sel_random(individuals, k):
    candidates = individuals
    return [random.choice(candidates) for i in range(k)]


def sel_tournament(individuals, k, tournament_size, fit_attr="fitness"):
    chosen = []
    for i in range(k):
        aspirants = sel_random(individuals, tournament_size)
        chosen.append(max(aspirants, key=operator.attrgetter(fit_attr)))
    return chosen


def init_interaction_function(discrimination, proficiency, q_matrix_line):
    if type(proficiency) is np.ndarray:
        return discrimination * np.sum(proficiency * q_matrix_line)
    else:
        return discrimination * (proficiency * q_matrix_line).sum(dim=1).unsqueeze(1)
