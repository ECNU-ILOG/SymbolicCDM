from torch.utils.data import Dataset
from deap.gp import Primitive
from inspect import isclass
from pandas import DataFrame
import numpy as np
import random
import ray

from .operators import tanh, dot


class StudentDataSet(Dataset):
    def __init__(self, loaded_data):
        self.data = loaded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def initInteractionFunction(q_matrix: np.ndarray,
                            proficiency_level: np.ndarray,
                            discrimination: np.ndarray, ):
    return dot(discrimination * proficiency_level, q_matrix)


def printLogs(metric, headers, title):
    print(title)
    df_string = DataFrame(data=[metric], columns=headers).to_string(index=False)
    print("-" * (len(df_string) // 2))
    print(df_string)
    print("-" * (len(df_string) // 2))


@ray.remote
def parallelCompute(obj, arg: tuple):
    obj.train(*arg)
    return obj


def exam(student_data: StudentDataSet,
         q_matrix: np.ndarray,
         proficiency_level: np.ndarray,
         discrimination: np.ndarray,
         interaction_function):
    prediction = []
    truth = []
    for line in student_data:
        # 0: studentID; 1: questionID; 2: result (right or wrong), the index of dataset begins with 1
        proficiencyLevelLine = proficiency_level[line[0] - 1]
        qMatrixLine = q_matrix[line[1] - 1]
        discriminationValue = np.float64(discrimination[line[1] - 1])
        truth.append(line[2])
        prediction.append(tanh(interaction_function(qMatrixLine,
                                                    proficiencyLevelLine,
                                                    discriminationValue)))

    prediction = np.array(prediction)
    # address invalid value
    prediction = np.nan_to_num(prediction, nan=np.random.uniform(0, 1))

    truth = np.array(truth)
    return prediction, truth
    # student_data = np.array(student_data)
    # student_ids = student_data[:, 0] - 1
    # question_ids = student_data[:, 1] - 1
    # truth = student_data[:, 2]
    # proficiency_levels = proficiency_level[student_ids]
    # q_matrix_lines = q_matrix[question_ids]
    # discrimination_values = discrimination[question_ids]
    # predictions = interaction_function(q_matrix_lines, proficiency_levels, discrimination_values)
    # predictions = tanh(predictions)
    # predictions = np.nan_to_num(predictions, nan=np.random.uniform(0, 1))
    # return predictions, truth


def mutUniformWithPruning(individual, pset, pruning=0.5):
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


def cxSimulatedBinary(ind1, ind2, eta):
    """Executes a simulated binary crossover that modify in-place the input
    individuals. The simulated binary crossover expects :term:`sequence`
    individuals of floating point numbers.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param eta: Crowding degree of the crossover. A high eta will produce
                children resembling to their parents, while a small eta will
                produce solutions much more different.
    :returns: A tuple of two individuals.

    This function uses the numpy.random to generate random number to support
    matrix calculation. (This function is modified based on the deap library)
    """
    size = min(len(ind1), len(ind2))
    for i in range(size):
        rand = np.random.uniform(0, 1, ind1[i].shape)
        beta = np.where(rand >= 0.5, 2 * rand, 1 / 2 * (1 - rand))
        beta **= 1 / (eta + 1)
        x1 = ind1[i]
        x2 = ind2[i]
        ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
        ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))

    return ind1, ind2,


def mutGaussian(individual, mu, sigma):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.

    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :returns: A tuple of one individual.

    This function uses the numpy.random to generate random number to support
    matrix calculation. (This function is modified based on the deap library)
    """
    lr = 0.1
    size = len(individual)
    for i in range(size):
        # garanteen positive
        individual[i] = individual[i] + lr * np.random.normal(mu, sigma, individual[i].shape)

    return individual,
