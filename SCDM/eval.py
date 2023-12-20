import numpy as np
from sklearn import metrics
from .utility import StudentDataSet


def accuracy(prediction: np.ndarray, truth: np.ndarray, threshold=0.5, weights=None):
    result = np.where(prediction > threshold, 1, 0)
    if weights is not None:
        correct = np.sum((truth == result) * weights)
        total = np.sum(weights)
        return correct / total
    else:
        return np.mean(truth == result)


def root_mean_squared_error(prediction: np.ndarray, truth: np.ndarray, weights=None):
    if weights is not None:
        return np.sqrt(np.sum(weights * np.square(truth - prediction)) / np.sum(weights))
    else:
        return np.sqrt(np.mean(np.square(truth - prediction)))


def area_under_curve(prediction: np.ndarray, truth: np.ndarray):
    fpr, tpr, thresholds = metrics.roc_curve(truth, prediction)
    return metrics.auc(fpr, tpr)


def f1_score(prediction: np.ndarray, truth: np.ndarray, threshold=0.5):
    result = np.where(prediction >= threshold, 1, 0)
    return metrics.f1_score(truth, result)


def loss(prediction: np.ndarray, truth: np.ndarray):
    losses = np.abs(prediction - truth)
    losses /= np.max(losses)
    return losses


def complexity(individual, argument_number):
    expression = str(individual)
    count = 0
    if expression.find("q_matrix") != -1:
        count += 1

    if expression.find("proficiency_level") != -1:
        count += 1

    if expression.find("discrimination") != -1:
        count += 1

    return count / argument_number


def degree_of_agreement(qMatrix: np.ndarray,
                        proficiencyLevel: np.ndarray,
                        studentData: StudentDataSet):
    problemNumber, knowledgeNumber = qMatrix.shape
    studentNumber = proficiencyLevel.shape[0]
    rMatrix = np.full((studentNumber, problemNumber), -1)
    for line in studentData:
        rMatrix[line[0] - 1][line[1] - 1] = line[2]
    doaList = []
    for k in range(knowledgeNumber):
        numerator = 0.0
        denominator = 0.0
        deltaMatrix = proficiencyLevel[:, k].reshape(-1, 1) > proficiencyLevel[:, k].reshape(1, -1)
        questionHask = np.where(qMatrix[:, k] != 0)[0].tolist()
        for j in questionHask:
            # avoid blank logs
            rowVec = (rMatrix[:, j].reshape(1, -1) != -1).astype(int)
            columnVec = (rMatrix[:, j].reshape(-1, 1) != -1).astype(int)
            mask = rowVec * columnVec
            deltaResponseLogs = rMatrix[:, j].reshape(-1, 1) > rMatrix[:, j].reshape(1, -1)
            iMatrix = rMatrix[:, j].reshape(-1, 1) != rMatrix[:, j].reshape(1, -1)
            numerator += np.sum(deltaMatrix * np.logical_and(mask, deltaResponseLogs))
            denominator += np.sum(deltaMatrix * np.logical_and(mask, iMatrix))
        doaList.append(numerator / denominator)

    return np.mean(doaList)
