import numpy as np
from sklearn import metrics


def accuracy(y_pred, y_true, threshold=0.5, weights=None):
    pred = np.array(y_pred)
    true = np.array(y_true)
    result = np.where(pred > threshold, 1, 0)
    if weights is not None:
        correct = np.sum((true == result) * weights)
        total = np.sum(weights)
        return correct / total
    else:
        return metrics.accuracy_score(true, result)


def area_under_curve(y_pred, y_true):
    pred = np.array(y_pred)
    true = np.array(y_true)
    fpr, tpr, thresholds = metrics.roc_curve(true, pred)
    return metrics.auc(fpr, tpr)


def f1_score(y_pred, y_true, threshold=0.5):
    pred = np.array(y_pred)
    true = np.array(y_true)
    result = np.where(pred >= threshold, 1, 0)
    return metrics.f1_score(true, result)


def loss(y_pred, y_true):
    pred = np.array(y_pred)
    true = np.array(y_true)
    losses = np.abs(pred - true)
    losses /= np.max(losses)
    return losses


def degree_of_agreement(q_matrix, proficiency, dataset):
    problem_number, knowledge_number = q_matrix.shape
    student_number = proficiency.shape[0]
    r_matrix = np.full((student_number, problem_number), -1)
    for lines in dataset:
        student_id_batch, question_batch, _, y_batch = lines
        for student_id, question, y in zip(student_id_batch, question_batch, y_batch ):
            r_matrix[student_id][question] = y
    doaList = []
    for k in range(knowledge_number):
        numerator = 0.0
        denominator = 0.0
        delta_matrix = proficiency[:, k].reshape(-1, 1) > proficiency[:, k].reshape(1, -1)
        question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
        for j in question_hask:
            # avoid blank logs
            row_vec = (r_matrix[:, j].reshape(1, -1) != -1).astype(int)
            column_vec = (r_matrix[:, j].reshape(-1, 1) != -1).astype(int)
            mask = row_vec * column_vec
            delta_response_logs = r_matrix[:, j].reshape(-1, 1) > r_matrix[:, j].reshape(1, -1)
            i_matrix = r_matrix[:, j].reshape(-1, 1) != r_matrix[:, j].reshape(1, -1)
            numerator += np.sum(delta_matrix * np.logical_and(mask, delta_response_logs))
            denominator += np.sum(delta_matrix * np.logical_and(mask, i_matrix))
        doaList.append(numerator / denominator)

    return np.mean(doaList)
