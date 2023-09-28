import torch
from utils import *


def choose_criterion(criterion):
    if criterion == "random":
        return random_sampling
    elif criterion == "BALD":
        return bald_nn
    elif criterion == "uncertainty":
        return uncertainty_nn


def bald_nn(model, train, query):
    model.eval()
    x_query = torch.tensor(query['x_duels'], dtype=torch.float)
    iterations = 10
    x1 = x_query[:, 0, :]
    x2 = x_query[:, 1, :]

    score_all = np.zeros(shape=(x1.shape[0], 2))
    all_entropy = np.zeros(shape=x1.shape[0])
    for t in range(iterations):
        out1, out2 = model(x1, x2)
        diff = out2 - out1
        prob_1 = logistic_function(diff)
        prob_0 = 1 - prob_1

        score = torch.cat((prob_0, prob_1), 1)
        score = score.detach().numpy()

        score_all += score
        score_log = np.log2(score)

        entropy_compute = - np.multiply(score, score_log)
        entropy_per_iter = np.sum(entropy_compute, axis=1)

        all_entropy += entropy_per_iter
    # score_all += np.finfo(float).eps
    Avg_Pi = np.divide(score_all, iterations)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

    G_X = Entropy_Average_Pi

    F_X = np.divide(all_entropy, iterations)

    U_X = (G_X - F_X).flatten()

    return np.argmax(U_X)


def random_sampling(model, train, query):
    n = len(query['pref'])
    return np.random.randint(0, n)


def uncertainty_nn(model, train, query):
    model.eval()
    x_query = query['x_duels']

    n_mc = 5
    confidence = torch.zeros((n_mc, len(query['pref'])))
    x1 = torch.tensor(x_query[:, 0, :])
    x2 = torch.tensor(x_query[:, 1, :])
    for t in range(n_mc):
        out1, out2 = model(x1, x2)
        diff = torch.abs(out1 - out2).reshape(-1)
        v = logistic_function(diff)
        confidence[t, :] = v
    confidence = torch.mean(confidence, dim=0)
    return torch.argmin(confidence)

