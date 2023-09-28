import numpy as np
import torch


def choose_criterion(criterion):
    if criterion == "random":
        return random_sampling
    elif criterion == "EI":
        return ei_mc
    elif criterion == "ucb":
        return ucb


def random_sampling(query, model, y_best):
    n = len(query['y'])
    return np.random.randint(0, n)


def ei_mc(query, model, y_best, device):
    N_mc = 50
    x = torch.tensor(query['x'], dtype=torch.float).to(device)

    score = np.zeros_like(query['y'])
    for i in range(N_mc):
        pred = model.forward_bo(x).reshape(-1)

        score += np.maximum((y_best - pred.cpu().detach().numpy()), np.zeros_like(query['y']))

    return np.argmax(score)


def ucb(query, model, y_best, device):
    N_mc = 100
    x = torch.tensor(query['x']).to(device)
    alpha = 5
    # score = np.zeros_like(query['y'])
    pred = np.zeros([N_mc, x.shape[0]])
    for i in range(N_mc):
        pred[i] = model.forward_bo(x).detach().numpy().reshape(-1)
    mean = pred.mean(axis=0)
    std = pred.std(axis=0)
    score = -mean + alpha*std
    return np.argmax(score)
