import copy

from model import PrefNet
from dataset import pref_dataset, utility_dataset
from itertools import cycle
from utils import *
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import al_acquisition
import bo_acquisition
import torchbnn as bnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def solver_nn(train0, query0, test, query_bo0, n_acq_al, n_acq_bo, al_acq, bo_acq):
    """
    function to combine active learning with Bayesian optimization
    """
    train = copy.deepcopy(train0)
    query = copy.deepcopy(query0)
    query_bo = copy.deepcopy(query_bo0)

    if al_acq is None:
        model = PrefNet(train['x_duels'][0][0].size).to(device)
        min_list, model, train_x_bo, train_y_bo = bo_nn(model, query_bo, n_acq_bo, bo_acq)
    else:
        model, train_al = apl_nn(train, query, test, n_acq_al, al_acq)
        model.fc_bo.load_state_dict(model.fc_expert.state_dict())
        min_list, model, train_x_bo, train_y_bo = bo_nn(model, query_bo, n_acq_bo, bo_acq, train_al=train_al)

    return min_list, model


def apl_nn(train, query, test, n_acq_al, al_acq):
    print("Start active learning with preference data")
    model = update_nn_pref(train['x_duels'], train['pref'], model=None)

    al_function = al_acquisition.choose_criterion(al_acq)

    for i in range(n_acq_al):
        query_index = al_function(model, train, query)

        train['x_duels'] = np.vstack((train['x_duels'], query['x_duels'][[query_index], :]))
        train['pref'] = np.hstack((train['pref'], query['pref'][query_index]))

        query['x_duels'] = np.delete(query['x_duels'], query_index, axis=0)
        query['pref'] = np.delete(query['pref'], query_index)

        model = update_nn_pref(train['x_duels'], train['pref'], model=model)

        print("{} query of preferential active learning".format(i+1))

        compute_nn_acc(model, test)

    return model, train


def compute_nn_acc(model, test):
    """
    compute the preference accuracy for neural network
    :param model: current model
    :param test: test set
    :return: prediction accuracy
    """
    model.eval()
    x_test = torch.tensor(test['x_duels'], dtype=torch.float)
    pref_test = test['pref']
    n_test = x_test.shape[0]
    acc = 0
    n_mc = 2
    for i in range(n_test):
        x1 = x_test[i][0]
        x2 = x_test[i][1]
        pref = pref_test[i]
        out = torch.zeros((n_mc, 2))
        for n in range(n_mc):
            out[n, 0], out[n, 1] = model(x1, x2)
        pred = torch.mean(out, dim=0)
        out1 = pred[0]
        out2 = pred[1]
        if pref == 1 and out1 < out2:
            acc += 1
        if pref == 0 and out1 > out2:
            acc += 1
    acc = acc / n_test
    print("Accuracy of expert model", acc)
    return acc


def update_nn_pref(x_duels, pref, model=None):
    x_duels = torch.tensor(x_duels, dtype=torch.float)
    pref = torch.tensor(pref, dtype=torch.long)

    pref_set = pref_dataset(x_duels, pref)
    pref_train_loader = DataLoader(pref_set, batch_size=10, shuffle=True, drop_last=False)

    pref_net = PrefNet(x_duels[0][0].shape[0]).to(device) if model is None else model

    criterion = torch.nn.NLLLoss()
    kl_criterion = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    optimizer = torch.optim.Adam(pref_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.001, T_max=20)
    pref_net.train()
    for epoch in range(100):
        nll_losses = 0
        kl_losses = 0
        # train with preference pairs
        for idx, data in enumerate(pref_train_loader):
            x1 = data['x1'].to(device)
            x2 = data['x2'].to(device)

            pref = data['pref'].to(device)

            optimizer.zero_grad()

            output1, output2 = pref_net(x1, x2)

            output = F.log_softmax(torch.hstack((output1, output2)), dim=1)

            # loss = criterion(output, pref) + 0.1 * kl_loss(pref_net)
            nll_loss = criterion(output, pref)
            kl_loss = 0.1 * kl_criterion(pref_net)
            total_loss = nll_loss + kl_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            nll_losses += nll_loss.item()
            kl_losses += kl_loss.item()

    return pref_net


def bo_nn(model, query, n_acq_bo, bo_acq, **kwargs):
    test = query.copy()
    min_list = np.zeros(n_acq_bo, )
    bo_function = bo_acquisition.choose_criterion(bo_acq)
    y_best = 1000

    print("Start Bayesian optimization with utility function")
    start_index = np.random.choice(len(query['x']), 1, replace=False)
    # start_index = bo_function(query, model, y_best)
    train_x = query['x'][start_index]
    train_y = query['y'][start_index]

    query['x'] = np.delete(query['x'], start_index, axis=0)
    query['y'] = np.delete(query['y'], start_index)

    if "train_al" in kwargs.keys():
        model = update_nn_multi(train_x, train_y, kwargs['train_al']['x_duels'], kwargs['train_al']['pref'], model)
    else:
        model = update_nn_reg(train_x, train_y, model, force_first_round=True)

    for i in range(n_acq_bo):
        model_0 = copy.deepcopy(model)
        query_index = bo_function(query, model_0, y_best, device)

        train_x = np.vstack((train_x, query['x'][[query_index], :]))
        train_y = np.hstack((train_y, query['y'][query_index]))

        query['x'] = np.delete(query['x'], query_index, axis=0)
        query['y'] = np.delete(query['y'], query_index)

        if "train_al" in kwargs.keys():
            model = update_nn_multi(train_x, train_y, kwargs['train_al']['x_duels'], kwargs['train_al']['pref'], model)
        else:
            model = update_nn_reg(train_x, train_y, model, force_first_round=False)

        pred_best = find_min_nn(model_0, test)
        y_best = pred_best if pred_best < y_best else y_best

        # y_best_simple = min(train_y)
        min_list[i] = y_best
        print("{} query of Bayesian optimization, min value {}".format(i + 1, min_list[i]))

    return min_list, model, train_x, train_y


def update_nn_reg(x, y, model, force_first_round=True):
    n_epoch = 500 if force_first_round is True else 100
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    inducing_set = utility_dataset(x, y)
    inducing_train_loader = DataLoader(inducing_set, batch_size=10, shuffle=True, drop_last=False)

    lr = 0.01 if force_first_round is True else 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = torch.nn.MSELoss()
    kl_criterion = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    model.train()
    for epoch in range(n_epoch):
        mse_losses = 0
        kl_losses = 0
        for idx, data in enumerate(inducing_train_loader):
            x = data['x'].to(device)
            y = data['y'].to(device)

            optimizer.zero_grad()

            pred = model.forward_bo(x)
            pred = pred.flatten()
            # print("pred", pred)
            mse_loss = criterion(pred, y)
            kl_loss = 0.1 * kl_criterion(model)

            total_loss = mse_loss + kl_loss
            total_loss.backward()
            optimizer.step()

            mse_losses += mse_loss.item()
            kl_losses += kl_loss.item()
    return model


def update_nn_multi(x, y, x_duels, y_pref, model):
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    inducing_set = utility_dataset(x, y)
    inducing_train_loader = DataLoader(inducing_set, batch_size=5, shuffle=True, drop_last=False)

    x_duels = torch.tensor(x_duels, dtype=torch.float)
    y_pref = torch.tensor(y_pref, dtype=torch.long)
    pref_set = pref_dataset(x_duels, y_pref)
    pref_train_loader = DataLoader(pref_set, batch_size=10, shuffle=True, drop_last=False)

    expert_criterion = torch.nn.NLLLoss()
    inducing_criterion = torch.nn.MSELoss()
    kl_criterion = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(100):
        inducing_losses = 0
        expert_losses = 0
        kl_losses = 0
        for data1, data2 in zip(pref_train_loader, cycle(inducing_train_loader)):
            optimizer.zero_grad()
            x1 = data1['x1']
            x2 = data1['x2']

            pref = data1['pref']
            x1, x2, pref = x1.to(device), x2.to(device), pref.to(device)
            output1, output2 = model(x1, x2)
            output = F.log_softmax(torch.hstack((output1, output2)), dim=1)
            expert_loss = expert_criterion(output, pref)

            inducing_x = data2['x'].to(device)
            inducing_y = data2['y'].to(device)
            pred = model.forward_bo(inducing_x)
            pred = pred.flatten()
            inducing_loss = inducing_criterion(pred, inducing_y)

            kl_loss = kl_criterion(model)

            total_loss = inducing_loss + 0.1 * expert_loss + kl_loss

            total_loss.backward()
            optimizer.step()

            inducing_losses += inducing_loss.item()
            expert_losses += expert_loss.item()
            kl_losses += kl_loss.item()

    return model


def find_min_nn(model, test):
    model.eval()
    x = test['x']
    y = test['y']

    query_x = torch.tensor(x, dtype=torch.float).to(device)
    pred = model.forward_bo(query_x)

    min_value = y[torch.argmin(pred.cpu())]

    return min_value