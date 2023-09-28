import torch.nn as nn
import torch
import torchbnn as bnn


class PrefNet(nn.Module):
    def __init__(self, n_input):
        super(PrefNet, self).__init__()

        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_input, out_features=100)
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=30)
        self.fc3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=30, out_features=15)

        self.fc_expert = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=15, out_features=1)
        self.fc_bo = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=15, out_features=1)

    def forward_once(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = torch.tanh(x)

        x = self.fc3(x)
        x = torch.tanh(x)

        x = self.fc_expert(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    def forward_bo(self, x):

        x = self.fc1(x)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = torch.tanh(x)

        x = self.fc3(x)
        x = torch.tanh(x)

        x = self.fc_bo(x)
        return x