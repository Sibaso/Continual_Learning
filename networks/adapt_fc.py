import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from torch.nn import Parameter
from torch.distributions import Beta, Normal, LogNormal
from networks.power_spherical import PowerSpherical
import math

class HypernetPS(nn.Module):
    """docstring for HypernetWeight"""
    def __init__(self, noise_shape, dim):
        super(HypernetPS, self).__init__()
        self.noise_shape = noise_shape
        self.dim = dim
        self.fc1 = nn.Linear(noise_shape, dim)
        # self.fc2 = nn.Linear(dim, dim)
        self.nonlinear = nn.ReLU()
        self.fc_loc = nn.Linear(dim, dim)
        self.fc_concentration = nn.Linear(dim, 1)

    def forward(self, noise):
        x = self.nonlinear(self.fc1(noise))
        # x = self.nonlinear(self.fc2(x))
        loc = self.fc_loc(x)
        loc = loc / loc.norm(dim=-1, keepdim=True)
        concentration = softplus(self.fc_concentration(x).squeeze()) + 1
        # the `+ 1` prevent collapsing behaviors
        sample = PowerSpherical(loc, concentration).rsample(torch.Size([1]))
        return sample

        

class NewLinear(nn.Module):
    """docstring for NewLinear"""
    def __init__(self, in_features, out_features, bias=True, noise_shape=1):
        super(NewLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.dir_concentration = nn.Parameter(torch.Tensor(out_features))
        self.dir_loc = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_normal_(self.dir_loc)
        nn.init.normal_(self.dir_concentration, out_features*10, 1)
        self.rad_mu = nn.Linear(in_features, in_features)
        self.rad_scale = nn.Linear(in_features, in_features)
        self.rad_mu1 = nn.Linear(in_features, in_features)
        self.rad_scale1 = nn.Linear(in_features, in_features)
        self.embed1 = nn.Linear(in_features, in_features)
        self.embed2 = nn.Linear(in_features, in_features)
        self.nonlinear = nn.ReLU()

        # self.rad_mu = nn.Parameter(torch.Tensor(in_features))
        # self.rad_scale = nn.Parameter(torch.Tensor(in_features))
        # self.rad_mu1 = nn.Parameter(torch.Tensor(in_features))
        # self.rad_scale1 = nn.Parameter(torch.Tensor(in_features))
        # nn.init.normal_(self.rad_mu, math.log(2.0), 0.0001)
        # nn.init.normal_(self.rad_scale, softplus_inv(0.0001), 0.0001)
        # nn.init.normal_(self.rad_mu1, math.log(2.0), 0.0001)
        # nn.init.normal_(self.rad_scale1, softplus_inv(0.0001), 0.0001)

        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_scale = nn.Parameter(torch.Tensor(out_features))
        nn.init.normal_(self.bias_mu, 0.0, 0.0001)
        nn.init.normal_(self.bias_scale, softplus_inv(0.0001), 0.0001)

    def forward(self, x):
        concentration = softplus(self.dir_concentration)
        loc = self.dir_loc / self.dir_loc .norm(dim=-1, keepdim=True)
        self.dir_sampler = PowerSpherical(loc, concentration)
        e = self.nonlinear(self.embed1(x))
        e = self.nonlinear(self.embed2(e))
        self.rad_sampler = LogNormal(self.rad_mu(e), softplus(self.rad_scale(e)))
        self.rad_sampler1 = LogNormal(self.rad_mu1(e), softplus(self.rad_scale1(e)))
        self.bias_sampler = Normal(self.bias_mu, softplus(self.bias_scale))

        direction_sample = self.dir_sampler.rsample()
        
        
        radius_sample = self.rad_sampler.rsample()
        radius_sample = (radius_sample * self.rad_sampler1.rsample()) ** 0.5
        radius_sample = radius_sample ** 0.5

        # radius_sample = LogNormal(self.rad_mu, softplus(self.rad_scale)).rsample()
        # radius_sample = (radius_sample * LogNormal(self.rad_mu1, softplus(self.rad_scale1)).rsample()) ** 0.5

        bias = self.bias_sampler.rsample() if self.bias else None

        # weight = direction_sample * radius_sample.unsqueeze(0) ** 0.5
        weight = direction_sample
        
        output = F.linear(x*radius_sample, weight, bias)
        
        return output

    def kl_divergence(self):
        pass


class NewFC(BayesianModel):
    def __init__(self):
        super(NewFC, self).__init__()

        self.fc1 = NewLinear(in_features=784, out_features=400, bias=True)
        self.nonlinear1 = nn.ReLU()
        self.fc2 = NewLinear(in_features=400, out_features=400, bias=True)
        self.nonlinear2 = nn.ReLU()
        self.fc3 = NewLinear(in_features=400, out_features=10, bias=True)

    def forward(self, input):
        x = input.view(input.size(0), -1)
        x = self.nonlinear1(self.fc1(x))
        x = self.nonlinear2(self.fc2(x))
        # x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x

    def kl_divergence(self):


