import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
from torch.nn import Parameter
from torch.distributions import Beta, Normal, LogNormal
import math
from networks.power_spherical import PowerSpherical
from networks.von_mises_fisher import VonMisesFisherReparametrizedSample
import numpy as np
from numbers import Number
from utils import ml_kappa, softplus_inv, _calculate_fan_in_and_fan_out


class FakeLinear(nn.Module):
	"""docstring for RadDirLinear"""
	def __init__(self, in_features, out_features, bias=True, eps=0.1):
		super(FakeLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.dir_loc = nn.Parameter(torch.Tensor(out_features, in_features))
		concentration_init = ml_kappa(dim=in_features, eps=eps)
		print('concentration init', concentration_init)
		self.dir_softplus_inv_concentration = nn.Parameter(
			torch.Tensor(out_features).uniform_(softplus_inv(concentration_init), softplus_inv(concentration_init))
		)
		nn.init.kaiming_normal_(self.dir_loc)
		self.dir_loc.data /= torch.sum(self.dir_loc.data ** 2, dim=-1, keepdim=True) ** 0.5

		self.bias = nn.Parameter(torch.Tensor(out_features).uniform_(0,0)) if bias else None

		self.rad_layer = nn.Linear(in_features, out_features)
		self.gate = nn.Softplus()


	def forward(self, input, sample=False):

		if sample:
			direction_sample = PowerSpherical(self.dir_loc, softplus(self.dir_softplus_inv_concentration)).rsample()
		else:
			direction_sample = PowerSpherical(self.dir_loc, softplus(self.dir_softplus_inv_concentration)).mean

		radius = self.gate(self.rad_layer(input))
			
		weight = direction_sample.unsqueeze(0) * radius.unsqueeze(-1)
		if self.bias is not None:
			output = (input.unsqueeze(1) * weight).sum(-1) + self.bias
		else:
			output = (input.unsqueeze(1) * weight).sum(-1)
		return output

	def parameter_adjustment(self):
		self.dir_loc.data /= torch.sum(self.dir_loc.data ** 2, dim=-1, keepdim=True) ** 0.5
		# self.softplus_inv_concentration.data = self.softplus_inv_concentration.data.clamp(max=self.softplus_inv_concentration_upper_bound)

	def kl_divergence(self):
		pass



class Model(nn.Module):
	def __init__(self, eps=0.1):
		super(Model, self).__init__()

		self.eps = eps
		self.fc1 = FakeLinear(in_features=784, out_features=400, bias=True, eps=eps)
		self.fc2 = FakeLinear(in_features=400, out_features=400, bias=True, eps=eps)
		self.fc3 = FakeLinear(in_features=400, out_features=10, bias=True, eps=eps)
		self.nonlinear = nn.ReLU()
		self.name = 'fake_hat_eps_{}'.format(eps)

	def forward(self, input, sample=False):
		x = input.view(input.size(0), -1)
		x = self.nonlinear(self.fc1(x, sample))
		x = self.nonlinear(self.fc2(x, sample))
		x = self.fc3(x, sample)
		return x

	def kl_divergence(self):
		pass


		
