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

class RDLinear(nn.Module):
	"""docstring for RadDirLinear"""
	def __init__(self, in_features, out_features, bias=True, ratio=0.5, eps=0.1):
		super(RDLinear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.dir_loc = nn.Parameter(torch.Tensor(out_features, in_features))
		concentration_init = ml_kappa(dim=in_features, eps=eps)
		print('concentration init', concentration_init)
		self.dir_softplus_inv_concentration = nn.Parameter(torch.Tensor(out_features).uniform_(softplus_inv(concentration_init), softplus_inv(concentration_init)))
		# self.dir_concentration = softplus(ml_kappa(dim=in_features, eps=eps) * torch.ones(out_features).cuda())
		nn.init.kaiming_normal_(self.dir_loc)
		self.dir_loc.data /= torch.sum(self.dir_loc.data ** 2, dim=-1, keepdim=True) ** 0.5
		# nn.init.normal_(self.dir_concentration, ml_kappa(dim=in_features, eps=0.1), 0.0001)
		# self.dir_rsampler = VonMisesFisherReparametrizedSample(out_features, in_features, eps)

		total_var = 2 / in_features
		noise_var = total_var * ratio
		mu_var = total_var - noise_var
        
		noise_std, mu_std = math.sqrt(noise_var), math.sqrt(mu_var)
		bound = math.sqrt(3.0) * mu_std
		rho_init = np.log(np.exp(noise_std)-1)

		self.rad_mu = nn.Parameter(torch.Tensor(out_features, 1))
		print('sigma init', noise_std, 'mu init', bound)
		# self.rad_rho = nn.Parameter(torch.Tensor(out_features))
		self.rad_rho = nn.Parameter(torch.Tensor(out_features, 1).uniform_(rho_init,rho_init))
		nn.init.uniform_(self.rad_mu, -bound, bound)
		# nn.init.normal_(self.rad_mu, math.log(2), 0.0001)
		# nn.init.normal_(self.rad_rho, np.log(np.exp(0.0001)-1), 0.0001)

		self.bias = nn.Parameter(torch.Tensor(out_features).uniform_(0,0)) if bias else None


	def forward(self, input, sample=False):
		# self.dir_loc.data /= torch.sum(self.dir_loc.data ** 2, dim=-1, keepdim=True) ** 0.5
		# direction_sample = self.dir_rsampler(1, sample)[0]
		if sample:
			direction_sample = PowerSpherical(self.dir_loc, softplus(self.dir_softplus_inv_concentration)).rsample()
			radius_sample = LogNormal(self.rad_mu, softplus(self.rad_rho)).rsample()
		else:
			direction_sample = PowerSpherical(self.dir_loc, softplus(self.dir_softplus_inv_concentration)).mean
			radius_sample = LogNormal(self.rad_mu, softplus(self.rad_rho)).mean
			
		weight = direction_sample * radius_sample#.unsqueeze(-1)
		return F.linear(input, weight, self.bias)

	def parameter_adjustment(self):
		self.dir_loc.data /= torch.sum(self.dir_loc.data ** 2, dim=-1, keepdim=True) ** 0.5
		# self.softplus_inv_concentration.data = self.softplus_inv_concentration.data.clamp(max=self.softplus_inv_concentration_upper_bound)

	def kl_divergence(self):
		pass



class Model(nn.Module):
	def __init__(self, ratio=0.5, eps=0.1):
		super(Model, self).__init__()

		self.eps = eps
		self.ratio = ratio
		self.fc1 = RDLinear(in_features=784, out_features=400, bias=True, ratio=ratio, eps=eps)
		self.fc2 = RDLinear(in_features=400, out_features=400, bias=True, ratio=ratio, eps=eps)
		self.fc3 = RDLinear(in_features=400, out_features=10, bias=True, ratio=ratio, eps=eps)
		self.nonlinear = nn.ReLU()
		self.name = 'dir_ucl_ratio_{}_eps{}'.format(ratio, eps)

	def forward(self, input, sample=False):
		x = input.view(input.size(0), -1)
		x = self.nonlinear(self.fc1(x, sample))
		x = self.nonlinear(self.fc2(x, sample))
		x = self.fc3(x, sample)
		return x

	def kl_divergence(self):
		pass


		
