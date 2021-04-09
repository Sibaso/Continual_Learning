import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli, LogNormal

class Model(nn.Module):
	"""docstring for Net"""
	def __init__(self):
		super(Model, self).__init__()
		self.fc1 = nn.Linear(28*28, 400)
		# nn.init.uniform_(self.fc1.bias, -0.7, -0.7)
		# nn.init.normal_(self.fc1.bias, -100, 99.3)
		self.fc2 = nn.Linear(400, 400)
		# nn.init.uniform_(self.fc2.bias, -0.7, -0.7)
		# nn.init.normal_(self.fc2.bias, -100, 99.3)
		self.fc3 = nn.Linear(400, 10)
		self.activate = nn.ReLU()
		self.name = 'mlp'
		ln = LogNormal(0, 1)
		self.mask1 = nn.Parameter(ln.sample(torch.Size([400]))-1.5)
		# nn.init.uniform_(self.mask1, -1, 0)
		self.mask2 = nn.Parameter(ln.sample(torch.Size([400]))-1.5)
		# nn.init.uniform_(self.mask2, -1, 0)
		# be = Bernoulli(torch.ones(1, 400)*0.1)
		# self.mask1 = be.sample().cuda()
		# self.mask2 = be.sample().cuda()


	def forward(self, x, s):
		x = x.view(x.size(0),-1)
		x = self.fc1(x)
		x = self.activate(x)*torch.sigmoid(s*self.mask1).unsqueeze(0)
		# s, i = torch.sort(x, descending=True)
		# x =  x * (x > s[:,5].unsqueeze(-1))
		x = self.fc2(x)
		x = self.activate(x)*torch.sigmoid(s*self.mask2).unsqueeze(0)
		# s, i = torch.sort(x, descending=True)
		# x =  x * (x > s[:,5].unsqueeze(-1))
		x = self.fc3(x)
		return x

class LinearSparse(nn.Module):
	"""docstring for LinearSparse"""
	def __init__(self, in_features, out_features, bias=True):
		super(LinearSparse, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.ratio = 1
		self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
		self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
		k = math.sqrt(1 / in_features)
		nn.init.uniform_(self.weight, -k, k)
		nn.init.uniform_(self.bias, -k, k)
		# self.mask = nn.Parameter(torch.Tensor(out_features, 1))
		# nn.init.sparse_(self.mask, 0.5)
		# self.mask_layer = nn.Linear(in_features, int(out_features*self.ratio))
		# nn.init.uniform_(self.mask_layer.weight, -k, k)
		# nn.init.uniform_(self.mask_layer.bias, -k, k)
	def forward(self, x):
		# mask = F.relu(self.mask_layer(x))
		# mask = torch.cat([mask, torch.zeros(mask.size(0), self.out_features-int(self.out_features*self.ratio)).cuda()], dim=-1)
		weight = self.weight.unsqueeze(0) * mask.unsqueeze(-1)
		if self.bias is not None:
			output = (x.unsqueeze(1) * weight).sum(-1) + self.bias
		else:
			output = (x.unsqueeze(1) * weight).sum(-1)

		return output
		# return F.linear(x, self.weight*F.relu(self.mask), self.bias)

class ModelSparse(nn.Module):
	"""docstring for Net"""
	def __init__(self):
		super(ModelSparse, self).__init__()
		self.fc1 = LinearSparse(28*28, 400)
		self.fc2 = LinearSparse(400, 400)
		self.fc3 = LinearSparse(400, 10)
		self.activate = nn.ReLU()
		self.name = 'model_sparse'


	def forward(self, x):
		x = x.view(x.size(0),-1)
		x = self.activate(self.fc1(x))
		x = self.activate(self.fc2(x))
		x = self.fc3(x)
		return x