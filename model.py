import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Epsilon = 1e-6

class VGAE(nn.Module):
	def __init__(self, args, adj):
		super(VGAE, self).__init__()
		self.input_dim = args.input_dim
		self.adj_matrix = adj
		self.hidden1_dim = args.hidden1_dim
		self.hidden2_dim = args.hidden2_dim
		self.base_gcn = GraphConvSparse(self.input_dim, self.hidden1_dim, adj, activation=F.relu)
		self.gcn_mean = GraphConvSparse(self.hidden1_dim, self.hidden2_dim, adj, activation=lambda x: x)
		self.gcn_logstddev = GraphConvSparse(self.hidden1_dim, self.hidden2_dim, adj, activation=lambda x: x)

		assert args.num_user is not None
		assert args.num_assertion is not None
		self.num_user = args.num_user
		self.num_assertion = args.num_assertion
		self.user_nodes_mask = torch.zeros((self.num_user + self.num_assertion, self.hidden2_dim), device=args.device)
		self.user_nodes_mask[:self.num_user, :] = 1.0
		self.asser_nodes_mask = torch.zeros((self.num_user + self.num_assertion, self.hidden2_dim), device=args.device)
		self.asser_nodes_mask[self.num_user:, :] = 1.0

	def encode(self, x):
		hidden = self.base_gcn(x)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)
		gaussian_noise = torch.randn(x.size(0), self.hidden2_dim, device=x.device)
		sampled_z = F.relu(gaussian_noise * torch.exp(self.logstd) + self.mean)
		# sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
		return sampled_z

	def decode(self, z):
		# Use this for bill
		u_b_m = torch.matmul(
			 z * self.user_nodes_mask,
			(z * self.asser_nodes_mask).t()
		)
		return torch.sigmoid(u_b_m + u_b_m.t())

		# return torch.sigmoid(torch.matmul(z, z.t()))

	def forward(self, x):
		z = self.encode(x)
		return self.decode(z)

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim)
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs
		x = torch.mm(x, self.weight)
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0 / (input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
	return nn.Parameter(initial)
