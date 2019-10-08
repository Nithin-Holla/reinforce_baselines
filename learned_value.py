from torch import nn

from mutils import *


class LearnedValueNetwork(nn.Module):

	def __init__(self, hidden_dims=[128, 128, 128], num_inputs=8, num_actions=4):
		super(LearnedValueNetwork, self).__init__()

		hidden_dims = [num_inputs] + hidden_dims
		shared_layers = []
		for i in range(len(hidden_dims) - 1):
			shared_layers += self._block(hidden_dims[i], hidden_dims[i + 1])

		self.shared_layers = nn.Sequential(*shared_layers)
		self.policy_layer = nn.Sequential(nn.Linear(hidden_dims[-1], num_actions),
										  nn.LogSoftmax(dim=-1))
		self.value_layer = nn.Sequential(nn.Linear(hidden_dims[-1], 1), nn.Tanh())
		self.to(get_device())

	def _block(self, c_in, c_out):
		return [
			nn.Linear(c_in, c_out),
			nn.ELU(),
			# nn.ReLU()
			nn.LayerNorm(c_out)
		]

	def forward(self, x):
		x = x.to(get_device())
		x = self.shared_layers(x)
		log_prob = self.policy_layer(x)
		value = self.value_layer(x)
		return log_prob, value







