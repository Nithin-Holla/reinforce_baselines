import numpy as np

import gym
import torch
import torch.nn as nn

import time
from mutils import *
from loss_utils import *
from train_utils import *


class LunarLinearPolicyNetwork(nn.Module):


	def __init__(self, hidden_dims=[128, 128, 128], num_inputs=8, num_actions=4):
		super(LunarLinearPolicyNetwork, self).__init__()

		hidden_dims = [num_inputs] + hidden_dims
		layers = []
		for i in range(len(hidden_dims)-1):
			layers += self._block(hidden_dims[i], hidden_dims[i+1])
		layers += [nn.Linear(hidden_dims[-1], num_actions), nn.LogSoftmax(dim=-1)]

		self.layers = nn.Sequential(*layers)
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
		x = self.layers(x)
		return x


if __name__ == '__main__':
	set_seed(seed=42)
	env = gym.make("LunarLander-v2")
	# env = gym.make("CartPole-v1")
	env.seed(42)
	model = LunarLinearPolicyNetwork(num_inputs=8, num_actions=4)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-7)
	# train(env=env, model=model, optimizer=optimizer, num_episodes=1000, loss_fun=compute_reinforce_loss, discount_factor=0.99)
	# train(env=env, model=model, optimizer=optimizer, num_episodes=1000, loss_fun=compute_reinforce_with_baseline_loss, discount_factor=0.99) 
	train(env=env, model=model, optimizer=optimizer, num_episodes=1000, loss_fun=compute_reinforce_with_baseline_fork_update_loss, discount_factor=0.99, print_freq=1,
			run_eps = lambda : run_episode_logN_copy_env(env, model, select_action=select_action_default,
								greedy_actions=False, render=False, beams_num=1, beam_start_freq=1,
								log_basis=4, beams_greedy=False, discount_factor=0.99))

	# set_seed(seed=42)
	# eps_1 = run_episode_logN(env, model, select_action=select_action_default,
	# 							greedy_actions=False, render=False, beams_num=1, beam_start_freq=2,
	# 							log_basis=2, beams_greedy=False, discount_factor=0.99)
	# set_seed(seed=42)
	# eps_2 = run_episode_logN_copy_env(env, model, select_action=select_action_default,
	# 							greedy_actions=False, render=False, beams_num=1, beam_start_freq=1,
	# 							log_basis=4, beams_greedy=False, discount_factor=0.99)
	# # print("Eps 1", eps_1[-4])
	# print("Eps 2", eps_2[-4])




