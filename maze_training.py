import numpy as np

import gym
import gym_maze
import torch
import torch.nn as nn

import time
from mutils import *


class MazePolicyNetwork(nn.Module):
	def __init__(self,  maze_size):
		super(PolicyNetwork, self).__init__()
		self.conv_layers = nn.Sequential(
			nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
			nn.ReLU()
			)

		# Maze size after the convolution layers
		reduced_maze_size = int((maze_size + 2 - 3) / 2) + 1

		self.linear_layers = nn.Sequential(
			nn.Linear(reduced_maze_size * reduced_maze_size * 32, 32),
			nn.ReLU(),
			nn.Linear(32, 4),
			nn.LogSoftmax(dim=0)
			)
		self.to(get_device())

	def forward(self, x):
		x = x.to(get_device())
		x = self.conv_layers(x)
		x = x.view(-1)
		x = self.linear_layers(x)
		return x

def normalize_input(maze_state):
	mean = np.mean(maze_state, axis=(0, 1), keepdims=True)
	std = np.std(maze_state, axis=(0, 1), keepdims=True)
	normalized_maze_state = (maze_state - mean) / std
	return normalized_maze_state

def select_action(model, state):
	state = torch.from_numpy(state).float().unsqueeze(0) # Convert state from ndarray to tensor and add a fake batch dimension
	state = torch.transpose(state, 1, 3) # Convolution needs channels as dim 2, so transpose it
	log_p = model(state).squeeze(0)
	action = torch.multinomial(torch.exp(log_p), 1)
	return ["N","S","W","E"][action.item()], log_p[action]

def generate_model_input(env, normalize=False):
	maze_cells = env.unwrapped.maze_view.maze.maze_cells
	expanded_maze_cells = np.expand_dims(maze_cells, 2)

	binarized_maze_cells = np.unpackbits(expanded_maze_cells.astype('uint8'), axis=2)
	coordinates = env.unwrapped.state.astype('int')
	binarized_maze_cells[coordinates[0], coordinates[1], 3] = 1
	binarized_maze_cells[binarized_maze_cells.shape[0] - 1, binarized_maze_cells.shape[1] - 1, 3] = -1
	binarized_maze_cells = binarized_maze_cells[:,:,3:8].astype('float')
	if normalize:
		normalized_maze_cells = normalize_input()
		return normalized_maze_cells
	else:
		return binarized_maze_cells