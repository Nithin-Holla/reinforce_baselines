import numpy as np

import gym
import gym_maze
from torch import nn
import torch
import time

import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
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

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1)
        x = self.linear_layers(x)
        return x

def normalize_input(maze_state):
    mean = np.mean(maze_state, axis=(0, 1), keepdims=True)
    std = np.std(maze_state, axis=(0, 1), keepdims=True)
    normalized_maze_state = (maze_state - mean) / std
    return normalized_maze_state


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


def select_action(model, state):
    state = torch.from_numpy(state).float().unsqueeze(0) # Convert state from ndarray to tensor and add a fake batch dimension
    state = torch.transpose(state, 1, 3) # Convolution needs channels as dim 2, so transpose it
    log_p = model(state).squeeze(0)
    action = torch.multinomial(torch.exp(log_p), 1)
    return ["N","S","W","E"][action.item()], log_p[action]

def render_episode(env, model):
    env.reset()
    env.render()
    s = generate_model_input(env, normalize=False)
    episode = []
    done = False
    while not done:
        with torch.no_grad():
            action, log_p = select_action(model, s)
        _, r, done, _ = env.step(action)
        env.render()
        time.sleep(0.05)
        s_next = generate_model_input(env)
        episode += [(s, log_p, r, done)]
        s = s_next
    env.close()

def run_episode(env, model, render=False):
    env.reset()
    if render:
        env.render()
    s = generate_model_input(env)
    episode = []
    done = False
    while not done:
        action, log_p = select_action(model, s)
        _, r, done, _ = env.step(action)
        if render:
            env.render()
            time.sleep(0.05)
        s_next = generate_model_input(env)
        episode += [(s, log_p, r, done)]
        s = s_next
    if render:
        env.close()
    return episode

def compute_reinforce_loss(episode, discount_factor):
    log_ps = torch.zeros(len(episode))
    Gs  = []

    G = 0
    loss = 0

    for i, (s, log_p, r, done) in enumerate(reversed(episode)):
        t = len(episode) - 1 - i
        G = discount_factor * G + r
        Gs += [G]
        log_ps[i] = log_p

    Gs = torch.tensor(Gs)
    Gs = (Gs - torch.mean(Gs)) / torch.std(Gs)
    loss = - torch.sum(Gs * log_ps)
    return loss

def train(model, env_name, num_episodes, optimizer, discount_factor, random_mazes=False):
    episode_durations = []
    env = gym.make(env_name)
    for i in range(num_episodes):
        if random_mazes:
            env = gym.make(env_name)
        episode = run_episode(env, model)
        episode_durations.append(len(episode))
        loss = compute_reinforce_loss(episode, discount_factor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Loss: ', loss.item())
        if i % 1 == 0:
            print('Number of steps taken to solve the maze: ', len(episode))

    plt.plot(episode_durations)
    plt.show()
    render_episode(env, model)

if __name__ == '__main__':
    model = PolicyNetwork(maze_size=5)
    learn_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)
    # maze-sample-5x5-v0 for a fixed maze
    train(model, "maze-random-5x5-v0", 500, optimizer, 1, random_mazes=False)



