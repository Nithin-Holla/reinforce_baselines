import numpy as np

import gym
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import time

import matplotlib.pyplot as plt
from loss_utils import *
from mutils import *


def select_action_default(model, state):
	state = torch.from_numpy(state).float().unsqueeze(dim=0) # Convert state from ndarray to tensor and add a fake batch dimension
	log_p = model(state).squeeze(dim=0)
	action = torch.multinomial(input=torch.exp(log_p), num_samples=1).item()
	return action, log_p[action]


def render_episode(env, model, select_action=select_action_default):
	s = env.reset()
	env.render()
	episode = []
	done = False
	while not done:
		with torch.no_grad():
			action, log_p = select_action(model, s)
		s_next, r, done, _ = env.step(action)
		env.render()
		time.sleep(0.05)
		episode += [(s, log_p, r, done)]
		s = s_next
	env.close()


def run_episode(env, model, select_action=select_action_default, loss_fun=compute_reinforce_loss, render=False):
	s = env.reset()
	if render:
		env.render()
	episode = []
	done = False
	while not done:
		action, log_p = select_action(model, s)
		s_next, r, done, _ = env.step(action)
		if render:
			env.render()
			time.sleep(0.05)
		episode += [{"state": s, "action": action, "log_p": log_p, "reward": r, "done" : done}]
		s = s_next
	if render:
		env.close()
	return episode


def train(model, env, num_episodes, optimizer, discount_factor):
	episode_durations = []
	model = model.to(get_device())
	metric_avg = np.zeros((4,2), dtype=np.float32)
	actions_taken = []
	for i in range(num_episodes):
		episode = run_episode(env, model)
		episode_durations.append(len(episode))
		loss = compute_reinforce_loss(episode, discount_factor)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		metric_avg[0,0] += loss.item()
		metric_avg[1,0] += len(episode)
		metric_avg[2,0] += sum([e["reward"] for e in episode])
		metric_avg[3,0] += episode[-1]["reward"]
		metric_avg[:,1] += 1
		actions_taken += [e["action"] for e in episode]
		if (i+1) % 10 == 0:
			metric_avg[:,0] = metric_avg[:,0] / metric_avg[:,1]
			print("Iteration %i: Loss=%4.2f, Episode length=%4.2f, Sum rewards=%4.2f, Final reward=%4.2f" % (i+1, metric_avg[0,0], metric_avg[1,0], metric_avg[2,0], metric_avg[3,0]))
			print("Action distribution: " + ", ".join(["%s: %4.2f%%" % (str(a), 100.0*sum([(a == at) for at in actions_taken])/len(actions_taken)) for a in set(actions_taken)]))
			metric_avg[:,:] = 0

	render_episode(env, model)



