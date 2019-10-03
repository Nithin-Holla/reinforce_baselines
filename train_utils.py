import numpy as np

import gym
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import time
from copy import copy, deepcopy

import matplotlib.pyplot as plt
from loss_utils import *
from mutils import *


def select_action_default(model, state, greedy=False):
	state = torch.from_numpy(state).float().unsqueeze(dim=0) # Convert state from ndarray to tensor and add a fake batch dimension
	log_p = model(state).squeeze(dim=0)
	if not greedy:
		action = torch.multinomial(input=torch.exp(log_p), num_samples=1).item()
	else:
		action = log_p.argmax().item()
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


def run_episode(env, model, select_action=select_action_default, 
				greedy_actions=False, render=False, 
				beams_num=-1, beam_freq=10, beams_greedy=False,
				discount_factor=0.99, last_state=None):

	call_beam = lambda state : run_episode(deepcopy(env), model, select_action, 
										   greedy_actions=beams_greedy, render=False, 
										   beams_num=-1, beam_freq=-1, beams_greedy=False, 
										   discount_factor=discount_factor, last_state=state)

	if last_state is None:
		s = env.reset()
	else:
		s = last_state
	if render:
		env.render()
	episode = []
	done = False
	frame_iter = 0
	beams_baseline = None
	while not done:
		beams_last_steps = None
		if beams_num > 0 and frame_iter % beam_freq == 0: # It is intended to be true also at step 0
			# print("Running beams at frame iteration %i" % frame_iter)
			beams = [call_beam(s) for _ in range(beams_num)]
			beams_returns = [get_returns_from_rewards([e["reward"] for e in b], discount_factor) for b in beams]
			beams_last_steps = [(beams[b_index][:beam_freq], beams_returns[b_index][:beam_freq]) for b_index in range(len(beams))]
			beams_returns = [Gbr[0] for Gbr in beams_returns]
			beams_baseline = sum(beams_returns)/len(beams_returns)
		action, log_p = select_action(model, s)
		s_next, r, done, _ = env.step(action)
		if render:
			env.render()
			time.sleep(0.05)
		episode += [{"state": s, "action": action, "log_p": log_p, "reward": r, "done" : done, "baseline": beams_baseline, "beams": beams_last_steps}]
		s = s_next
		frame_iter += 1
	if render:
		env.close()
	return episode


def train(model, env, num_episodes, optimizer, discount_factor, loss_fun=compute_reinforce_loss,
		  print_freq=10, final_render=True):

	if loss_fun == compute_reinforce_with_baseline_fork_update_loss:
		run_eps = lambda : run_episode(env, model, select_action=select_action_default,
									   greedy_actions=True, render=False, beams_num=2, beam_freq=4,
									   beams_greedy=False, discount_factor=discount_factor)
	elif loss_fun == compute_reinforce_with_baseline_loss:
		run_eps = lambda : run_episode(env, model, select_action=select_action_default,
									   greedy_actions=False, render=False, beams_num=4, beam_freq=2,
									   beams_greedy=False, discount_factor=discount_factor)
	elif loss_fun == compute_reinforce_loss:
		run_eps = lambda : run_episode(env, model, select_action=select_action_default,
									   greedy_actions=False, render=False, beams_num=-1, beam_freq=-1,
									   beams_greedy=False, discount_factor=discount_factor)

	episode_durations = []
	model = model.to(get_device())
	metric_avg = np.zeros((4,2), dtype=np.float32)
	actions_taken = []
	for i in range(num_episodes):
		episode = run_eps()
		episode_durations.append(len(episode))
		loss = loss_fun(episode, discount_factor)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		metric_avg[0,0] += loss.item()
		metric_avg[1,0] += len(episode)
		metric_avg[2,0] += sum([e["reward"] for e in episode])
		metric_avg[3,0] += episode[-1]["reward"]
		metric_avg[:,1] += 1
		actions_taken += [e["action"] for e in episode]
		if (i+1) % print_freq == 0:
			metric_avg[:,0] = metric_avg[:,0] / metric_avg[:,1]
			print("Iteration %i: Loss=%4.2f, Episode length=%4.2f, Sum rewards=%4.2f, Final reward=%4.2f" % (i+1, metric_avg[0,0], metric_avg[1,0], metric_avg[2,0], metric_avg[3,0]))
			print("Action distribution: " + ", ".join(["%s: %4.2f%%" % (str(a), 100.0*sum([(a == at) for at in actions_taken])/len(actions_taken)) for a in set(actions_taken)]))
			metric_avg[:,:] = 0
	if final_render:
		render_episode(env, model)



