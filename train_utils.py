import numpy as np
import math 

import gym
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import time
from copy import copy, deepcopy

import matplotlib.pyplot as plt
from loss_utils import *
from mutils import *

NUM_INTERACTIONS = 0
def add_interaction():
	global NUM_INTERACTIONS
	NUM_INTERACTIONS += 1

def reset_interactions():
	global NUM_INTERACTIONS
	NUM_INTERACTIONS = 0

def get_interactions():
	global NUM_INTERACTIONS
	return NUM_INTERACTIONS



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
	beams_returns = None
	while not done:
		beams_last_steps = None
		if beams_num > 0 and frame_iter % beam_freq == 0: # It is intended to be true also at step 0
			# print("Running beams at frame iteration %i" % frame_iter)
			beams = [call_beam(s) for _ in range(beams_num)]
			beams_returns = [get_returns_from_rewards([e["reward"] for e in b], discount_factor) for b in beams]
			beams_last_steps = [(beams[b_index][:beam_freq], beams_returns[b_index][:beam_freq]) for b_index in range(len(beams))]
			beams_returns = [sum([Gbr[baseline_step] if len(Gbr) > baseline_step else Gbr[-1] for Gbr in beams_returns])/len(beams_returns) for baseline_step in range(beam_freq)]
		if beams_returns is not None:
			beams_baseline = beams_returns[0]
			del beams_returns[0]
		action, log_p = select_action(model, s)
		s_next, r, done, _ = env.step(action)
		if render:
			env.render()
			time.sleep(0.05)
		episode += [{"state": s, "action": action, "log_p": log_p, "reward": r, "done" : done, "baseline": beams_baseline, "beams": beams_last_steps}]
		s = s_next
		frame_iter += 1
		add_interaction()
	if render:
		env.close()
	return episode


def run_episode_logN(env, model, select_action=select_action_default, 
					greedy_actions=False, render=False, 
					beams_num=-1, beam_start_freq=2, beams_greedy=False,
					discount_factor=0.99):

	call_beam = lambda state : run_episode(deepcopy(env), model, select_action, 
										   greedy_actions=beams_greedy, render=False, 
										   beams_num=-1, beam_freq=-1, beams_greedy=False, 
										   discount_factor=discount_factor, last_state=state)

	seed = random.randint(0, 10000)
	env.seed(seed)
	s = env.reset()
	if render:
		env.render()
	episode = []
	done = False
	frame_iter = 0
	while not done:
		action, log_p = select_action(model, s)
		s_next, r, done, _ = env.step(action)
		if render:
			env.render()
			time.sleep(0.05)
		episode += [{"state": s, "action": action, "log_p": log_p, "reward": r, "done" : done, "baseline": None, "beams": None}]
		s = s_next
		frame_iter += 1
		add_interaction()
	if render:
		env.close()

	episode_length = len(episode)
	num_beam_start_states = math.ceil(math.log2(episode_length))
	beam_start_states = sorted(list(set([(episode_length - 2**i) for i in range(beam_start_freq, num_beam_start_states)] + [0])))
	print("Episode length", episode_length)
	print("Beam start states", beam_start_states)
	
	beams_baseline = None
	beams_returns = None
	env.seed(seed)
	s = env.reset()
	for i in range(episode_length):
		beams_last_steps = None
		if i in beam_start_states:
			# print("Running beams at frame iteration %i" % frame_iter)
			beam_state = beam_start_states.index(i)
			beam_freq = (beam_start_states[beam_state+1] if (beam_state+1)<len(beam_start_states) else (episode_length)) - i
			beams = [call_beam(s) for _ in range(beams_num)]
			beams_returns = [get_returns_from_rewards([e["reward"] for e in b], discount_factor) for b in beams]
			beams_last_steps = [(beams[b_index][:beam_freq], beams_returns[b_index][:beam_freq]) for b_index in range(len(beams))]
			beams_returns = [sum([Gbr[baseline_step] if len(Gbr) > baseline_step else Gbr[-1] for Gbr in beams_returns])/len(beams_returns) for baseline_step in range(beam_freq)]
		if beams_returns is not None:
			beams_baseline = beams_returns[0]
			del beams_returns[0]
		_ = env.step(episode[i]["action"])
		episode[i]["baseline"] = beams_baseline
		episode[i]["beams"] = beams_last_steps
		# We do not add an interaction here because we just revisit the same trajectory as sampled above. 

	return episode


def train(model, env, num_episodes, optimizer, discount_factor, loss_fun=compute_reinforce_loss,
		  print_freq=10, final_render=True):

	if loss_fun == compute_reinforce_with_baseline_fork_update_loss:
		run_eps = lambda : run_episode_logN(env, model, select_action=select_action_default,
									   greedy_actions=True, render=False, beams_num=2, beam_start_freq=2,
									   beams_greedy=False, discount_factor=discount_factor)
	elif loss_fun == compute_reinforce_with_baseline_fork_update_loss:
		run_eps = lambda : run_episode(env, model, select_action=select_action_default,
									   greedy_actions=True, render=False, beams_num=2, beam_freq=40,
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
	reset_interactions()
	for i in range(num_episodes):
		episode = run_eps()
		episode_durations.append(len(episode))
		loss = loss_fun(episode, discount_factor)
		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
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
			print("Number of interactions so far: %i" % get_interactions())
			metric_avg[:,:] = 0
	if final_render:
		render_episode(env, model)



