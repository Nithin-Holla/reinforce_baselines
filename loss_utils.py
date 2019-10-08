import numpy as np

import torch
import torch.nn as nn
import time

import matplotlib.pyplot as plt
from mutils import *


def get_returns_from_rewards(rewards, discount_factor):
	T = len(rewards)
	G_t = np.zeros((T,), dtype=np.float32)
	G_t[-1] = rewards[-1]
	for i in range(T-2, -1, -1):
		G_t[i] = discount_factor * G_t[i+1] + rewards[i]
	return G_t


def compute_reinforce_loss(episode, discount_factor):
	Gs = get_returns_from_rewards([e["reward"] for e in episode], discount_factor)
	log_ps = torch.stack([e["log_p"] for e in episode])

	Gs = torch.tensor(Gs).to(get_device())
	Gs = (Gs - torch.mean(Gs)) / torch.std(Gs)
	loss = - torch.sum(Gs * log_ps)
	return loss


def compute_reinforce_with_baseline_loss(episode, discount_factor):
	Gs = get_returns_from_rewards([e["reward"] for e in episode], discount_factor)
	log_ps = torch.stack([e["log_p"] for e in episode])
	baseline = [e["baseline"] for e in episode]

	Gs = torch.FloatTensor(Gs).to(get_device())
	baseline = torch.FloatTensor(baseline).to(get_device())
	# print("Rewards", [e["reward"] for e in episode])
	# print("Gs", Gs)
	# print("baseline", baseline)

	# print("Gs", Gs)
	# print("Baseline", baseline)
	Gs = Gs - baseline
	# Gs = Gs / Gs.std()
	Gs = Gs / Gs.abs().mean()
	loss = - torch.sum(Gs * log_ps)
	return loss


def compute_reinforce_with_baseline_fork_update_loss(episode, discount_factor):
	Gs = get_returns_from_rewards([e["reward"] for e in episode], discount_factor).tolist()
	log_ps = [e["log_p"] for e in episode]
	baseline = [e["baseline"] for e in episode]

	for e_step_index, e in enumerate(episode):
		if e["beams"] is not None:
			beam_length = max([len(b_episode) for b_episode, b_GT in e["beams"]])
			baseline_Gts = [[Gs[min(e_step_index+j,len(Gs)-1)]] + [b_Gt[min(j, len(b_Gt)-1)] for _, b_Gt in e["beams"]] for j in range(beam_length)]
			b_index = 0
			for b_episode, b_Gt in e["beams"]:
				b_index += 1
				b_baseline_Gts = [sum(baseGt[:b_index]+baseGt[b_index+1:]) / (len(baseGt)-1) for baseGt in baseline_Gts]
				for b_step_index in range(len(b_episode)):
					Gs.append(b_Gt[b_step_index])
					log_ps.append(b_episode[b_step_index]["log_p"])
					baseline.append(b_baseline_Gts[b_step_index])

	# print("Length of Gs", len(Gs))

	log_ps = torch.stack(log_ps)
	Gs = torch.FloatTensor(Gs).to(get_device())
	baseline = torch.FloatTensor(baseline).to(get_device())
	# print("Rewards", [e["reward"] for e in episode])
	# print("Gs", Gs)
	# print("baseline", baseline)

	# print("Gs", Gs)
	# print("Baseline", baseline)
	Gs = Gs - baseline
	# Gs = Gs / Gs.std()
	Gs = Gs / (Gs.abs().mean() + 1e-5)
	# print("Gs postprocessing", Gs)
	# print("log_ps", log_ps)
	loss = - torch.mean(Gs * log_ps)
	return loss


def compute_lv_loss(episode, discount_factor, alpha=0.5):
	Gs = get_returns_from_rewards([e["reward"] for e in episode], discount_factor)
	
	log_ps = torch.stack([e["log_p"] for e in episode])
	values = torch.stack([e["baseline"] for e in episode]).view(-1)
	Gs = torch.tensor(Gs).to(get_device())
	Gs = Gs / (Gs.abs().mean() + 1e-5)
	value_loss =  nn.MSELoss()(Gs, values)
	

	adv = Gs - values.detach()
	
	policy_loss = -torch.sum(adv * log_ps)
	
	loss = (1-alpha)*policy_loss + alpha*value_loss
	return loss
