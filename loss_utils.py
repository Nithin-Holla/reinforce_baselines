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

	print("Gs", Gs)
	print("Baseline", baseline)
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
			baseline_Gts = [Gs[e_step_index]] + [b_Gt[0] for _, b_Gt in e["beams"]]
			baseline_Gts = sum(baseline_Gts) / len(baseline_Gts)
			for b_episode, b_Gt in e["beams"]:
				for b_step_index in range(len(b_episode)):
					Gs.append(b_Gt[b_step_index])
					log_ps.append(b_episode[b_step_index]["log_p"])
					baseline.append(baseline_Gts)

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
	Gs = Gs / Gs.abs().mean()
	# print(Gs)
	loss = - torch.mean(Gs * log_ps)
	return loss

