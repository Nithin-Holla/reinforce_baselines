import numpy as np

import torch
import torch.nn as nn
import time

import matplotlib.pyplot as plt
from mutils import *


def compute_reinforce_loss(episode, discount_factor):
	log_ps = torch.zeros(len(episode))
	Gs  = []

	G = 0
	loss = 0

	for i, e in enumerate(reversed(episode)):
		t = len(episode) - 1 - i
		G = discount_factor * G + e["reward"]
		Gs += [G]
		log_ps[i] = e["log_p"]

	Gs = torch.tensor(Gs).to(get_device())
	Gs = (Gs - torch.mean(Gs)) / torch.std(Gs)
	loss = - torch.sum(Gs * log_ps)
	return loss