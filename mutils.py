import torch
import numpy as np
import random

def get_device():
	return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available:
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
