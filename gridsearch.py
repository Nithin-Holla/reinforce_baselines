import torch 
import torch.nn as nn
import numpy as np 

from lunar_training import *
from train_utils import *
from loss_utils import *


def perform_reinforce_with_baseline_gridsearch(learning_rates=[2e-3, 5e-3, 1e-2], 
											   number_of_beams=[(1, True), (1, False), (2, False), (4, False), (8, False)],
											   intermediate_steps=[2, 3, 4],
											   num_seeds=32,
											   num_episodes=500,
											   discount_factor=0.99,
											   export_filename="gridsearch_reinforce_with_baseline"):
	meta_information = ""
	results = None

	def add_result(episode_lens, param_dict):
		if len(meta_information) > 0:
			meta_information += "\n"
		meta_information += ",".join(sorted([key + "=" + str(val) for key, val in param_dict.items()]))
		episode_lens = np.array([episode_lens], dtype=np.int32)
		if results is None:
			results = episode_lens
		else:
			results = np.concatenate([results, episode_lens], axis=0)
		
		with open(export_filename + "_meta.txt", "w") as f:
			f.write(meta_information)
		np.savez_compressed(export_filename+".npz", results)

	exp_index = 0
	for seed in range(42, 42+num_seeds):
		for lr in learning_rates:
			for nr_beams in number_of_beams:
				for beam_steps in intermediate_steps:
					exp_index += 1
					params = {"lr": lr, "nr_beams": nr_beams, "logbasis": beam_steps, "seed": seed}
					print("Gridsearch experiment %i: params %s" % (exp_index, str(params)))

					set_seed(seed=seed)
					env = gym.make("CartPole-v1")
					env.seed(seed)
					model = LunarLinearPolicyNetwork(num_inputs=4, num_actions=2)
					optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

					run_eps = lambda : run_episode_logN(env, model, select_action=select_action_default,
														greedy_actions=False, render=False, beams_num=nr_beams[0], 
														log_basis=beam_steps, beam_start_freq=2 if beam_steps <= 2 else 1,
														beams_greedy=nr_beams[1], discount_factor=discount_factor)

					res = train(env=env, model=model, num_episodes=num_episodes, 
								optimizer=optimizer, discount_factor=discount_factor, 
								loss_fun=compute_reinforce_with_baseline_fork_update_loss, 
								print_freq=10, final_render=False, run_eps=run_eps, 
								early_stopping=True)

					add_result(res, params)


if __name__ == "__main__":
	perform_reinforce_with_baseline_gridsearch()

