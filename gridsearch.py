import torch 
import torch.nn as nn
import numpy as np 
import datetime
import os
import time

from lunar_training import *
from train_utils import *
from loss_utils import *
from learned_value import *

GRIDSEARCH_FOLDER = "data_gridsearch"

def perform_reinforce_with_baseline_gridsearch(learning_rates=[2e-3, 5e-3, 1e-2], 
											   number_of_beams=[(1, True), (1, False), (2, False), (4, False)],
											   intermediate_steps=[2, 3, 4],
											   num_seeds=32,
											   num_episodes=500,
											   discount_factor=0.99,
											   export_filename="gridsearch_reinforce_with_baseline",
											   save_results=True):
	global GRIDSEARCH_FOLDER
	current_date = datetime.datetime.now()
	base_dir = os.path.join(GRIDSEARCH_FOLDER, "%02d_%02d_%02d__%02d_%02d_%02d/" % ((current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute, current_date.second)))
	os.makedirs(base_dir, exist_ok=True)
	meta_information = ""
	results = None

	def add_result(episode_lens, param_dict, meta_information, results):
		if len(meta_information) > 0:
			meta_information += "\n"
		meta_information += ",".join(sorted([key + "=" + str(val) for key, val in param_dict.items()]))
		if results is None:
			results = episode_lens[None,:,:]
		else:
			results = np.concatenate([results, episode_lens[None,:,:]], axis=0)
		
		with open(base_dir + export_filename + "_meta.txt", "w") as f:
			f.write(meta_information)
		np.savez_compressed(base_dir + export_filename+".npz", results)
		return meta_information, results

	print(("*"*80+"\n")*2 + "Starting gridsearch for REINFORCE with baseline over %i learning rates, %i number of beams and %i intermediate steps..." % (len(learning_rates), len(number_of_beams), len(intermediate_steps)))
	print(("*"*80+"\n")*2)

	exp_index = 0
	num_exps = num_seeds * len(learning_rates) * len(number_of_beams) * len(intermediate_steps)
	start_time = time.time()
	for seed in range(42, 42+num_seeds):
		for lr in learning_rates:
			for nr_beams in number_of_beams:
				for beam_steps in intermediate_steps:
					exp_index += 1
					params = {"lr": lr, "nr_beams": nr_beams, "logbasis": beam_steps, "seed": seed}
					print("="*75)
					print("Gridsearch experiment %i | %i: params %s" % (exp_index, num_exps, str(params)))
					if exp_index > 1:
						time_passed = int(time.time() - start_time)
						time_estimate = int(time_passed * ((num_exps - exp_index + 1) / (exp_index - 1)))
						print("-> Time passed until now: %ih %imin %isec" % (time_passed//3600, (time_passed//60)%60, time_passed%60))
						print("-> Estimated time until finishing gridsearch: %ih %imin %isec" % (time_estimate//3600, (time_estimate//60)%60, time_estimate%60))
					print("="*75)

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

					if save_results:
						meta_information, results = add_result(res, params, meta_information, results)


def perform_reinforce_gridsearch(learning_rates=[4e-5, 1e-4, 4e-4, 1e-3], 
								 num_seeds=32,
								 num_episodes=2000,
								 discount_factor=0.99,
								 export_filename="gridsearch_reinforce",
								 save_results=True):
	global GRIDSEARCH_FOLDER
	current_date = datetime.datetime.now()
	base_dir = os.path.join(GRIDSEARCH_FOLDER, "%02d_%02d_%02d__%02d_%02d_%02d/" % ((current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute, current_date.second)))
	os.makedirs(base_dir, exist_ok=True)
	meta_information = ""
	results = None

	def add_result(episode_lens, param_dict, meta_information, results):
		if len(meta_information) > 0:
			meta_information += "\n"
		meta_information += ",".join(sorted([key + "=" + str(val) for key, val in param_dict.items()]))
		if results is None:
			results = episode_lens[None,:,:]
		else:
			results = np.concatenate([results, episode_lens[None,:,:]], axis=0)
		
		with open(base_dir + export_filename + "_meta.txt", "w") as f:
			f.write(meta_information)
		np.savez_compressed(base_dir + export_filename+".npz", results)
		return meta_information, results

	print(("*"*80+"\n")*2 + "Starting gridsearch for REINFORCE with %i learning rates..." % len(learning_rates))
	print(("*"*80+"\n")*2)

	exp_index = 0
	num_exps = num_seeds * len(learning_rates)
	start_time = time.time()
	for seed in range(42, 42+num_seeds):
		for lr in learning_rates:
			exp_index += 1
			params = {"lr": lr, "seed": seed}
			print("="*75)
			print("Gridsearch experiment %i | %i: params %s" % (exp_index, num_exps, str(params)))
			if exp_index > 1:
				time_passed = int(time.time() - start_time)
				time_estimate = int(time_passed * ((num_exps - exp_index + 1) / (exp_index - 1)))
				print("-> Time passed until now: %ih %imin %isec" % (time_passed//3600, (time_passed//60)%60, time_passed%60))
				print("-> Estimated time until finishing gridsearch: %ih %imin %isec" % (time_estimate//3600, (time_estimate//60)%60, time_estimate%60))
			print("="*75)

			set_seed(seed=seed)
			env = gym.make("CartPole-v1")
			env.seed(seed)
			model = LunarLinearPolicyNetwork(num_inputs=4, num_actions=2)
			optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

			run_eps = lambda : run_episode(env, model, select_action=select_action_default,
										   greedy_actions=False, render=False, discount_factor=discount_factor)

			res = train(env=env, model=model, num_episodes=num_episodes, 
						optimizer=optimizer, discount_factor=discount_factor, 
						loss_fun=compute_reinforce_loss, 
						print_freq=50, final_render=False, run_eps=run_eps, 
						early_stopping=False)

			if save_results:
				meta_information, results = add_result(res, params, meta_information, results)

def perform_lv_gridsearch(learning_rates=[2e-4, 5e-4, 1e-3],
						  	alphas=[0.5,0.25,0.75,0.1,0.9],
						  	num_seeds=32,
						  	num_episodes=2000,
							discount_factor=0.99,
						    export_filename="gridsearch_lv",
							save_results=True):
	global GRIDSEARCH_FOLDER
	current_date = datetime.datetime.now()
	base_dir = os.path.join(GRIDSEARCH_FOLDER, "%02d_%02d_%02d__%02d_%02d_%02d/" % ((current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute, current_date.second)))
	os.makedirs(base_dir, exist_ok=True)
	meta_information = ""
	results = None

	def add_result(episode_lens, param_dict, meta_information, results):
		if len(meta_information) > 0:
			meta_information += "\n"
		meta_information += ",".join(sorted([key + "=" + str(val) for key, val in param_dict.items()]))
		if results is None:
			results = episode_lens[None,:,:]
		else:
			results = np.concatenate([results, episode_lens[None,:,:]], axis=0)
		
		with open(base_dir + export_filename + "_meta.txt", "w") as f:
			f.write(meta_information)
		np.savez_compressed(base_dir + export_filename+".npz", results)
		return meta_information, results

	print(("*"*80+"\n")*2 + "Starting gridsearch for Actor Critic Learned Value Function with %i learning rates and %i alphas..." % (len(learning_rates), len(alphas)))
	print(("*"*80+"\n")*2)

	exp_index = 0
	num_exps = num_seeds * len(learning_rates) * len(alphas)
	start_time = time.time()
	for seed in range(42, 42+num_seeds):
		for lr in learning_rates:
			for alpha in alphas:
				exp_index += 1
				params = {"lr": lr, "seed": seed, "alpha": alpha}
				print("="*75)
				print("Gridsearch experiment %i | %i: params %s" % (exp_index, num_exps, str(params)))
				if exp_index > 1:
					time_passed = int(time.time() - start_time)
					time_estimate = int(time_passed * ((num_exps - exp_index + 1) / (exp_index - 1)))
					print("-> Time passed until now: %ih %imin %isec" % (time_passed//3600, (time_passed//60)%60, time_passed%60))
					print("-> Estimated time until finishing gridsearch: %ih %imin %isec" % (time_estimate//3600, (time_estimate//60)%60, time_estimate%60))
				print("="*75)

				set_seed(seed=seed)
				env = gym.make("CartPole-v1")
				env.seed(seed)
				model = LearnedValueNetwork(num_inputs=4, num_actions=2)
				optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)
				loss_fun = lambda x, y: compute_lv_loss(x, y, alpha=alpha)
				run_eps = lambda: run_episode_lv(env, model, select_action=select_action_lv,
											 	 render=False, discount_factor=discount_factor)
				res = train(env=env, model=model, select_action=select_action_lv, optimizer=optimizer, num_episodes=1000, 
							loss_fun=loss_fun, discount_factor=discount_factor, print_freq=50, final_render=False,
							early_stopping=False, run_eps=run_eps)

				if save_results:
					meta_information, results = add_result(res, params, meta_information, results)




if __name__ == "__main__":
	# perform_reinforce_with_baseline_gridsearch(num_episodes=50, learning_rates=[5e-3], number_of_beams=[(1, True)], intermediate_steps=[2,3,4])
	# perform_reinforce_with_baseline_gridsearch(num_episodes=50, save_results=False)
	# perform_reinforce_gridsearch()
	perform_lv_gridsearch()
	# perform_reinforce_with_baseline_gridsearch(num_episodes=500, learning_rates=[1e-3, 2e-3, 4e-3], intermediate_steps=[2, 4])

