import numpy as np 
import matplotlib.pyplot as plt
import random
random.seed(40)
import seaborn as sns
sns.set()
current_palette = sns.color_palette()
print(current_palette)


def smooth_vals(vals, N=20):
	vals_smooth = np.zeros((vals.shape[0],), dtype=np.float32)
	for i in range(vals.shape[0]):
		start_x = max(0, int(i-N/2))
		stop_x = min(vals.shape[0], int(i+N/2))
		vals_smooth[i] = vals[start_x:stop_x].sum() / (stop_x - start_x)
	return vals_smooth


def plot_gridsearch_result(meta_file, numpy_file):

	with open(meta_file, "r") as f:
		meta_information = [l.strip() for l in f.readlines()]
	results = np.load(numpy_file)["arr_0"]
	print(results.shape)
	results = results[:,:1000,:]

	assert len(meta_information) == results.shape[0], "ERROR: Meta information needs to have the same number of entries as the results, but got %i results and %i meta information" % (results.shape[0], len(meta_information))

	experiments = dict()
	for index, meta_info in enumerate(meta_information):
		exp_id = ",".join([a for a in meta_info.split(",") if "seed=" not in a])
		if exp_id not in experiments:
			experiments[exp_id] = []
		experiments[exp_id].append(results[index])

	fig, ax = plt.subplots(1, 2, figsize=(12, 4))
	prev_colors = [(0,0,0), (1,1,1)]

	with plt.style.context("seaborn"):
		for exp_labels, exp_infos in experiments.items():
			print(exp_labels)
			if (not "nr_beams=(1, False)" in exp_labels and not "nr_beams=(2, False)" in exp_labels and not "nr_beams=(4, False)" in exp_labels) or not "logbasis=2" in exp_labels or not "lr=0.002" in exp_labels:
				continue
			if "nr_beams=(1, False)" in exp_labels:
				exp_labels = "1 beam"
			if "nr_beams=(2, False)" in exp_labels:
				exp_labels = "2 beams"
			if "nr_beams=(4, False)" in exp_labels:
				exp_labels = "4 beams"
			# if not "lr=4e-05" in exp_labels:
			# 	continue

			color = None
			color_iter = 0
			while color is None:
				color_iter += 1
				color = (random.random(), random.random(), random.random())
				if any([sum([abs(color[j]-c[j]) for j in range(3)]) < 0.9**color_iter for c in prev_colors]):
					color = None
			prev_colors.append(color)
			color = current_palette[len(prev_colors)-1]

			##########
			## Plotting over iterations
			##########
			exp_infos = np.stack(exp_infos, axis=0)
			episode_len_means = np.mean(exp_infos[:,:,1], axis=0)
			episode_len_percentile_25 = np.percentile(exp_infos[:,:,1], q=25, axis=0)
			episode_len_percentile_75 = np.percentile(exp_infos[:,:,1], q=75, axis=0)

			smooth_window = 10
			# smooth_window = 50
			episode_len_means = smooth_vals(episode_len_means, N=smooth_window)
			episode_len_percentile_25 = smooth_vals(episode_len_percentile_25, N=smooth_window)
			episode_len_percentile_75 = smooth_vals(episode_len_percentile_75, N=smooth_window)

			ax[0].plot(exp_infos[0,:,0], episode_len_means, color=color, label=exp_labels)
			ax[0].fill_between(exp_infos[0,:,0], episode_len_percentile_25, episode_len_percentile_75, color=color, facecolor=color, alpha=0.25)
			ax[0].plot(exp_infos[0,:,0], episode_len_percentile_25, '--', color=color, alpha=0.65)
			ax[0].plot(exp_infos[0,:,0], episode_len_percentile_75, '--', color=color, alpha=0.65)

			##########
			## Plotting over number of interactions
			##########
			exp_infos_flatten = np.reshape(exp_infos, (exp_infos.shape[0] * exp_infos.shape[1], exp_infos.shape[2]) )
			sort_indices = np.argsort(exp_infos_flatten[:,2])
			exp_infos_episodes = exp_infos_flatten[sort_indices,1]
			exp_infos_interactions = exp_infos_flatten[sort_indices,2]

			smooth_window = 500
			# smooth_window = 500
			exp_infos_episodes_smoothed = smooth_vals(exp_infos_episodes, N=smooth_window)
			exp_infos_episodes_stacked = np.stack([exp_infos_episodes[i:-(smooth_window-(i+1))] if i<smooth_window-1 else exp_infos_episodes[i:] for i in range(smooth_window)], axis=0)
			exp_infos_episodes_percentile_25 = np.percentile(exp_infos_episodes_stacked, q=25, axis=0)
			exp_infos_episodes_percentile_75 = np.percentile(exp_infos_episodes_stacked, q=75, axis=0)
			exp_infos_episodes_percentile_25 = smooth_vals(exp_infos_episodes_percentile_25, N=smooth_window)
			exp_infos_episodes_percentile_75 = smooth_vals(exp_infos_episodes_percentile_75, N=smooth_window)

			ax[1].plot(exp_infos_interactions[:-smooth_window//2+1], exp_infos_episodes_smoothed[:-smooth_window//2+1], color=color, label=exp_labels)
			ax[1].fill_between(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, exp_infos_episodes_percentile_75, color=color, alpha=0.25)
			ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, '--', color=color, alpha=0.65)
			ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_75, '--', color=color, alpha=0.65)

			##########
			## Plotting over time
			##########
			# sort_indices = np.argsort(exp_infos_flatten[:,3])
			# exp_infos_episodes = exp_infos_flatten[sort_indices,1]
			# exp_infos_time = exp_infos_flatten[sort_indices,3]
			# print(exp_infos_time)

			# smooth_window = 500
			# # smooth_window = 500
			# exp_infos_episodes_smoothed = smooth_vals(exp_infos_episodes, N=smooth_window)
			# exp_infos_episodes_stacked = np.stack([exp_infos_episodes[i:-(smooth_window-(i+1))] if i<smooth_window-1 else exp_infos_episodes[i:] for i in range(smooth_window)], axis=0)
			# exp_infos_episodes_percentile_25 = np.percentile(exp_infos_episodes_stacked, q=25, axis=0)
			# exp_infos_episodes_percentile_75 = np.percentile(exp_infos_episodes_stacked, q=75, axis=0)
			# exp_infos_episodes_percentile_25 = smooth_vals(exp_infos_episodes_percentile_25, N=smooth_window)
			# exp_infos_episodes_percentile_75 = smooth_vals(exp_infos_episodes_percentile_75, N=smooth_window)

			# ax[2].plot(exp_infos_time[:-smooth_window//2+1], exp_infos_episodes_smoothed[:-smooth_window//2+1], color=color, label=exp_labels)
			# ax[2].fill_between(exp_infos_time[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, exp_infos_episodes_percentile_75, color=color, alpha=0.2)
			# ax[2].plot(exp_infos_time[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, '--', color=color, alpha=0.5)
			# ax[2].plot(exp_infos_time[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_75, '--', color=color, alpha=0.5)

	# fig.suptitle("REINFORCE with learned baseline over different number of beams/samples", fontsize=14)

	ax[0].title.set_text("Mean episode length over training iterations")
	ax[0].set_xlabel("Iterations")
	ax[0].set_ylabel("Episode length")
	ax[0].set_ylim(390,510)
	ax[0].set_xlim(0, 200)
	
	ax[1].title.set_text("Mean episode length over interactions")
	ax[1].set_xlabel("Interactions with environment")
	ax[1].set_ylabel("Episode length")
	ax[1].set_ylim(390,510)
	ax[1].set_xlim(0, 250000)

	# ax[2].title.set_text("Mean episode length over time")
	# ax[2].set_xlabel("Time in seconds")
	# ax[2].set_ylabel("Episode length")

	ax[0].legend()
	ax[1].legend()
	# ax[2].legend()
	plt.tight_layout()
	plt.show()
	# plt.savefig("gridsearch_figure.png")


def plot_gridsearch_forked_beams(meta_file, numpy_file):

	with open(meta_file, "r") as f:
		meta_information = [l.strip() for l in f.readlines()]
	results = np.load(numpy_file)["arr_0"]

	experiments = dict()
	for index, meta_info in enumerate(meta_information):
		exp_id = ",".join([a for a in meta_info.split(",") if "seed=" not in a])
		if exp_id not in experiments:
			experiments[exp_id] = []
		experiments[exp_id].append(results[index])

	fig, ax = plt.subplots(1, 2, figsize=(12, 4))

	with plt.style.context("seaborn"):
		for exp_labels, exp_infos in experiments.items():
			print(exp_labels)
			if (not "nr_beams=(1, False)" in exp_labels and not "nr_beams=(2, False)" in exp_labels and not "nr_beams=(4, False)" in exp_labels) or not "logbasis=2" in exp_labels or not "lr=0.002" in exp_labels:
				continue
			if "nr_beams=(1, False)" in exp_labels:
				exp_labels = "1 beam"
				color = current_palette[2]
			if "nr_beams=(2, False)" in exp_labels:
				exp_labels = "2 beams"
				color = current_palette[3]
			if "nr_beams=(4, False)" in exp_labels:
				exp_labels = "4 beams"
				color = current_palette[4]

			##########
			## Plotting over iterations
			##########
			exp_infos = np.stack(exp_infos, axis=0)
			episode_len_means = np.mean(exp_infos[:,:,1], axis=0)
			episode_len_percentile_25 = np.percentile(exp_infos[:,:,1], q=25, axis=0)
			episode_len_percentile_75 = np.percentile(exp_infos[:,:,1], q=75, axis=0)

			smooth_window = 10
			episode_len_means = smooth_vals(episode_len_means, N=smooth_window)
			episode_len_percentile_25 = smooth_vals(episode_len_percentile_25, N=smooth_window)
			episode_len_percentile_75 = smooth_vals(episode_len_percentile_75, N=smooth_window)

			ax[0].plot(exp_infos[0,:,0], episode_len_means, color=color, label=exp_labels)
			ax[0].fill_between(exp_infos[0,:,0], episode_len_percentile_25, episode_len_percentile_75, color=color, facecolor=color, alpha=0.25)
			ax[0].plot(exp_infos[0,:,0], episode_len_percentile_25, '--', color=color, alpha=0.65)
			ax[0].plot(exp_infos[0,:,0], episode_len_percentile_75, '--', color=color, alpha=0.65)

			##########
			## Plotting over number of interactions
			##########
			exp_infos_flatten = np.reshape(exp_infos, (exp_infos.shape[0] * exp_infos.shape[1], exp_infos.shape[2]) )
			sort_indices = np.argsort(exp_infos_flatten[:,2])
			exp_infos_episodes = exp_infos_flatten[sort_indices,1]
			exp_infos_interactions = exp_infos_flatten[sort_indices,2]

			smooth_window = 500
			exp_infos_episodes_smoothed = smooth_vals(exp_infos_episodes, N=smooth_window)
			exp_infos_episodes_stacked = np.stack([exp_infos_episodes[i:-(smooth_window-(i+1))] if i<smooth_window-1 else exp_infos_episodes[i:] for i in range(smooth_window)], axis=0)
			exp_infos_episodes_percentile_25 = np.percentile(exp_infos_episodes_stacked, q=25, axis=0)
			exp_infos_episodes_percentile_75 = np.percentile(exp_infos_episodes_stacked, q=75, axis=0)
			exp_infos_episodes_percentile_25 = smooth_vals(exp_infos_episodes_percentile_25, N=smooth_window)
			exp_infos_episodes_percentile_75 = smooth_vals(exp_infos_episodes_percentile_75, N=smooth_window)

			ax[1].plot(exp_infos_interactions[:-smooth_window//2+1], exp_infos_episodes_smoothed[:-smooth_window//2+1], color=color, label=exp_labels)
			ax[1].fill_between(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, exp_infos_episodes_percentile_75, color=color, alpha=0.25)
			ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, '--', color=color, alpha=0.65)
			ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_75, '--', color=color, alpha=0.65)

	# fig.suptitle("REINFORCE with learned baseline over different number of beams/samples", fontsize=14)

	ax[0].title.set_text("Mean episode length over training iterations")
	ax[0].set_xlabel("Iterations")
	ax[0].set_ylabel("Episode length")
	ax[0].set_ylim(390,510)
	ax[0].set_xlim(0, 200)
	
	ax[1].title.set_text("Mean episode length over interactions")
	ax[1].set_xlabel("Interactions with environment")
	ax[1].set_ylabel("Episode length")
	ax[1].set_ylim(390,510)
	ax[1].set_xlim(0, 250000)

	ax[0].legend()
	ax[1].legend()

	plt.tight_layout()
	# plt.show()
	plt.savefig('/home/phillip/Downloads/reinforce_with_baseline_different_beams.png', dpi=300)



def plot_gridsearch_reinforce(meta_file, numpy_file, save=True):

	with open(meta_file, "r") as f:
		meta_information = [l.strip() for l in f.readlines()]
	results = np.load(numpy_file)["arr_0"]
	results = results[:,:1000,:]

	experiments = dict()
	for index, meta_info in enumerate(meta_information):
		exp_id = ",".join([a for a in meta_info.split(",") if "seed=" not in a])
		if exp_id not in experiments:
			experiments[exp_id] = []
		experiments[exp_id].append(results[index])

	fig, ax = plt.subplots(1, 2, figsize=(12, 5))

	with plt.style.context("seaborn"):
		for exp_labels, exp_infos in experiments.items():
			print(exp_labels)
			if not "lr=4e-05" in exp_labels:
				continue
			exp_labels = "REINFORCE"
			color = current_palette[0]

			##########
			## Plotting over iterations
			##########
			exp_infos = np.stack(exp_infos, axis=0)
			episode_len_means = np.mean(exp_infos[:,:,1], axis=0)
			episode_len_percentile_25 = np.percentile(exp_infos[:,:,1], q=25, axis=0)
			episode_len_percentile_75 = np.percentile(exp_infos[:,:,1], q=75, axis=0)

			smooth_window = 50
			episode_len_means = smooth_vals(episode_len_means, N=smooth_window)
			episode_len_percentile_25 = smooth_vals(episode_len_percentile_25, N=smooth_window)
			episode_len_percentile_75 = smooth_vals(episode_len_percentile_75, N=smooth_window)

			ax[0].plot(exp_infos[0,:,0], episode_len_means, color=color, label=exp_labels)
			ax[0].fill_between(exp_infos[0,:,0], episode_len_percentile_25, episode_len_percentile_75, color=color, facecolor=color, alpha=0.25)
			ax[0].plot(exp_infos[0,:,0], episode_len_percentile_25, '--', color=color, alpha=0.65)
			ax[0].plot(exp_infos[0,:,0], episode_len_percentile_75, '--', color=color, alpha=0.65)

			##########
			## Plotting over number of interactions
			##########
			exp_infos_flatten = np.reshape(exp_infos, (exp_infos.shape[0] * exp_infos.shape[1], exp_infos.shape[2]) )
			sort_indices = np.argsort(exp_infos_flatten[:,2])
			exp_infos_episodes = exp_infos_flatten[sort_indices,1]
			exp_infos_interactions = exp_infos_flatten[sort_indices,2]

			smooth_window = 500
			exp_infos_episodes_smoothed = smooth_vals(exp_infos_episodes, N=smooth_window)
			exp_infos_episodes_stacked = np.stack([exp_infos_episodes[i:-(smooth_window-(i+1))] if i<smooth_window-1 else exp_infos_episodes[i:] for i in range(smooth_window)], axis=0)
			exp_infos_episodes_percentile_25 = np.percentile(exp_infos_episodes_stacked, q=25, axis=0)
			exp_infos_episodes_percentile_75 = np.percentile(exp_infos_episodes_stacked, q=75, axis=0)
			exp_infos_episodes_percentile_25 = smooth_vals(exp_infos_episodes_percentile_25, N=smooth_window)
			exp_infos_episodes_percentile_75 = smooth_vals(exp_infos_episodes_percentile_75, N=smooth_window)

			ax[1].plot(exp_infos_interactions[:-smooth_window//2+1], exp_infos_episodes_smoothed[:-smooth_window//2+1], color=color, label=exp_labels)
			ax[1].fill_between(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, exp_infos_episodes_percentile_75, color=color, alpha=0.25)
			ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, '--', color=color, alpha=0.65)
			ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_75, '--', color=color, alpha=0.65)

	# fig.suptitle("REINFORCE with learned baseline over different number of beams/samples", fontsize=14)

	ax[0].title.set_text("Mean episode length over training iterations")
	ax[0].set_xlabel("Iterations")
	ax[0].set_ylabel("Episode length")
	ax[0].set_ylim(-10,510)
	ax[0].set_xlim(0, 1000)
	
	ax[1].title.set_text("Mean episode length over interactions")
	ax[1].set_xlabel("Interactions with environment")
	ax[1].set_ylabel("Episode length")
	ax[1].set_ylim(-10,510)
	ax[1].set_xlim(0, 250000)

	plt.tight_layout()
	# plt.show()
	if save:
		plt.savefig('/home/phillip/Downloads/reinforce.png', dpi=300)
	return ax


def plot_gridsearch_learned_value_function(meta_file, numpy_file, save=True, ax=None):

	with open(meta_file, "r") as f:
		meta_information = [l.strip() for l in f.readlines()]
	results = np.load(numpy_file)["arr_0"]
	results = results[:,:1000,:]

	experiments = dict()
	for index, meta_info in enumerate(meta_information):
		exp_id = ",".join([a for a in meta_info.split(",") if "seed=" not in a])
		if exp_id not in experiments:
			experiments[exp_id] = []
		experiments[exp_id].append(results[index])

	if ax is None:
		fig, ax = plt.subplots(1, 2, figsize=(12, 4))

	with plt.style.context("seaborn"):
		for exp_labels, exp_infos in experiments.items():
			print(exp_labels)
			if not ("lr=0.0002" in exp_labels and "alpha=0.1" in exp_labels):
				continue
			exp_labels = "REINFORCE learned baseline"
			color = current_palette[1]

			##########
			## Plotting over iterations
			##########
			exp_infos = np.stack(exp_infos, axis=0)
			episode_len_means = np.mean(exp_infos[:,:,1], axis=0)
			episode_len_percentile_25 = np.percentile(exp_infos[:,:,1], q=25, axis=0)
			episode_len_percentile_75 = np.percentile(exp_infos[:,:,1], q=75, axis=0)

			smooth_window = 50
			episode_len_means = smooth_vals(episode_len_means, N=smooth_window)
			episode_len_percentile_25 = smooth_vals(episode_len_percentile_25, N=smooth_window)
			episode_len_percentile_75 = smooth_vals(episode_len_percentile_75, N=smooth_window)

			ax[0].plot(exp_infos[0,:,0], episode_len_means, color=color, label=exp_labels)
			ax[0].fill_between(exp_infos[0,:,0], episode_len_percentile_25, episode_len_percentile_75, color=color, facecolor=color, alpha=0.25)
			ax[0].plot(exp_infos[0,:,0], episode_len_percentile_25, '--', color=color, alpha=0.65)
			ax[0].plot(exp_infos[0,:,0], episode_len_percentile_75, '--', color=color, alpha=0.65)

			##########
			## Plotting over number of interactions
			##########
			exp_infos_flatten = np.reshape(exp_infos, (exp_infos.shape[0] * exp_infos.shape[1], exp_infos.shape[2]) )
			sort_indices = np.argsort(exp_infos_flatten[:,2])
			exp_infos_episodes = exp_infos_flatten[sort_indices,1]
			exp_infos_interactions = exp_infos_flatten[sort_indices,2]

			smooth_window = 500
			exp_infos_episodes_smoothed = smooth_vals(exp_infos_episodes, N=smooth_window)
			exp_infos_episodes_stacked = np.stack([exp_infos_episodes[i:-(smooth_window-(i+1))] if i<smooth_window-1 else exp_infos_episodes[i:] for i in range(smooth_window)], axis=0)
			exp_infos_episodes_percentile_25 = np.percentile(exp_infos_episodes_stacked, q=25, axis=0)
			exp_infos_episodes_percentile_75 = np.percentile(exp_infos_episodes_stacked, q=75, axis=0)
			exp_infos_episodes_percentile_25 = smooth_vals(exp_infos_episodes_percentile_25, N=smooth_window)
			exp_infos_episodes_percentile_75 = smooth_vals(exp_infos_episodes_percentile_75, N=smooth_window)

			ax[1].plot(exp_infos_interactions[:-smooth_window//2+1], exp_infos_episodes_smoothed[:-smooth_window//2+1], color=color, label=exp_labels)
			ax[1].fill_between(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, exp_infos_episodes_percentile_75, color=color, alpha=0.25)
			ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, '--', color=color, alpha=0.65)
			ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_75, '--', color=color, alpha=0.65)

	# fig.suptitle("REINFORCE with learned baseline over different number of beams/samples", fontsize=14)

	ax[0].title.set_text("Mean episode length over training iterations")
	ax[0].set_xlabel("Iterations")
	ax[0].set_ylabel("Episode length")
	ax[0].set_ylim(-10,510)
	ax[0].set_xlim(0, 1000)
	
	ax[1].title.set_text("Mean episode length over interactions")
	ax[1].set_xlabel("Interactions with environment")
	ax[1].set_ylabel("Episode length")
	ax[1].set_ylim(-10,510)
	ax[1].set_xlim(0, 250000)

	ax[0].legend()
	ax[1].legend()

	plt.tight_layout()
	# plt.show()
	if save:
		plt.savefig('/home/phillip/Downloads/reinforce_learned_val.png', dpi=300)
	return ax


def plot_gridsearch_sampled_baseline(meta_file, numpy_file, save=True, ax=None):

	with open(meta_file, "r") as f:
		meta_information = [l.strip() for l in f.readlines()]
	results = np.load(numpy_file)["arr_0"]
	results = np.concatenate([results, np.tile(results[:,-1:,:], (1,500,1))], axis=1)
	print(results.shape)

	experiments = dict()
	for index, meta_info in enumerate(meta_information):
		exp_id = ",".join([a for a in meta_info.split(",") if "seed=" not in a])
		if exp_id not in experiments:
			experiments[exp_id] = []
		experiments[exp_id].append(results[index])

	if ax is None:
		fig, ax = plt.subplots(1, 2, figsize=(12, 4))

	with plt.style.context("seaborn"):
		for exp_labels, exp_infos in experiments.items():
			print(exp_labels)
			if not ("nr_beams=(1, False)" in exp_labels and "logbasis=2" in exp_labels and "lr=0.002" in exp_labels):
				continue
			exp_labels = "REINFORCE sampled baseline"
			color = current_palette[2]

			##########
			## Plotting over iterations
			##########
			exp_infos = np.stack(exp_infos, axis=0)
			exp_infos[0,:,0] = np.arange(0, exp_infos.shape[1])
			episode_len_means = np.mean(exp_infos[:,:,1], axis=0)
			episode_len_percentile_25 = np.percentile(exp_infos[:,:,1], q=25, axis=0)
			episode_len_percentile_75 = np.percentile(exp_infos[:,:,1], q=75, axis=0)

			smooth_window = 10
			episode_len_means = smooth_vals(episode_len_means, N=smooth_window)
			episode_len_percentile_25 = smooth_vals(episode_len_percentile_25, N=smooth_window)
			episode_len_percentile_75 = smooth_vals(episode_len_percentile_75, N=smooth_window)

			ax[0].plot(exp_infos[0,:,0], episode_len_means, color=color, label=exp_labels)
			ax[0].fill_between(exp_infos[0,:,0], episode_len_percentile_25, episode_len_percentile_75, color=color, facecolor=color, alpha=0.25)
			ax[0].plot(exp_infos[0,:,0], episode_len_percentile_25, '--', color=color, alpha=0.65)
			ax[0].plot(exp_infos[0,:,0], episode_len_percentile_75, '--', color=color, alpha=0.65)

			##########
			## Plotting over number of interactions
			##########
			exp_infos_flatten = np.reshape(exp_infos, (exp_infos.shape[0] * exp_infos.shape[1], exp_infos.shape[2]) )
			sort_indices = np.argsort(exp_infos_flatten[:,2])
			exp_infos_episodes = exp_infos_flatten[sort_indices,1]
			exp_infos_interactions = exp_infos_flatten[sort_indices,2]

			smooth_window = 200
			exp_infos_episodes_smoothed = smooth_vals(exp_infos_episodes, N=smooth_window)
			exp_infos_episodes_stacked = np.stack([exp_infos_episodes[i:-(smooth_window-(i+1))] if i<smooth_window-1 else exp_infos_episodes[i:] for i in range(smooth_window)], axis=0)
			exp_infos_episodes_percentile_25 = np.percentile(exp_infos_episodes_stacked, q=25, axis=0)
			exp_infos_episodes_percentile_75 = np.percentile(exp_infos_episodes_stacked, q=75, axis=0)
			exp_infos_episodes_percentile_25 = smooth_vals(exp_infos_episodes_percentile_25, N=smooth_window)
			exp_infos_episodes_percentile_75 = smooth_vals(exp_infos_episodes_percentile_75, N=smooth_window)

			ax[1].plot(exp_infos_interactions[:-smooth_window//2+1], exp_infos_episodes_smoothed[:-smooth_window//2+1], color=color, label=exp_labels)
			ax[1].fill_between(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, exp_infos_episodes_percentile_75, color=color, alpha=0.25)
			ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, '--', color=color, alpha=0.65)
			ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_75, '--', color=color, alpha=0.65)

	# fig.suptitle("REINFORCE with learned baseline over different number of beams/samples", fontsize=14)

	ax[0].title.set_text("Mean episode length over training iterations")
	ax[0].set_xlabel("Iterations")
	ax[0].set_ylabel("Episode length")
	ax[0].set_ylim(-10,510)
	ax[0].set_xlim(0, 1000)
	
	ax[1].title.set_text("Mean episode length over interactions")
	ax[1].set_xlabel("Interactions with environment")
	ax[1].set_ylabel("Episode length")
	ax[1].set_ylim(-10,510)
	ax[1].set_xlim(0, 250000)

	ax[0].legend()
	ax[1].legend()

	plt.tight_layout()
	if save:
		plt.savefig('/home/phillip/Downloads/reinforce_all.png', dpi=300)
	return ax


def combine_gridsearches(meta_files, numpy_files, new_name):
	meta_info = []
	numpy_arrays = []
	for mf, nf in zip(meta_files, numpy_files):
		with open(mf, "r") as f:
			meta_info += [mf+","+line.strip().replace("\n","") for line in f.readlines()]
		numpy_arrays.append(np.load(nf)["arr_0"])
	with open(new_name + "_meta.txt", "w") as f:
		f.write("\n".join(meta_info))
	print([numpy_arrays])
	np.savez_compressed(new_name+".npz", np.concatenate(numpy_arrays, axis=0))


if __name__ == '__main__':
	# combine_gridsearches(meta_files = ["data_gridsearch/07_10_2019__19_50_16/gridsearch_reinforce_with_baseline_meta.txt",
	# 								   "data_gridsearch/07_10_2019__20_07_59/gridsearch_reinforce_with_baseline_meta.txt"],
	# 					 numpy_files = ["data_gridsearch/07_10_2019__19_50_16/gridsearch_reinforce_with_baseline.npz",
	# 					 				"data_gridsearch/07_10_2019__20_07_59/gridsearch_reinforce_with_baseline.npz"],
	# 					 new_name = "data_gridsearch/combined_gridsearch")
	# plot_gridsearch_forked_beams(meta_file="data_gridsearch/08_10_2019__10_19_08/gridsearch_reinforce_with_baseline_meta.txt", 
	# 					   		 numpy_file="data_gridsearch/08_10_2019__10_19_08/gridsearch_reinforce_with_baseline.npz")
	ax = plot_gridsearch_reinforce(meta_file="data_gridsearch/07_10_2019__23_24_46/gridsearch_reinforce_meta.txt",
						   numpy_file="data_gridsearch/07_10_2019__23_24_46/gridsearch_reinforce.npz",
						   save=True)
	ax = plot_gridsearch_learned_value_function(ax=ax,
											meta_file="data_gridsearch/08_10_2019__21_03_53/gridsearch_lv_meta.txt",
											numpy_file="data_gridsearch/08_10_2019__21_03_53/gridsearch_lv.npz",
											save=True)
	plot_gridsearch_sampled_baseline(ax=ax,
								 meta_file="data_gridsearch/08_10_2019__10_19_08/gridsearch_reinforce_with_baseline_meta.txt", 
	 					   		 numpy_file="data_gridsearch/08_10_2019__10_19_08/gridsearch_reinforce_with_baseline.npz",
	 					   		 save=True)