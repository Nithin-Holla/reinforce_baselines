import numpy as np 
import matplotlib.pyplot as plt
import random
random.seed(43)


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

	assert len(meta_information) == results.shape[0], "ERROR: Meta information needs to have the same number of entries as the results, but got %i results and %i meta information" % (results.shape[0], len(meta_information))

	experiments = dict()
	for index, meta_info in enumerate(meta_information):
		exp_id = ",".join([a for a in meta_info.split(",") if "seed=" not in a])
		if exp_id not in experiments:
			experiments[exp_id] = []
		experiments[exp_id].append(results[index])

	fig, ax = plt.subplots(1, 3)
	prev_colors = []

	for exp_labels, exp_infos in experiments.items():
		print(exp_labels)
		# if not "nr_beams=(4, False)" in exp_labels or not "lr=0.002" in exp_labels:
		# 	continue
		if not "lr=4e-05" in exp_labels:
			continue

		color = None
		color_iter = 0
		while color is None:
			color_iter += 1
			color = (random.random(), random.random(), random.random())
			if any([sum([abs(color[j]-c[j]) for j in range(3)]) < 0.9**color_iter for c in prev_colors]):
				color = None
		prev_colors.append(color)

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
		ax[0].fill_between(exp_infos[0,:,0], episode_len_percentile_25, episode_len_percentile_75, color=color, alpha=0.2)
		ax[0].plot(exp_infos[0,:,0], episode_len_percentile_25, '--', color=color, alpha=0.5)
		ax[0].plot(exp_infos[0,:,0], episode_len_percentile_75, '--', color=color, alpha=0.5)

		##########
		## Plotting over number of interactions
		##########
		exp_infos_flatten = np.reshape(exp_infos, (exp_infos.shape[0] * exp_infos.shape[1], exp_infos.shape[2]) )
		sort_indices = np.argsort(exp_infos_flatten[:,2])
		exp_infos_episodes = exp_infos_flatten[sort_indices,1]
		exp_infos_interactions = exp_infos_flatten[sort_indices,2]

		smooth_window = 50
		# smooth_window = 500
		exp_infos_episodes_smoothed = smooth_vals(exp_infos_episodes, N=smooth_window)
		exp_infos_episodes_stacked = np.stack([exp_infos_episodes[i:-(smooth_window-(i+1))] if i<smooth_window-1 else exp_infos_episodes[i:] for i in range(smooth_window)], axis=0)
		exp_infos_episodes_percentile_25 = np.percentile(exp_infos_episodes_stacked, q=25, axis=0)
		exp_infos_episodes_percentile_75 = np.percentile(exp_infos_episodes_stacked, q=75, axis=0)
		exp_infos_episodes_percentile_25 = smooth_vals(exp_infos_episodes_percentile_25, N=smooth_window)
		exp_infos_episodes_percentile_75 = smooth_vals(exp_infos_episodes_percentile_75, N=smooth_window)

		ax[1].plot(exp_infos_interactions[:-smooth_window//2+1], exp_infos_episodes_smoothed[:-smooth_window//2+1], color=color, label=exp_labels)
		ax[1].fill_between(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, exp_infos_episodes_percentile_75, color=color, alpha=0.2)
		ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, '--', color=color, alpha=0.5)
		ax[1].plot(exp_infos_interactions[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_75, '--', color=color, alpha=0.5)

		##########
		## Plotting over time
		##########
		sort_indices = np.argsort(exp_infos_flatten[:,3])
		exp_infos_episodes = exp_infos_flatten[sort_indices,1]
		exp_infos_time = exp_infos_flatten[sort_indices,3]
		print(exp_infos_time)

		smooth_window = 50
		# smooth_window = 500
		exp_infos_episodes_smoothed = smooth_vals(exp_infos_episodes, N=smooth_window)
		exp_infos_episodes_stacked = np.stack([exp_infos_episodes[i:-(smooth_window-(i+1))] if i<smooth_window-1 else exp_infos_episodes[i:] for i in range(smooth_window)], axis=0)
		exp_infos_episodes_percentile_25 = np.percentile(exp_infos_episodes_stacked, q=25, axis=0)
		exp_infos_episodes_percentile_75 = np.percentile(exp_infos_episodes_stacked, q=75, axis=0)
		exp_infos_episodes_percentile_25 = smooth_vals(exp_infos_episodes_percentile_25, N=smooth_window)
		exp_infos_episodes_percentile_75 = smooth_vals(exp_infos_episodes_percentile_75, N=smooth_window)

		ax[2].plot(exp_infos_time[:-smooth_window//2+1], exp_infos_episodes_smoothed[:-smooth_window//2+1], color=color, label=exp_labels)
		ax[2].fill_between(exp_infos_time[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, exp_infos_episodes_percentile_75, color=color, alpha=0.2)
		ax[2].plot(exp_infos_time[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_25, '--', color=color, alpha=0.5)
		ax[2].plot(exp_infos_time[smooth_window//2:-smooth_window//2+1], exp_infos_episodes_percentile_75, '--', color=color, alpha=0.5)

	fig.suptitle("REINFORCE over different learning rates", fontsize=14)

	ax[0].title.set_text("Mean episode length over training iterations")
	ax[0].set_xlabel("Iterations")
	ax[0].set_ylabel("Episode length")

	ax[1].title.set_text("Mean episode length over interactions")
	ax[1].set_xlabel("Interactions with environment")
	ax[1].set_ylabel("Episode length")

	ax[2].title.set_text("Mean episode length over time")
	ax[2].set_xlabel("Time in seconds")
	ax[2].set_ylabel("Episode length")

	#ax[0].legend()
	#ax[1].legend()
	ax[2].legend()
	plt.show()
	# plt.savefig("gridsearch_figure.png")

if __name__ == '__main__':
	# plot_gridsearch_result(meta_file="data_gridsearch/07_10_2019__21_46_17/gridsearch_reinforce_with_baseline_meta.txt", 
	# 					   numpy_file="data_gridsearch/07_10_2019__21_46_17/gridsearch_reinforce_with_baseline.npz")
	plot_gridsearch_result(meta_file="data_gridsearch/07_10_2019__23_24_46/gridsearch_reinforce_meta.txt",
						   numpy_file="data_gridsearch/07_10_2019__23_24_46/gridsearch_reinforce.npz")