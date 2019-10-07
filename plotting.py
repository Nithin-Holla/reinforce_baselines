import numpy as np 
import matplotlib.pyplot as plt


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

	fig, ax = plt.subplots(2)

	for exp_labels, exp_infos in experiments.items():
		exp_infos = np.stack(exp_infos, axis=0)
		episode_len_means = np.mean(exp_infos[:,:,1], axis=0)
		episode_len_percentile_25 = np.percentile(exp_infos[:,:,1], q=25, axis=0)
		episode_len_percentile_75 = np.percentile(exp_infos[:,:,1], q=75, axis=0)

		episode_len_means = smooth_vals(episode_len_means, N=10)
		episode_len_percentile_25 = smooth_vals(episode_len_percentile_25, N=10)
		episode_len_percentile_75 = smooth_vals(episode_len_percentile_75, N=10)

		print("Episode x axis", exp_infos.shape)
		print("Episode len means", episode_len_means.shape)
		print("Episode len percentile 25", episode_len_percentile_25.shape)
		print("Episode len percentile 75", episode_len_percentile_75.shape)

		ax[0].plot(exp_infos[0,:,0], episode_len_means, label=exp_labels)
		ax[0].fill_between(exp_infos[0,:,0], episode_len_percentile_25, episode_len_percentile_75, alpha=0.2)
	ax[0].legend()
	plt.savefig("gridsearch_figure.png")

if __name__ == '__main__':
	plot_gridsearch_result(meta_file="gridsearch_reinforce_with_baseline_meta.txt", numpy_file="gridsearch_reinforce_with_baseline.npz")