import torch
import numpy as np 
import os
import imageio

from lunar_training import * 
from train_utils import * 
from LunarLanderEnv import *


def run_action_episode(env, model, select_action=select_action_default, 
					   greedy_actions=False, render=False, 
					   beams_num=-1, beam_freq=10, beams_greedy=False,
					   discount_factor=0.99, last_state=None):

	call_beam = lambda state : run_action_episode(deepcopy(env), model, select_action, 
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
	while not done:
		beams_actions = None
		if beams_num > 0 and frame_iter % beam_freq == 0: # It is intended to be true also at step 0
			# print("Running beams at frame iteration %i" % frame_iter)
			beams_actions = [call_beam(s) for _ in range(beams_num)]
			beams_actions = [([e["action"] for e in episode]+[a["action"] for a in b]) for b in beams_actions]
		action, log_p = select_action(model, s)
		s_next, r, done, _ = env.step(action)
		if render:
			env.render()
			time.sleep(0.05)
		episode += [{"action": action, "beams": beams_actions}]
		s = s_next
		frame_iter += 1
	if render:
		env.close()
	return episode


def render_action_episode(env, seed, action_list):
	env.seed(seed)
	env.reset()
	img_list = []
	img_list.append(env.render("rgb_array"))
	episode = []
	for a in action_list:
		env.step(a)
		img_list.append(env.render("rgb_array"))
	env.close()
	imgs = np.stack(img_list, axis=0).astype(np.uint8)
	imgs = imgs[:,150:320]
	return imgs


def visualize_treefork_algorithm(env, model, select_action=select_action_default, 
								 greedy_actions=False, seed=42,
								 beams_num=-1, beam_freq=10, beams_greedy=False,
								 discount_factor=0.99, last_actions=None):

	env.seed(seed)
	episode = run_action_episode(env=env, model=model, beams_num=beams_num, beam_freq=beam_freq)
	imgs = render_action_episode(env=env, seed=seed, action_list=[e["action"] for e in episode])
	beam_imgs = {}
	for e_index, e in enumerate(episode):
		if e["beams"] is not None:
			print("Visualizing beams at iteration %i..." % (e_index))
			beam_imgs["beams_%i" % e_index] = [render_action_episode(env=env, seed=seed, action_list=b)[e_index:] for b in e["beams"]]
	return imgs, beam_imgs


def imgs_to_gif(imgs, beam_imgs, filename, sequential_beams=False):
	img_list = []
	for i in range(imgs.shape[0]):
		img_list.append(imgs[i])
		k = ("beams_%i" % i)
		if k in beam_imgs:
			if sequential_beams:
				for b in beam_imgs[k]:
					img_list += [(imgs[i] * 0.5 + b[j]*0.5) for j in range(b.shape[0])]
			else:
				beam_prop = 1.0/(len(beam_imgs[k]) + 2)
				orig_prop = 2*beam_prop
				bimgs = [imgs[i]*orig_prop] * max([b.shape[0] for b in beam_imgs[k]])
				print("Bimg length: ", len(bimgs))
				for b in beam_imgs[k]:
					print("B shape: ", b.shape[0])
					bimgs = [(bimgs[j] + beam_prop*b[min(j, b.shape[0]-1)]) for j in range(len(bimgs))]
				img_list += bimgs
	img_list = [img.astype(np.uint8) for img in img_list]
	imageio.mimsave(filename, img_list, duration = 0.03)



if __name__ == '__main__':
	if True or not os.path.isfile("visualization_imgs.npz"):
		set_seed(seed=42)
		env = gym.make("CartPole-v1") # gym.make("LunarLander-v2")
		env.seed(42)
		model = LunarLinearPolicyNetwork(num_inputs=4, num_actions=2)
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
		train(env=env, model=model, optimizer=optimizer, num_episodes=70, loss_fun=compute_reinforce_with_baseline_fork_update_loss, discount_factor=0.99, final_render=False)
		imgs, beam_imgs = visualize_treefork_algorithm(env=env, model=model, beams_num=4, beam_freq=80, seed=42)
		np.savez_compressed("visualization_imgs.npz", imgs=imgs, **beam_imgs, allow_pickle=True)
	loaded_imgs = np.load("visualization_imgs.npz", allow_pickle=True)
	for k in loaded_imgs:
		print(k)
	loaded_imgs = {k:loaded_imgs[k] for k in loaded_imgs}
	imgs = loaded_imgs.pop("imgs")
	beam_imgs = loaded_imgs
	imgs_to_gif(imgs, beam_imgs, "visualization_gif_seq.gif", sequential_beams=True)
	imgs_to_gif(imgs, beam_imgs, "visualization_gif_par.gif", sequential_beams=False)