# Baseline Techniques for REINFORCE Algorithm in Policy Based Reinforcement Learning

Four techniques compared on [CartPole](https://gym.openai.com/envs/CartPole-v0/) environment:
1. REINFORCE with whitened returns
2. REINFORCE with learned learned value function as baseline
3. Self-critic with greedy rollout
4. Self-critic with sampled rollout

## Overview

TODO: What is in which file (shortly)

## Usage

The reported experiments in the blog can be reproduced by executing _gridsearch.py_, where we provide a function for each running a gridsearch for REINFORCE, REINFORCE with learned baseline and the self-critic approach. The used hyperparameters, as described in the blog, are:
* REINFORCE: Learning rates 4e-5,1e-4,4e-4,1e-3
* REINFORCE with learned baseline: Learning rates 2e-4,5e-4,1e-3, Alpha 0.1,0.25,0.5,0.75,0.9
* REINFORCE Self-critic: Learning rates 2e-3,5e-3,1e-2, Number of beams 1,2,4 (_for greedy only 1_), Log basis 2,3,4
The experiments were performed on a NVIDIA GeForce 1080Ti. Note that using a different GPU can lead to slightly different results due to the different number generators of the device.

The results of the gridsearch is stored in the directory _data_gridsearch_, where the subdirectory is by default based on the date and algorithm. We export the results of each run (episode length over time, number of interactions and number of update steps) in a compressed npz file, as well as the information about the experiments (i.e. which hyperparameters were used for each experiment). For plotting purposes, we take these two files as input and generate plots in the file _plotting.py_.
