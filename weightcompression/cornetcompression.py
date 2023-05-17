# Assigning Compressed Weights to CORnet-S
# Author: Jay Hill (jdh1g19@soton.ac.uk)
# Date: 2023-05-08
# University of Southampton

import torch

import numpy as np

def assign_kernel(layer, config):
  weights = layer.weight
  centroids     = config["centroids"]
  cluster_stats = config["cluster-stats"]
  weights_stats = config["weights-stats"]
  full_kernel = np.zeros(list(weights.shape))
  for idx_kernel in range(weights.shape[0]):
    cluster_selection = [
      np.floor(0.5 + np.abs(np.random.normal(stat[0], stat[1]))).astype(int)
      for stat in cluster_stats
    ]
    new_kernel = np.zeros(weights.shape[1:])
    pointer = 0
    for idx, selection in enumerate(cluster_selection):
      centroid = centroids[idx]
      for times in range(selection):
        if pointer >= new_kernel.shape[0]: break
        new_kernel[pointer] = np.random.normal(
          centroid,
          weights_stats[idx],
          centroid.shape
        ).reshape(new_kernel.shape[1:])
        pointer += 1
    for idx in range(pointer + 1, new_kernel.shape[0]):
      new_kernel[idx] = np.random.normal(
        centroids[-1],
        weights_stats[-1],
        centroids[-1].shape
      ).reshape(new_kernel.shape[1:])
    permutations = np.random.permutation(new_kernel.shape[0])
    new_kernel = new_kernel[permutations]
    full_kernel[idx_kernel] = new_kernel
  weights.data = torch.from_numpy(full_kernel)

INIT_TYPE = {
  "conv" : assign_kernel,
  #"gabor": assign_gabor,
  #"norm" : assign_batchnorm
}

def assign_inits(model, config):
  model_layers = dict(model.named_modules())
  config_names = [name for name in config]
  for layer_name in model_layers:
    print(layer_name)
    if layer_name in config_names:
      layer_config = config[layer_name]
      INIT_TYPE[layer_config["type"]](model_layers[layer_name], layer_config)

