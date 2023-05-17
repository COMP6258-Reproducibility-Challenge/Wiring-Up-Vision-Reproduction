# Clustering Kernel Weights using k-Means
# Author: Jay Hill (jdh1g19@soton.ac.uk)
# Date: 2023-04-29
# University of Southampton

from weightcompression.kmeanslse import *

from sklearn.cluster import KMeans
import torch
import numpy as np

# Clusters the convolutional layers of a CORnet-S model.
def cluster_kernels(cornet):
  # Predefined locations of layers and their cluster sizes.
  layer_dict = {
    "V1.conv2"    : (cornet.V1.conv2,     7),
    "V2.conv0"    : (cornet.V2.conv0,     4),
    "V2.conv1"    : (cornet.V2.conv1,     5),
    "V2.conv2"    : (cornet.V2.conv2,     7),
    "V2.conv3"    : (cornet.V2.conv3,     5),
    "V2.conv_skip": (cornet.V2.conv_skip, 4),
    "V4.conv0"    : (cornet.V4.conv0,     4),
    "V4.conv1"    : (cornet.V4.conv1,     5),
    "V4.conv2"    : (cornet.V4.conv2,     9),
    "V4.conv3"    : (cornet.V4.conv3,     5),
    "V4.conv_skip": (cornet.V4.conv_skip, 5),
    "IT.conv0"    : (cornet.IT.conv0,     4),
    "IT.conv1"    : (cornet.IT.conv1,     4),
    "IT.conv2"    : (cornet.IT.conv2,     7),
    "IT.conv3"    : (cornet.IT.conv3,     5),
    "IT.conv_skip": (cornet.IT.conv_skip, 4)
  }
  cluster_info = {}
  for name in layer_dict:
    print("Clustering %s..." % name)
    cluster_info[name] = {"type": "conv"}
    # Weight adjustment.
    kernel_weights = layer_dict[name][0].weight.clone().detach()
    kernel_weights.requires_grad = False
    kernel_weights = torch.squeeze(kernel_weights)
    weight_matrix = kernel_weights.reshape(
      kernel_weights.shape[0],
      kernel_weights.shape[1],
      -1
    )
    weight_matrix = weight_matrix.reshape(
      weight_matrix.shape[0] * weight_matrix.shape[1],
      -1
    ).detach().numpy()
    # Clustering.
    kmeans = KMeans(
      n_clusters   = layer_dict[name][1],
      n_init       = "auto",
      random_state = 42
    )
    kmeans.fit(weight_matrix)
    cluster_info[name]["centroids"] = torch.tensor(kmeans.cluster_centers_)
    # Frequency of clusters.
    total_cluster = np.max(kmeans.labels_) + 1
    freqs = {x: [] for x in range(total_cluster)}
    labels_reshape = kmeans.labels_.reshape(
      kernel_weights.shape[0],
      kernel_weights.shape[1]
    )
    for closest_cluster in labels_reshape:
      bins = np.bincount(closest_cluster, minlength = total_cluster)
      for idx, bincount in enumerate(bins):
        freqs[idx] += [bincount]
    cluster_stats = np.array(
      [[np.mean(freqs[x]), np.std(freqs[x])] for x in freqs]
    )
    cluster_info[name]["cluster-stats"] = torch.tensor(cluster_stats)
    # Distribution of kernel weights.
    sigmas = {x: [] for x in range(total_cluster)}
    stds = np.std(
      weight_matrix - kmeans.cluster_centers_[kmeans.labels_],
      axis = 1
    )
    for idx, std in enumerate(stds):
      sigmas[kmeans.labels_[idx]] += [std]
    weights_stats = np.array([np.mean(sigmas[x]) for x in sigmas])
    cluster_info[name]["weights-stats"] = torch.tensor(weights_stats)
  return cluster_info
