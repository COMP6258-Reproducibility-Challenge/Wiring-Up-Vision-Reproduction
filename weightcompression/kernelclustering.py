# Clustering Kernel Weights using k-Means
# Author: Jay Hill (jdh1g19@soton.ac.uk)
# Date: 2023-04-29
# University of Southampton

from weightcompression.kmeanslse import *

from sklearn.cluster import KMeans
import torch


class KernelCluster():

  def __init__(self,
      n_clusters # Number of clusters, k.
    ):
    self.n_clusters = n_clusters
    self.parameters = None
    self.closest    = None

  def get_clusters_from_weights(self,
      kernel_weights,
      iterations = 300
    ):
    k1 = torch.squeeze(kernel_weights)
    k2 = k1.reshape(k1.shape[0],  k1.shape[1], -1)
    k3 = k2.reshape(k2.shape[0] * k2.shape[1], -1)
    kmeans = KMeans(
      n_clusters   = self.n_clusters,
      n_init       = "auto",
      random_state = 42,
      max_iter     = iterations
    )
    kmeans.fit(k3.detach().numpy())
    self.parameters = kmeans.cluster_centers_
    self.closest    = kmeans.labels_
    return kmeans.inertia_

  def perform_elbow(self,
      kernel_weights,
      min_clusters =  1,
      max_clusters = 10,
      repeats      =  1
    ):
    losses = []
    for n_clusters in range(min_clusters, max_clusters):
      self.n_clusters = n_clusters
      loss = self.get_clusters_from_weights(
        kernel_weights
      )
      losses.append(loss)
    return losses
