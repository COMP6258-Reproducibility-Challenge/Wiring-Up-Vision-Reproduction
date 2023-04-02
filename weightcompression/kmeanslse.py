# Differentiable k-Means Clustering via LogSumExp
# Author: Jay Hill (jdh1g19@soton.ac.uk)
# Date: 2023-04-02
# University of Southampton

import torch

# Inheriting from "nn.Module" for simplicity of optimization.
class KMeansLSE(torch.nn.Module):

  def __init__(self,
      dim,          # Number of dimensions.
      clusters,     # Number of clusters.
      c_mean = 0.0, # Mean for random initialization of centroids.
      c_std  = 1.0, # St. deviation for random initialization of centroids.
      beta   = -1   # Beta. More negative = better approximation, but harder
                    # to train.
    ):
    super(KMeansLSE, self).__init__()

    self.beta     = beta
    self.clusters = clusters
    self.dim      = dim

    # Initialize centroids as parameters.
    self.centroids = torch.nn.parameter.Parameter(
      torch.normal(
        c_mean,
        c_std,
        size = (self.clusters, self.dim)
      )
    )

  def forward(self,
      x # Input: NxD matrix. N data points in D dimensions.
    ):
    squared_norm = torch.norm(
      # Expand dimensions to perform an outer-product-esque subtraction.
      # (Basically each point subtracted by each centroid.)
      x[:, None, :] - self.centroids[None, :, :],
      dim = 2
    ) ** 2
    lse = torch.logsumexp(self.beta * squared_norm, 1)
    return lse / self.beta
  
def train_step(
    model,    # k-Means model.
    points,   # NxD matrix. N data points in D dimensions.
    optimizer # Optimizer, from torch.optim.
  ):
  output = model(points)
  loss = torch.norm(output)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss.item()

def cluster(
    model,             # k-Means model.
    points,            # NxD matrix. N data points in D dimensions.
    optimizer,         # Optimizer, from torch.optim.
    iterations = 1000, # Number of training iterations.
    verbose = False    # Whether to print loss on each iteration.
  ):
  for i in range(iterations):
    loss = train_step(model, points, optimizer)
    # Verbose will erase and rewrite the line to terminal. If this behavior
    # is *absolutely* necessary to remove, then remove 'end = ""'.
    if verbose:
      print("\rIter. ", i, ": Loss. ", loss, sep = "", end = "")
  if verbose:
    print()
