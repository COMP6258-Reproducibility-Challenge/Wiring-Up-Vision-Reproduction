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
    x_expanded = x.unsqueeze(1)
    c_expanded = self.centroids.unsqueeze(0)
    lse = torch.logsumexp(
      torch.sum(
        (x_expanded - c_expanded) ** 2,
        dim = 2
        ) * self.beta,
      dim = 1
    )
    return torch.abs(lse / self.beta)

  def wcss(self,
      x
    ):
    #distances = torch.cdist(x, self.centroids.clone().detach())
    #mins      = torch.min(distances ** 2, dim = 1).values
    #return mins.sum().item()
    c = self.centroids.clone().detach()
    dist_total = 0
    print(c.shape)
    for d in range(x.shape[0]):
      dist = torch.sum((x[d] - c) ** 2, dim = 1)
      dist_total += torch.min(dist).item()
    return dist_total

  def get_parameters(self):
    return self.centroids.clone().detach()

def cluster_train_step(
    model,              # k-Means model.
    points,             # NxD matrix. N data points in D dimensions.
    optimizer,          # Optimizer, from torch.optim.
    use_closure = False # Use closure, if needed.
  ):
  def closure():
    output = model(points)
    loss = torch.mean(output)
    optimizer.zero_grad()
    loss.backward()
    return loss
  if use_closure:
    loss = optimizer.step(closure)
  else:
    loss = closure()
    optimizer.step()
  return loss.item()

def cluster(
    model,               # k-Means model.
    points,              # NxD matrix. N data points in D dimensions.
    optimizer,           # Optimizer, from torch.optim.
    iterations  = 1000,  # Number of training iterations.
    use_closure = False, # Use closure, if needed.
    verbose     = False  # Whether to print loss on each iteration.
  ):
  for i in range(iterations):
    loss = cluster_train_step(model, points, optimizer, use_closure)
    # Verbose will erase and rewrite the line to terminal. If this behavior
    # is *absolutely* necessary to remove, then remove 'end = ""'.
    if verbose:
      print("\rIter. ", i, ": Loss. ", loss, sep = "", end = "")
  if verbose:
    print()
  return loss
