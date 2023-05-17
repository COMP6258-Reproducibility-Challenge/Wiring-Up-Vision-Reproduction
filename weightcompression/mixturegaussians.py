# Multivariate Mixture of Gaussians Function Fitting via Gradient Descent
# Author: Jay Hill (jdh1g19@soton.ac.uk)
# Date: 2023-04-29
# University of Southampton

import torch

MN = torch.distributions.multivariate_normal.MultivariateNormal

class MixtureOfGaussians(torch.nn.Module):

  def __init__(self,
      n_data,
      n_components
    ):
    super(MixtureOfGaussians, self).__init__()

    self.n_data       = n_data
    self.n_components = n_components

    self.means = torch.nn.parameter.Parameter(
      torch.rand(
        size = (n_components, n_data),
        dtype = torch.float32
      )
    )
    self.cov_parts = torch.nn.parameter.Parameter(
      torch.eye(n_data).expand(n_components, -1, -1).clone()
    )
    self.log_weights = torch.nn.parameter.Parameter(
      torch.zeros(n_components)
    )

  def forward(self,
      data
    ):
    covariances = self.cov_parts @ torch.transpose(self.cov_parts, 1, 2)
    log_likelihoods = torch.stack(
      [
        MN(self.means[component], covariances[component]).log_prob(data)
        for component in range(self.n_components)
      ],
      dim = 1
    )
    weighted_likelihoods = log_likelihoods + self.log_weights ** 2
    # Mean is being used instead of sum to scale the loss down.
    log_likelihood = torch.mean(
      torch.logsumexp(weighted_likelihoods, dim = 1)
    )
    return log_likelihood
  
def mixture_train_step(
    model,       # Mixture of Gaussians model.
    data_matrix, # Data matrix.
    optimizer    # Optimizer.
  ):
  output = model(data_matrix)
  loss   = -output
  optimizer.zero_grad()
  loss.backward() #retain_graph = True)
  optimizer.step()
  return loss.item()

def fit_mixture(
    data_matrix,       # Data matrix.
    n_components,      # Number of Gaussian components.
    iterations = 1000, # Number of training iterations.
    optimizer  = None, # Existing optimizer.
    mixture    = None, # Existing mixture model.
    verbose    = False # Whether to print loss on each iteration.
  ):
  if mixture is None:
    mixture = MixtureOfGaussians(data_matrix.shape[1], n_components)
    print([name for name, _ in mixture.named_parameters()])
  if optimizer is None:
    optimizer = torch.optim.Adam(mixture.parameters(), lr = 0.1)
  for i in range(iterations):
    loss = mixture_train_step(mixture, data_matrix, optimizer)
    if verbose:
      print("\rIter. ", i, ": Loss. ", loss, sep = "", end = "")
  if verbose:
    print()
  return mixture