# Gabor Function Fitting via Gradient Descent
# Author: Jay Hill (jdh1g19@soton.ac.uk)
# Date: 2023-04-29
# University of Southampton

from math import pi
import torch

class Gabor(torch.nn.Module):

  def __init__(self,
      n_kernels # Number of kernels (channels out).
    ):
    super(Gabor, self).__init__()

    self.n_kernels   = n_kernels

    # All parameters required to fit the Gabor.
    self.theta = torch.nn.parameter.Parameter(
      2 * pi * torch.rand(
        size = (self.n_kernels, 1, 1), dtype = torch.float32
      ) - 1
    )
    self.freq = torch.nn.parameter.Parameter(
      torch.normal(
        3, 0.5,
        size = (self.n_kernels, 1, 1),
        dtype = torch.float32
      )
    )
    self.phi = torch.nn.parameter.Parameter(
      torch.normal(
        0, 1,
        size = (self.n_kernels, 1, 1),
        dtype = torch.float32
      )
    )
    self.sigma_x = torch.nn.parameter.Parameter(
      torch.rand(size = (self.n_kernels, 1, 1), dtype = torch.float32)
    )
    self.sigma_y = torch.nn.parameter.Parameter(
      torch.rand(size = (self.n_kernels, 1, 1), dtype = torch.float32)
    )
    self.C = torch.nn.parameter.Parameter(
      torch.rand(size = (self.n_kernels, 1, 1), dtype = torch.float32)
    )

  def forward(self,
      x_range,
      y_range
    ):
    # Generate the x and y coordinates for each "pixel" of the kernel.
    x_mesh, y_mesh = torch.meshgrid(x_range, y_range)
    x_mesh         = x_mesh.unsqueeze(0)
    y_mesh         = y_mesh.unsqueeze(0)

    sin_theta, cos_theta = torch.sin(self.theta), torch.cos(self.theta)
    exp_scale            = 1 / (2 * pi * self.sigma_x * self.sigma_y)

    x_rot =  x_mesh * cos_theta + y_mesh * sin_theta
    y_rot = -x_mesh * sin_theta - y_mesh * cos_theta

    x_scale = x_rot ** 2 / self.sigma_x ** 2
    y_scale = y_rot ** 2 / self.sigma_y ** 2

    exp_part = torch.exp(-0.5 * (x_scale + y_scale))
    cos_part = torch.cos(2 * pi * self.freq + self.phi)

    final = exp_scale * exp_part * cos_part * self.C
    return final
  
  def get_parameters(self):
    return torch.stack(
      [param.detach()[:, 0, 0] for param in self.parameters()]
    )

def gabor_train_step(
    model,          # Gabor model.
    kernel_weights, # Kernel weights from 2D convolution.
    optimizer       # Optimizer.
  ):
  x_range = torch.arange(0, kernel_weights.shape[1], dtype = torch.float32)
  y_range = torch.arange(0, kernel_weights.shape[2], dtype = torch.float32)
  output = model(x_range, y_range)
  loss = torch.mean((output - kernel_weights) ** 2)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss.item()

def fit_gabor(
    kernel_weights,    # Kernel weights from 2D convolution.
    iterations = 1000, # Number of training iterations.
    optimizer  = None, # Existing optimizer.
    gabor      = None, # Existing Gabor model.
    verbose    = False # Whether to print loss on each iteration.
  ):
  if gabor is None:
    gabor = Gabor(kernel_weights.shape[0])
  if optimizer is None:
    optimizer = torch.optim.SGD(gabor.parameters(), lr = 10.0)
  for i in range(iterations):
    loss = gabor_train_step(gabor, kernel_weights, optimizer)
    if verbose:
      print("\rIter. ", i, ": Loss. ", loss, sep = "", end = "")
  if verbose:
    print()
  return gabor

def fit_gabors(
    kernel_weights,    # Kernel weights from 2D convolution (including input).
    iterations = 1000, # Number of training iterations.
    optim_gen  = None, # Existing optimizer generator.
    gabors     = None, # Existing Gabor models.
    verbose    = False # Whether to print loss on each iteration.
  ):
  if gabors is None:
    gabors = [None] * kernel_weights.shape[1]
  if optim_gen is None:
    def optim_gen(_):
      return None
  for index in range(kernel_weights.shape[1]):
    gabor     = gabors[index]
    optimizer = optim_gen(gabor)
    gabor = fit_gabor(
      kernel_weights[:, index, :, :],
      iterations = iterations,
      optimizer  = optimizer,
      gabor      = gabor,
      verbose    = verbose
    )
    gabors[index] = gabor
  return gabors