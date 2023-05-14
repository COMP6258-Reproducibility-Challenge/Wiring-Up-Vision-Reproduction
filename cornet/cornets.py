# CORnet-S Implementation from "CORnet: Modeling the Neural Mechanisms of Core
# Object Recognition" [1]
# Code Author: Jay Hill (jdh1g19@soton.ac.uk)
# Model Authors: Kubilius, J. et al. [1, 2]
# Date: 2023-04-05
# University of Southampton

from torch import nn

# V1 is different from the rest of the cortical areas.
class CORnetSV1(nn.Module):

  def __init__(self,
      channels_input,  # Number of channels input.
      channels_output, # Number of channels output.
    ):
    super(CORnetSV1, self).__init__()

    self.channels_input  = channels_input
    self.channels_output = channels_output

    self.conv1 = nn.Conv2d(
      self.channels_input, self.channels_output,
      kernel_size = (7, 7),
      stride      = (2, 2),
      padding     = 3,
      bias        = False
    )
    self.batchnorm1 = nn.BatchNorm2d(self.channels_output)
    # ReLU is safe to be inplace because negative values give dead gradients
    # and positive is identity.
    self.activation1 = nn.ReLU(inplace = True)

    self.maxpool = nn.MaxPool2d(
      (3, 3),
      stride  = (2, 2),
      padding = 1
    )

    self.conv2 = nn.Conv2d(
      self.channels_output, self.channels_output,
      kernel_size = (3, 3),
      padding     = 1,
      bias        = False
    )
    self.batchnorm2  = nn.BatchNorm2d(self.channels_output)
    self.activation2 = nn.ReLU(inplace = True)

    self.final = nn.Identity()

  def forward(self,
      inp,
    ):
    x = self.conv1(inp)
    x = self.batchnorm1(x)
    x = self.activation1(x)

    x = self.maxpool(x)

    x = self.conv2(x)
    x = self.batchnorm2(x)
    x = self.activation2(x)

    return self.final(x)
  
  def initialize(self):
    for module in self.modules():
      if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
      elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias,   0)
    
# All other cortical areas follow roughly the same pattern.
class CORnetSOther(nn.Module):
  
  def __init__(self,
      channels_input,  # Number of channels input.
      channels_output, # Number of channels output.
      recurrences = 1, # Variable number of recurrences for a cortical area.
      depth_scale = 4  # Scale increase in a cortical area. Defined as 4
                       # in [1].
    ):
    super(CORnetSOther, self).__init__()

    self.channels_input  = channels_input
    self.channels_output = channels_output
    self.recurrences     = recurrences
    self.depth_scale     = depth_scale

    channels_scaled = self.channels_output * self.depth_scale

    # Input convolution.
    self.conv0 = nn.Conv2d(
      self.channels_input, self.channels_output,
      kernel_size = (1, 1),
      bias = False
    )

    self.conv1 = nn.Conv2d(
      self.channels_output, channels_scaled,
      kernel_size = (1, 1),
      bias = False
    )
    # A different batch normalization needs to be performed on each
    # recurrence.
    for recurrence in range(self.recurrences):
      setattr(
        self,
        "batchnorm1_%d" % recurrence,
        nn.BatchNorm2d(channels_scaled)
      )
    self.activation1 = nn.ReLU(inplace = True)

    self.conv2 = nn.Conv2d(
      channels_scaled, channels_scaled,
      kernel_size = (3, 3),
      stride = (2, 2),
      padding = 1,
      bias = False
    )
    for recurrence in range(self.recurrences):
      setattr(
        self,
        "batchnorm2_%d" % recurrence,
        nn.BatchNorm2d(channels_scaled)
      )
    self.activation2 = nn.ReLU(inplace = True)

    self.conv3 = nn.Conv2d(
      channels_scaled, self.channels_output,
      kernel_size = (1, 1),
      bias = False
    )
    for recurrence in range(self.recurrences):
      setattr(
        self,
        "batchnorm3_%d" % recurrence,
        nn.BatchNorm2d(self.channels_output)
      )

    # Skip connection needs to downsample.
    self.conv_skip = nn.Conv2d(
      self.channels_output, self.channels_output,
      kernel_size = (1, 1),
      stride = (2, 2),
      bias = False
    )
    self.batchnorm_skip = nn.BatchNorm2d(self.channels_output)

    self.activation_final = nn.ReLU(inplace = True)
    self.final = nn.Identity()

  def forward(self,
      inp
    ):
    x = self.conv0(inp)

    for recurrence in range(self.recurrences):
      skip = x if recurrence > 0 else self.batchnorm_skip(self.conv_skip(x))

      x = self.conv1(x)
      x = getattr(self, "batchnorm1_%d" % recurrence)(x)
      x = self.activation1(x)

      self.conv2.stride = (1, 1) if recurrence > 0 else (2, 2)
      x = self.conv2(x)
      x = getattr(self, "batchnorm2_%d" % recurrence)(x)
      x = self.activation2(x)

      x = self.conv3(x)
      x = getattr(self, "batchnorm3_%d" % recurrence)(x)
      
      x = x + skip
      x = self.activation_final(x)
      final = self.final(x)
      
    return final
  
  def initialize(self):
    for module in self.modules():
      if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
      elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias,   0)
  
# Aliases.
CORnetSV2 = CORnetSV4 = CORnetSIT = CORnetSOther

# Full module as specified by [1].
class CORnetS(nn.Module):
  
  def __init__(self,
      channels_input = 3,
      classes_output = 1000,
      V1_channels    = 64,
      V2_channels    = 128,
      V4_channels    = 256,
      IT_channels    = 512,
      V2_recurrences = 2,
      V4_recurrences = 4,
      IT_recurrences = 2
    ):
    super(CORnetS, self).__init__()

    self.channels_input = channels_input
    self.classes_output = classes_output
    self.V1_channels    = V1_channels
    self.V2_channels    = V2_channels
    self.V4_channels    = V4_channels
    self.IT_channels    = IT_channels
    self.V2_recurrences = V2_recurrences
    self.V4_recurrences = V4_recurrences
    self.IT_recurrences = IT_recurrences

    self.V1 = CORnetSV1(self.channels_input, self.V1_channels)
    self.V2 = CORnetSV2(
      self.V1_channels, self.V2_channels,
      self.V2_recurrences
    )
    self.V4 = CORnetSV4(
      self.V2_channels, self.V4_channels,
      self.V4_recurrences
    )
    self.IT = CORnetSIT(
      self.V4_channels, self.IT_channels,
      self.IT_recurrences
    )

    # Final block to get classes.
    self.average_pool = nn.AdaptiveAvgPool2d(1)
    self.flatten      = nn.Flatten()
    self.linear       = nn.Linear(self.IT_channels, self.classes_output)

    self.final = nn.Identity()

  def forward(self,
      inp
    ):
    x = self.V1(inp)
    x = self.V2(x)
    x = self.V4(x)
    x = self.IT(x)

    x = self.average_pool(x)
    x = self.flatten(x)
    x = self.linear(x)

    return self.final(x)
  
  def initialize(self):
    self.V1.initialize()
    self.V2.initialize()
    self.V4.initialize()
    self.IT.initialize()
    nn.init.xavier_normal_(self.linear.weight)
    nn.init.constant_(self.linear.bias, 0)
   
# This replaces any OLD checkpoint state dictionaries with the new format.
# (For anyone wondering, the previous version of this code gave some
# values the wrong names).
REPLACEMENTS = {
  "v1": "V1",
  "v2": "V2",
  "v3": "V4",
  "it": "IT"
}
def update_model_state_dict(
    model_state_dict
  ):
  keys_to_change = []
  for key in model_state_dict:
    for replace_key in REPLACEMENTS:
      if key.startswith(replace_key):
        replacement = key.replace(replace_key, REPLACEMENTS[replace_key], 1)
        keys_to_change.append((replacement, key))
  for (replacement, key) in keys_to_change:
    model_state_dict[replacement] = model_state_dict.pop(key)

# REFERENCES:
# [1]
#     Jonas Kubilius, Martin Schrimpf, Aran Nayebi, Daniel Bear,
#     Daniel Yamins, James DiCarlo.
#     CORnet: Modeling the Neural Mechanisms of Core Object Recognition.
#     09 2018.
#     Available at: https://www.biorxiv.org/content/10.1101/408385v1.full.pdf
# [2]
#     Jonas Kubilius, Martin Schrimpf, Ha Hong, Najib Majaj,
#     Rishi Rajalingham, Elias Issa, Kohitij Kar, Pouya Bashivan,
#     Jonathan Prescott-Roy, Kailyn Schmidt, Aran Nayebi, Daniel Bear,
#     Daniel Yamins, and James DiCarlo.
#     Brain-Like Object Recognition with High-Performing Shallow Recurrent
#     ANNs. 09 2019.
#     Available at: https://arxiv.org/pdf/1909.06161.pdf

# Code was compared and checked against https://github.com/dicarlolab/CORnet
# to ensure model consistency.
