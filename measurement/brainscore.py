# Brain-Score model commitment for CORnet-S.
# Code Author: Jay Hill (jdh1g19@soton.ac.uk)
#              AND
#              Martin Schrimpf <1>
# Model Authors: Kubilius, J. et al. [1, 2]
# Date: 2023-05-13
# University of Southampton

from brainio.assemblies import merge_data_arrays, NeuroidAssembly, walk_coords
from brainscore import score_model as score_model_function
from brainscore.submission.utils import UniqueKeyDict
from brainscore.utils import LazyLoad
from collections import defaultdict, OrderedDict
from model_tools.activations.core import ActivationsExtractorHelper
from model_tools.activations.pytorch import PytorchWrapper, \
                                            load_preprocess_images
from model_tools.brain_transformation import ModelCommitment
from model_tools.brain_transformation.temporal import fix_timebin_naming
from tqdm import tqdm

import numpy as np
import functools
import re

# Time mappings for CORnet-S as given by <1>.
# I'm not entirely sure where this comes from, but it appears to have
# some relation to the brain measurements.
TIME_MAPPING = {
  "V1": ( 50, 100, 1),
  "V2": ( 70, 100, 2),
  "V4": ( 90,  50, 4),
  "IT": (100, 100, 2)
}

BUILT_TIME_MAPPING = {region: {
  timestep: (
    time_start +  timestep      * time_step_size,
    time_start + (timestep + 1) * time_step_size
  )
  for timestep in range(0, timesteps)}
  for region, (time_start, time_step_size, timesteps) in TIME_MAPPING.items()
}

# From <1>.
class CORnetSModelCommitment(ModelCommitment):

  def __init__(self,
      *args,
      **kwargs
    ):
    super(CORnetSModelCommitment, self).__init__(*args, **kwargs)
    self.time_mapping        = BUILT_TIME_MAPPING
    self.recording_layers    = None
    self.recording_time_bins = None
    for key, executor in self.behavior_model.mapping.items():
      executor.activations_model = TemporalIgnore(executor.activations_model)

  def start_recording(self,
      recording_target,
      time_bins
    ):
    self.recording_target = recording_target
    self.recording_layers = [
      layer for layer in self.layers if layer.startswith(recording_target)
    ]
    self.recording_time_bins = time_bins

  def look_at(self,
      stimuli,
      number_of_trials = 1
    ):
    if self.do_behavior:
      return super(CORnetSModelCommitment, self).look_at(
        stimuli,
        number_of_trials = number_of_trials
      )
    else:
      return self.look_at_temporal(stimuli=stimuli)

  def look_at_temporal(self,
      stimuli
    ):
    responses = self.activations_model(stimuli, layers = self.recording_layers)
    if hasattr(self, "recording_target"):
      regions = set([self.recording_target])
    else:
      regions = set(responses["region"].values)
    if len(regions) > 1:
      raise NotImplementedError(
        "Cannot handle more than one simultaneous region."
      )
    region    = list(regions)[0]
    time_bins = [
      self.time_mapping[region][timestep]
      if timestep in self.time_mapping[region] else (None, None)
      for timestep in responses["time_step"].values
    ]
    responses["time_bin_start"] = (
      "time_step",
      [time_bin[0] for time_bin in time_bins]
    )
    responses["time_bin_end"] = (
      "time_step",
      [time_bin[1] for time_bin in time_bins]
    )
    responses = NeuroidAssembly(responses.rename({"time_step": "time_bin"}))
    responses = responses[{
      "time_bin": [
        not np.isnan(time_start) for time_start in responses["time_bin_start"]
      ]
    }]
    time_responses = []
    for time_bin in tqdm(self.recording_time_bins, desc = "CORnet-S T-to-T"):
      time_bin = time_bin if not isinstance(time_bin, np.ndarray) \
                          else time_bin.tolist()
      time_bin_start, time_bin_end = time_bin
      nearest_start = find_nearest(
        responses["time_bin_start"].values,
        time_bin_start
      )
      bin_responses = responses.sel(time_bin_start = nearest_start)
      bin_responses = NeuroidAssembly(bin_responses.values,
        coords = {
          **{coord: (dims, values)
             for coord, dims, values in walk_coords(bin_responses)
             if coord not in ["time_bin_level_0", "time_bin_end"]
          },
          **{
            "time_bin_start": ("time_bin", [time_bin_start]),
            "time_bin_end"  : ("time_bin", [time_bin_end])
          }
        },
        dims = bin_responses.dims
      )
      time_responses.append(bin_responses)
      responses = merge_data_arrays(time_responses)
      responses = fix_timebin_naming(responses)
      return responses

class TemporalIgnore():

  def __init__(self,
      temporal_activations_model
    ):
    self._activations_model = temporal_activations_model

  def __call__(self,
      *args,
      **kwargs
    ):
    activations = self._activations_model(*args, **kwargs)
    activations = activations.squeeze("time_step")
    return activations

class TemporalPytorchWrapper(PytorchWrapper):

  def __init__(self,
      *args,
      separate_time = True,
      **kwargs
    ):
    self._separate_time = separate_time
    super(TemporalPytorchWrapper, self).__init__(*args, **kwargs)

  def _build_extractor(self,
      *args,
      **kwargs
    ):
    if self._separate_time:
      return TemporalExtractor(*args, **kwargs)
    else:
      return super(TemporalPytorchWrapper, self)._build_extractor(
        *args,
        **kwargs
      )

  def get_activations(self,
      images,
      layer_names
    ):
    self._layer_counter = defaultdict(lambda: 0)
    self._layer_hooks = {}
    return super(TemporalPytorchWrapper, self).get_activations(
      images      = images,
      layer_names = layer_names
    )

  def register_hook(self,
      layer,
      layer_name,
      target_dict
    ):
    layer_name = self._strip_layer_timestep(layer_name)
    if layer_name in self._layer_hooks:
      return self._layer_hooks[layer_name]

    def hook_function(
        _layer,
        _input,
        output
      ):
      target_dict[f"{layer_name}-t{self._layer_counter[layer_name]}"] = \
        PytorchWrapper._tensor_to_numpy(output)
      self._layer_counter[layer_name] += 1

    hook = layer.register_forward_hook(hook_function)
    self._layer_hooks[layer_name] = hook
    return hook

  def get_layer(self,
      layer_name
    ):
    layer_name = self._strip_layer_timestep(layer_name)
    return super(TemporalPytorchWrapper, self).get_layer(layer_name)

  def _strip_layer_timestep(self,
      layer_name
    ):
    match = re.search("-t[0-9]+$", layer_name)
    if match:
      layer_name = layer_name[:match.start()]
    return layer_name


class TemporalExtractor(ActivationsExtractorHelper):

  def from_paths(self,
      *args,
      **kwargs
    ):
    raw_activations = super(TemporalExtractor, self).from_paths(
      *args,
      **kwargs
    )
    regions = defaultdict(list)
    for layer in set(raw_activations["layer"].values):
      match = re.match(r"(([^-]*)\..*|logits|average_pool)-t([0-9]+)", layer)
      region, timestep = match.group(2) if match.group(2) \
                                        else match.group(1), match.group(3)
      stripped_layer = match.group(1)
      regions[region].append((layer, stripped_layer, timestep))
    activations = {}
    for region, time_layers in regions.items():
      for (full_layer, stripped_layer, timestep) in time_layers:
        region_time_activations = raw_activations.sel(layer = full_layer)
        region_time_activations["layer"] = (
          "neuroid",
          [stripped_layer] * len(region_time_activations["neuroid"])
        )
        activations[(region, timestep)] = region_time_activations
    for key, key_activations in activations.items():
      region, timestep = key
      key_activations["region"] = (
        "neuroid",
        [region] * len(key_activations["neuroid"])
      )
      activations[key] = NeuroidAssembly(
        [key_activations.values],
        coords = {
          **{coord: (dims, values)
             for coord, dims, values in walk_coords(activations[key])
             if coord != "neuroid_id"
          },
          **{"time_step": [int(timestep)]}
        },
        dims = ["time_step"] + list(key_activations.dims)
      )
    activations = list(activations.values())
    activations = merge_data_arrays(activations)
    neuroid_id = [
      ".".join([f"{value}" for value in values])
      for values in zip(
        *[activations[coord].values
        for coord in ["model", "region", "neuroid_num"]]
      )
    ]
    activations["neuroid_id"] = ("neuroid", neuroid_id)
    return activations

# Finds the nearest item in an array to a specified value by absolute
# difference.
def find_nearest(
    array,
    value
  ):
  array = np.asarray(array)
  idx   = (np.abs(array - value)).argmin()
  return array[idx]

def brain_model(
    model,
    image_size    = 224,
    separate_time = True,
    identifier    = "CORnet-S"
  ):
  preprocessing = functools.partial(
    load_preprocess_images,
    image_size = image_size
  )
  wrapper = TemporalPytorchWrapper(
    identifier    = identifier,
    model         = model,
    preprocessing = preprocessing,
    separate_time = separate_time
  )
  wrapper.image_size = image_size
  return CORnetSModelCommitment(
    identifier = identifier,
    activations_model = wrapper,
    layers = ["V1.final-t0"] + [
      f"{area}.final-t{timestep}"
      for area, timesteps in [
        ("V2", range(model.V2_recurrences)),
        ("V4", range(model.V4_recurrences)),
        ("IT", range(model.IT_recurrences))
      ]
      for timestep in timesteps
    ] + ["average_pool-t0"]
  )

def score(
    brain_model,
    benchmark,
    identifier = "CORnet-S"
  ):
  result = score_model_function(
    model_identifier     = identifier,
    benchmark_identifier = benchmark,
    model                = brain_model
  )
  return result

def score_benchmarks(
    brain_model,
    benchmarks,
    identifier = "CORnet-S"
  ):
  scores = OrderedDict()
  for benchmark in benchmarks:
    scores[benchmark] = score(
      brain_model,
      benchmark,
      identifier = identifier
    )
  return scores

def score_public_benchmarks(
    brain_model,
    identifier = "CORnet-S"
  ):
  return score_benchmarks(
    brain_model,
    [
      "movshon.FreemanZiemba2013public.V1-pls",
      "movshon.FreemanZiemba2013public.V2-pls",
      "dicarlo.MajajHong2015public.V4-pls",
      "dicarlo.MajajHong2015public.IT-pls",
      "dicarlo.Rajalingham2018public-i2n"
    ],
    identifier = identifier
  )

# Takes a set of scores and calculates a mean score.
# Specifically, it returns a raw mean, an adjusted mean (by the uncertainty)
# and the uncertainty of the mean.
def mean_score(
    scores
  ):
  centers = np.array([
    scores[x].sel(aggregation = "center").to_numpy() for x in scores
  ])
  errors = np.array([
    scores[x].sel(aggregation = "error").to_numpy() for x in scores
  ])
  sum_inv_var      = np.sum(1 / np.square(errors))
  adj_centers_mean = np.sum(centers / np.square(errors)) / sum_inv_var
  adj_errors_mean  = 1 / np.sqrt(sum_inv_var)
  raw_centers_mean = np.mean(centers)
  return (raw_centers_mean, adj_centers_mean, adj_errors_mean)
  

# SOURCES:
# <1>
#     Code has been adapted and inspired from:
#     https://github.com/brain-score/candidate_models
#     By Martin Schrimpf (mschrimpf), MIT.
#
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
