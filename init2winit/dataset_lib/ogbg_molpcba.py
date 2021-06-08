# coding=utf-8
# Copyright 2021 The init2winit Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fake image input pipeline. Returns the same batch of ones over and over."""

import itertools

from init2winit.dataset_lib import data_utils
from init2winit.dataset_lib.data_utils import Dataset
import jax
import jraph
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow_datasets as tfds

DEFAULT_HPARAMS = config_dict.ConfigDict(
    dict(
        output_shape=(128,),
        input_edge_shape=(None, 3),
        input_node_shape=(None, 9),
        batch_size=512,
        model_dtype='float32',
        train_size=350343,
        valid_size=43793,
        test_size=43793))

METADATA = {
    'apply_one_hot_in_loss': False,
}


def _load_dataset(split, shuffle_seed=None, shuffle_buffer_size=2**15):
  """Loads a dataset split from TFDS."""
  is_train = split == 'train'
  read_config = tfds.ReadConfig(add_tfds_id=True, shuffle_seed=shuffle_seed)
  dataset = tfds.load(
      'ogbg_molpcba',
      split=split,
      shuffle_files=is_train,
      read_config=read_config)

  if is_train:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.repeat()

  return dataset


def _to_jraph(example):
  """Converts an example graph to jraph.GraphsTuple."""
  example = data_utils.tf_to_numpy(example)
  edge_feat = example['edge_feat']
  node_feat = example['node_feat']
  edge_index = example['edge_index']
  labels = example['labels']
  num_nodes = example['num_nodes']

  senders = edge_index[:, 0]
  receivers = edge_index[:, 1]

  return jraph.GraphsTuple(
      n_node=num_nodes,
      n_edge=np.array([len(edge_index) * 2]),
      nodes=node_feat,
      edges=np.concatenate([edge_feat, edge_feat]),
      # Make the edges bidirectional
      senders=np.concatenate([senders, receivers]),
      receivers=np.concatenate([receivers, senders]),
      # The globals will contain the final prediction.
      # Add an extra dimension because batching graphs will concatenate them.
      globals=np.expand_dims(np.zeros_like(labels), axis=0)), labels


def _nearest_bigger_power_of_two(x):
  return 2**int(x).bit_length()


def _get_weights_by_nan_and_padding(labels, padding_mask):
  """Creates weights by replacing NaN labels by 0 and setting a corresponding label weight to 0, and setting 0 to labels in examples which are padding."""
  nan_mask = np.isnan(labels)
  np.nan_to_num(labels)

  weights = (~nan_mask).astype(np.float32)
  # Weights for all labels of a padded element will be 0
  weights = weights * padding_mask[:, None]
  return weights


def _pad_graph(graphs, labels, num_shards=None):
  """Batches and pads the examples.

  Since each graph has a different number of nodes and edges, we pad each
  batch to the closest power of 2, so that we can rely on the cache while
  JITing. We also have to make sure it's cleanly divisible by num_shards, so
  that each device gets separate graphs when using pmap, so divide into
  num_shards segments and pad each of them separately to the highest required
  power of 2.

  Args:
    graphs: A list of graphs to be batched.
    labels: Corresponding labels.
    num_shards: How many shards the data will be sharded into.

  Returns:
    A tuple of (jraph.GraphsTuple, labels, per-label weights)
  """
  if not num_shards:
    num_shards = jax.device_count()

  assert len(graphs) % num_shards == 0

  chunk_size = len(graphs) // num_shards
  batched_graphs = [
      jraph.batch(graphs[i:i + chunk_size])
      for i in range(0, len(graphs), chunk_size)
  ]
  batched_labels = [
      labels[i:i + chunk_size] for i in range(0, len(labels), chunk_size)
  ]

  max_n_node = np.max([np.sum(graph.n_node) for graph in batched_graphs])
  max_n_edge = np.max([np.sum(graph.n_edge) for graph in batched_graphs])

  new_n_node = _nearest_bigger_power_of_two(max_n_node) + 1
  new_n_edge = _nearest_bigger_power_of_two(max_n_edge)
  new_n_graph = chunk_size + 1

  padded_graphs = [
      jraph.pad_with_graphs(graph, new_n_node, new_n_edge, new_n_graph)
      for graph in batched_graphs
  ]
  # We add one extra graph in each shard for padding, so add a corresponding
  # label.
  padded_labels = np.concatenate([
      np.stack(labels + [np.zeros_like(labels[0])], axis=0)
      for labels in batched_labels
  ])

  weights = _get_weights_by_nan_and_padding(
      padded_labels,
      np.concatenate(
          [jraph.get_graph_padding_mask(graph) for graph in padded_graphs]))

  padded_batched_graph = jraph.batch(padded_graphs)

  return padded_batched_graph, padded_labels, weights


def _get_batch_iterator(dataset_iter, batch_size):
  """Turns a TFDS per-example iterator into a batched iterator in the init2winit format."""
  graphs = []
  labels = []
  count = 0
  for example in dataset_iter:
    count += 1
    graph, label = _to_jraph(example)
    graphs.append(graph)
    labels.append(label)
    if count == batch_size:
      padded_graph, padded_labels, weights = _pad_graph(graphs, labels)

      yield {
          'inputs': padded_graph,
          'targets': padded_labels,
          'weights': weights
      }

      count = 0
      graphs = []
      labels = []


def get_ogbg_molpcba(shuffle_rng, batch_size, eval_batch_size, hps=None):
  """Data generators for ogbg-molpcba."""
  del hps
  train_ds = _load_dataset('train', shuffle_rng)
  valid_ds = _load_dataset('validation')
  test_ds = _load_dataset('test')

  def train_iterator_fn():
    return _get_batch_iterator(iter(train_ds), batch_size)

  def eval_train_epoch(num_batches=None):
    return itertools.islice(
        _get_batch_iterator(iter(train_ds), batch_size), num_batches)

  def valid_epoch(num_batches=None):
    return itertools.islice(
        _get_batch_iterator(iter(valid_ds), eval_batch_size), num_batches)

  def test_epoch(num_batches=None):
    return itertools.islice(
        _get_batch_iterator(iter(test_ds), eval_batch_size), num_batches)

  return Dataset(train_iterator_fn, eval_train_epoch, valid_epoch, test_epoch)
