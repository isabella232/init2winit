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

"""Tests for the ogbg-molpcba dataset."""

import itertools

from init2winit.dataset_lib.datasets import get_dataset
from init2winit.dataset_lib.datasets import get_dataset_hparams
import jax.random
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

tf.enable_eager_execution()

NUM_LABELS = 2
NORMAL_LABELS = np.array([1, 1]).astype('float32')
NAN_LABELS = np.array([np.nan, 1]).astype('float32')

NUMS_NODES = [3, 7, 15, 6]
NUMS_EDGES = [4, 6, 20, 9]
LABELS = [NORMAL_LABELS, NAN_LABELS, NORMAL_LABELS, NAN_LABELS]


def _make_graph(num_nodes, num_edges, labels):
  return {
      'num_edges':
          np.array([num_edges]),
      'num_nodes':
          np.array([num_nodes]),
      'edge_index':
          np.array(
              list(
                  itertools.islice(
                      itertools.combinations(range(num_nodes), 2), num_edges))),
      'edge_feat':
          np.ones((num_edges, 3)).astype('float32'),
      'node_feat':
          np.ones((num_nodes, 9)).astype('float32'),
      'labels':
          labels
  }


def _as_dataset(*args, **kwargs):
  """Creates a mock TFDS graphs dataset."""
  del args, kwargs
  def get_iter():
    return (
        _make_graph(num_nodes, num_edges, labels)
        for num_nodes, num_edges, labels in zip(NUMS_NODES, NUMS_EDGES, LABELS))

  return tf.data.Dataset.from_generator(
      get_iter,
      output_signature={
          'edge_feat': tf.TensorSpec(shape=(None, 3), dtype=np.float32),
          'edge_index': tf.TensorSpec(shape=(None, 2), dtype=np.int64),
          'labels': tf.TensorSpec(shape=(NUM_LABELS,), dtype=np.float32),
          'node_feat': tf.TensorSpec(shape=(None, 9), dtype=np.float32),
          'num_edges': tf.TensorSpec(shape=(1,), dtype=np.int64),
          'num_nodes': tf.TensorSpec(shape=(1,), dtype=np.int64),
      })


def _get_dataset():
  """Loads the ogbg-molpcba dataset using mock data."""
  with tfds.testing.mock_data(as_dataset_fn=_as_dataset):
    ds = 'ogbg_molpcba'
    dataset_builder = get_dataset(ds)
    hps = get_dataset_hparams(ds)
    hps.train_size = 4
    hps.valid_size = 4
    hps.test_size = 4
    batch_size = 2
    eval_batch_size = 2
    dataset = dataset_builder(
        shuffle_rng=jax.random.PRNGKey(0),
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        hps=hps)
    return dataset


class OgbgMolpcbaTest(tf.test.TestCase):
  """Tests data loading for the ogbg-molpcba dataset."""

  def test_get_batch(self):
    dataset = _get_dataset()
    # Use validation batch to maintain the example order, since the train batch
    # will be shuffled.
    batch = next(dataset.valid_epoch())

    n_nodes = batch['inputs'].n_node
    self.assertNDArrayNear(n_nodes[:2], np.array(NUMS_NODES[:2]), 1e-3)
    # n_nodes should sum to the closest power of 2 + 1
    self.assertEqual(np.sum(n_nodes), 17)
    # Weights are zero at NaN labels and in padded examples
    self.assertNDArrayNear(batch['weights'], np.array([[1, 1], [0, 1], [0, 0]]),
                           1e-3)


if __name__ == '__main__':
  tf.test.main()
