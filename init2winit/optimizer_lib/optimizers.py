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

"""Getter function for selecting optimizers."""

import optax


def get_optimizer(hps):
  """Constructs the optax optimizer from the given HParams.

  We use optax.inject_hyperparams to wrap the optimizer transformations that
  accept learning rates. This allows us to "inject" the learning rate at each
  step in a training loop by manually setting it in the optimizer_state,
  calculating it using whatever (Python or Jax) logic we want.

  Args:
    hps: the experiment hyperparameters, as a ConfigDict.
  Returns:
    A tuple of the initialization and update functions returned by optax.
  """
  opt_init = None
  opt_update = None

  if 'weight_decay' in hps.opt_hparams:
    weight_decay = hps.opt_hparams['weight_decay']
  else:
    weight_decay = 0

  if hps.optimizer == 'sgd':
    opt_init, opt_update = optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
    )
  elif hps.optimizer == 'nesterov':
    # NOTE: in order to match the behavior of the Flax optimizers, we apply
    # weight decay **before** computing the Nesterov momentum update. This is
    # equivalent to applying WD after for heavy-ball momentum, but slightly
    # different for Nesterov.
    opt_init, opt_update = optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.inject_hyperparams(optax.sgd)(
            learning_rate=0.0,
            momentum=hps.opt_hparams['momentum'],
            nesterov=True),
    )
  elif hps.optimizer == 'momentum':
    opt_init, opt_update = optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.inject_hyperparams(optax.sgd)(
            learning_rate=0.0,
            momentum=hps.opt_hparams['momentum'],
            nesterov=False),
    )
  elif hps.optimizer == 'adam':
    assert hps.l2_decay_factor is None or weight_decay == 0.0
    opt_init, opt_update = optax.inject_hyperparams(optax.adamw)(
        learning_rate=0.0,
        b1=hps.opt_hparams['beta1'],
        b2=hps.opt_hparams['beta2'],
        eps=hps.opt_hparams['epsilon'],
        weight_decay=weight_decay)

  if opt_init is None or opt_update is None:
    raise NotImplementedError('Optimizer {} not implemented'.format(
        hps.optimizer))
  return opt_init, opt_update
