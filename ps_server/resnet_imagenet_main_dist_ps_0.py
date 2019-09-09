# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
from absl import logging
import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers
from official.vision.image_classification import common
from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import resnet_model
from official.vision.image_classification import trivial_model

import os
import json
tf.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CONFIG"] = json.dumps({
    'cluster': {
        'ps': ["9.73.165.158:1111"],
        'worker':
            ["9.73.165.158:1112", "9.73.165.158:1113", "9.73.165.158:1114", "9.73.136.185:1111", "9.73.136.185:1112",
             "9.73.136.185:1113", "9.73.136.185:1114", "9.73.169.29:1111", "9.73.169.29:1112", "9.73.169.29:1113",
             "9.73.169.29:1114", "9.73.165.16:1111", "9.73.165.16:1112", "9.73.165.16:1113", "9.73.165.16:1114"]
    },
    'task': {'type': 'ps', 'index': 0}
    # {'type': 'worker', 'index': 0}
})

LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


def learning_rate_schedule(current_epoch,
                           current_batch,
                           batches_per_epoch,
                           batch_size):
    """Handles linear scaling rule, gradual warmup, and LR decay.

    Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
    provided scaling factor.

    Args:
      current_epoch: integer, current epoch indexed from 0.
      current_batch: integer, current batch in the current epoch, indexed from 0.
      batches_per_epoch: integer, number of steps in an epoch.
      batch_size: integer, total batch sized.

    Returns:
      Adjusted learning rate.
    """
    initial_lr = common.BASE_LEARNING_RATE * batch_size / 256
    epoch = current_epoch + float(current_batch) / batches_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = LR_SCHEDULE[0]
    if epoch < warmup_end_epoch:
        # Learning rate increases linearly per step.
        return initial_lr * warmup_lr_multiplier * epoch / warmup_end_epoch
    for mult, start_epoch in LR_SCHEDULE:
        if epoch >= start_epoch:
            learning_rate = initial_lr * mult
        else:
            break
    return learning_rate


def run(flags_obj):
    """Run ResNet ImageNet training and eval loop using native Keras APIs.

    Args:
      flags_obj: An object containing parsed flag values.

    Raises:
      ValueError: If fp16 is passed as it is not currently supported.

    Returns:
      Dictionary of training and eval stats.
    """
    keras_utils.set_session_config(
        enable_eager=flags_obj.enable_eager,
        enable_xla=flags_obj.enable_xla)

    # Execute flag override logic for better model performance
    if flags_obj.tf_gpu_thread_mode:
        common.set_gpu_thread_mode_and_count(flags_obj)
    if flags_obj.data_delay_prefetch:
        common.data_delay_prefetch()
    common.set_cudnn_batchnorm_mode()

    dtype = flags_core.get_tf_dtype(flags_obj)
    if dtype == 'float16':
        policy = tf.keras.mixed_precision.experimental.Policy('infer_float32_vars')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    data_format = flags_obj.data_format
    if data_format is None:
        data_format = ('channels_first'
                       if tf.test.is_built_with_cuda() else 'channels_last')
    tf.keras.backend.set_image_data_format(data_format)

    # Configures cluster spec for distribution strategy.
    num_workers = distribution_utils.configure_cluster(flags_obj.worker_hosts,
                                                       flags_obj.task_index)

    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=flags_obj.num_gpus,
        num_workers=num_workers,
        all_reduce_alg=flags_obj.all_reduce_alg,
        num_packs=flags_obj.num_packs)

    if strategy:
        # flags_obj.enable_get_next_as_optional controls whether enabling
        # get_next_as_optional behavior in DistributedIterator. If true, last
        # partial batch can be supported.
        strategy.extended.experimental_enable_get_next_as_optional = (
            flags_obj.enable_get_next_as_optional
        )

    strategy_scope = distribution_utils.get_strategy_scope(strategy)

    # pylint: disable=protected-access
    if flags_obj.use_synthetic_data:
        distribution_utils.set_up_synthetic_data()
        input_fn = common.get_synth_input_fn(
            height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
            width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
            num_channels=imagenet_preprocessing.NUM_CHANNELS,
            num_classes=imagenet_preprocessing.NUM_CLASSES,
            dtype=dtype,
            drop_remainder=True)
    else:
        distribution_utils.undo_set_up_synthetic_data()
        input_fn = imagenet_preprocessing.input_fn

    # When `enable_xla` is True, we always drop the remainder of the batches
    # in the dataset, as XLA-GPU doesn't support dynamic shapes.
    drop_remainder = flags_obj.enable_xla

    train_input_dataset = input_fn(
        is_training=True,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        num_epochs=flags_obj.train_epochs,
        parse_record_fn=imagenet_preprocessing.parse_record,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads,
        dtype=dtype,
        drop_remainder=drop_remainder,
        tf_data_experimental_slack=flags_obj.tf_data_experimental_slack,
    )

    eval_input_dataset = None
    if not flags_obj.skip_eval:
        eval_input_dataset = input_fn(
            is_training=False,
            data_dir=flags_obj.data_dir,
            batch_size=flags_obj.batch_size,
            num_epochs=flags_obj.train_epochs,
            parse_record_fn=imagenet_preprocessing.parse_record,
            dtype=dtype,
            drop_remainder=drop_remainder)

    lr_schedule = 0.1
    if flags_obj.use_tensor_lr:
        lr_schedule = common.PiecewiseConstantDecayWithWarmup(
            batch_size=flags_obj.batch_size,
            epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
            warmup_epochs=LR_SCHEDULE[0][1],
            boundaries=list(p[1] for p in LR_SCHEDULE[1:]),
            multipliers=list(p[0] for p in LR_SCHEDULE),
            compute_lr_on_cpu=True)

    with strategy_scope:
        optimizer = common.get_optimizer(lr_schedule)
        if dtype == 'float16':
            # TODO(reedwm): Remove manually wrapping optimizer once mixed precision
            # can be enabled with a single line of code.
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                optimizer, loss_scale=flags_core.get_loss_scale(flags_obj,
                                                                default_for_fp16=128))

        if flags_obj.use_trivial_model:
            model = trivial_model.trivial_model(
                imagenet_preprocessing.NUM_CLASSES, dtype)
        else:
            model = resnet_model.resnet50(
                num_classes=imagenet_preprocessing.NUM_CLASSES, dtype=dtype)

        # TODO(b/138957587): Remove when force_v2_in_keras_compile is on longer
        # a valid arg for this model. Also remove as a valid flag.
        if flags_obj.force_v2_in_keras_compile is not None:
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=(['sparse_categorical_accuracy']
                         if flags_obj.report_accuracy_metrics else None),
                run_eagerly=flags_obj.run_eagerly,
                experimental_run_tf_function=flags_obj.force_v2_in_keras_compile)
        else:
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=(['sparse_categorical_accuracy']
                         if flags_obj.report_accuracy_metrics else None),
                run_eagerly=flags_obj.run_eagerly)

    callbacks = common.get_callbacks(
        learning_rate_schedule, imagenet_preprocessing.NUM_IMAGES['train'])

    train_steps = (
            imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
    train_epochs = flags_obj.train_epochs

    if flags_obj.train_steps:
        train_steps = min(flags_obj.train_steps, train_steps)
        train_epochs = 1

    num_eval_steps = (
            imagenet_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size)

    validation_data = eval_input_dataset
    if flags_obj.skip_eval:
        # Only build the training graph. This reduces memory usage introduced by
        # control flow ops in layers that have different implementations for
        # training and inference (e.g., batch norm).
        if flags_obj.set_learning_phase_to_train:
            # TODO(haoyuzhang): Understand slowdown of setting learning phase when
            # not using distribution strategy.
            tf.keras.backend.set_learning_phase(1)
        num_eval_steps = None
        validation_data = None

    if not strategy and flags_obj.explicit_gpu_placement:
        # TODO(b/135607227): Add device scope automatically in Keras training loop
        # when not using distribition strategy.
        no_dist_strat_device = tf.device('/device:GPU:0')
        no_dist_strat_device.__enter__()

    history = model.fit(train_input_dataset,
                        epochs=train_epochs,
                        steps_per_epoch=train_steps//15,
                        callbacks=callbacks,
                        validation_steps=num_eval_steps,
                        validation_data=validation_data,
                        validation_freq=flags_obj.epochs_between_evals,
                        verbose=1)

    eval_output = None
    if not flags_obj.skip_eval:
        eval_output = model.evaluate(eval_input_dataset,
                                     steps=num_eval_steps,
                                     verbose=2)

    if not strategy and flags_obj.explicit_gpu_placement:
        no_dist_strat_device.__exit__()

    stats = common.build_stats(history, eval_output, callbacks)
    return stats


def define_imagenet_keras_flags():
    common.define_keras_flags()
    flags_core.set_defaults(train_epochs=90)
    flags.adopt_module_key_flags(common)


def main(_):
    model_helpers.apply_clean(flags.FLAGS)
    with logger.benchmark_context(flags.FLAGS):
        stats = run(flags.FLAGS)
    logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    define_imagenet_keras_flags()
    absl_app.run(main)
