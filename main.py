import os
import argparse
import numpy as np
from time import time
from tqdm import tqdm
import tensorflow as tf
from shutil import rmtree

from tensorflow.keras.mixed_precision import experimental as mixed_precision

np.random.seed(1234)
tf.random.set_seed(1234)

from gan.utils import utils
from gan.utils import spike_helper
from gan.models.registry import get_models
from gan.utils.summary_helper import Summary
from gan.utils.dataset_helper import get_dataset
from gan.algorithms.registry import get_algorithm


def set_precision_policy(hparams):
  policy = None
  if hparams.mixed_precision:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    if hparams.verbose:
      print('\nCompute dtype: {}\nVariable dtype: {}\n'.format(
          policy.compute_dtype, policy.variable_dtype))
  return policy


def train(hparams, train_ds, gan, summary, epoch):
  gen_losses, dis_losses, gradient_penalties = [], [], []
  batch_count = 0

  start = time()

  for signal, _ in tqdm(
      train_ds,
      desc='Train',
      total=hparams.train_steps,
      disable=not bool(hparams.verbose)):

    if hparams.profile and batch_count == 2 and epoch == 1:
      # profile the training session of the 2nd batch in 2nd epoch
      summary.profiler_trace()

    gen_loss, dis_loss, gradient_penalty, metrics = gan.train(signal)

    if hparams.profile and batch_count == 6 and epoch == 1:
      summary.profiler_export()

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)
    if gradient_penalty is not None:
      gradient_penalties.append(gradient_penalty)

    hparams.global_step += 1
    batch_count += 1

  end = time()

  gen_loss, dis_loss = np.mean(gen_losses), np.mean(dis_losses)

  summary.log(
      gen_loss,
      dis_loss,
      np.mean(gradient_penalties) if gradient_penalties else None,
      elapse=end - start,
      gan=gan,
      step=epoch,
      training=True)

  return gen_loss, dis_loss


def validate(hparams, validation_ds, gan, summary, epoch):
  gen_losses, dis_losses, gradient_penalties, results = [], [], [], {}

  save_generated = (hparams.save_generated == 'all' and
                    (epoch % 10 == 0 or epoch == hparams.epochs - 1)) or (
                        hparams.save_generated == 'last' and
                        epoch == hparams.epochs - 1)

  start = time()

  for signal, _ in tqdm(
      validation_ds,
      desc='Validate',
      total=hparams.validation_steps,
      disable=not bool(hparams.verbose)):
    fake, gen_loss, dis_loss, gradient_penalty, metrics = gan.validate(signal)

    # append losses and metrics
    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)
    if gradient_penalty is not None:
      gradient_penalties.append(gradient_penalty)
    for key, item in metrics.items():
      if key not in results:
        results[key] = []
      results[key].append(item)

    if save_generated:
      utils.save_fake_signals(hparams, epoch, signals=fake)

  gen_loss, dis_loss = np.mean(gen_losses), np.mean(dis_losses)
  results = {key: np.mean(item) for key, item in results.items()}

  end = time()

  summary.log(
      gen_loss,
      dis_loss,
      np.mean(gradient_penalties) if gradient_penalties else None,
      metrics=results,
      elapse=end - start,
      step=epoch,
      training=False)

  return gen_loss, dis_loss


def train_and_validate(hparams, train_ds, validation_ds, gan, summary):
  # noise to test generator and plot to TensorBoard
  test_noise = gan.get_noise(batch_size=1)

  for epoch in range(hparams.start_epoch, hparams.epochs):
    if hparams.verbose:
      print('Epoch {:03d}/{:03d}'.format(epoch, hparams.epochs))

    start = time()

    train_gen_loss, train_dis_loss = train(
        hparams, train_ds, gan=gan, summary=summary, epoch=epoch)

    val_gen_loss, val_dis_loss = validate(
        hparams, validation_ds, gan=gan, summary=summary, epoch=epoch)

    if epoch % 10 == 0 or epoch == hparams.epochs - 1:
      # test generated data and plot in TensorBoard
      fake_signals = gan.generate(test_noise)
      fake_signals = utils.reverse_preprocessing(hparams, fake_signals)
      fake_signals = utils.set_array_format(
          fake_signals[0], data_format='CW', hparams=hparams)
      fake_spikes = spike_helper.deconvolve_signals(fake_signals)
      summary.plot_traces(
          'fake_traces',
          fake_signals,
          fake_spikes,
          indexes=hparams.focus_neurons,
          step=epoch,
          training=False)
      if not hparams.skip_checkpoints:
        utils.save_models(hparams, gan, epoch)

    end = time()

    if hparams.verbose:
      print('Train: generator loss {:.04f} discriminator loss {:.04f}\n'
            'Eval: generator loss {:.04f} discriminator loss {:.04f}\n'
            'Elapse: {:.02f} mins\n'.format(train_gen_loss, train_dis_loss,
                                            val_gen_loss, val_dis_loss,
                                            (end - start) / 60))


def test(validation_ds, gan):
  gen_losses, dis_losses, results = [], [], {}

  for signal, _ in validation_ds:
    _, gen_loss, dis_loss, _, metrics = gan.validate(signal)

    gen_losses.append(gen_loss)
    dis_losses.append(dis_loss)
    for key, item in metrics.items():
      if key not in results:
        results[key] = []
      results[key].append(item)

  return {key: np.mean(item) for key, item in results.items()}


def main(hparams, return_metrics=False):
  if hparams.clear_output_dir and os.path.exists(hparams.output_dir):
    rmtree(hparams.output_dir)

  tf.keras.backend.clear_session()

  # hparams.focus_neurons = [87, 58, 90, 39, 7, 60, 14, 5, 13]
  hparams.focus_neurons = list(range(4))

  policy = set_precision_policy(hparams)

  summary = Summary(hparams, policy=policy)

  train_ds, validation_ds = get_dataset(hparams, summary)

  generator, discriminator = get_models(hparams, summary)

  utils.save_hparams(hparams)

  gan = get_algorithm(hparams, generator, discriminator, summary)

  utils.load_models(hparams, gan)

  start = time()

  train_and_validate(
      hparams,
      train_ds=train_ds,
      validation_ds=validation_ds,
      gan=gan,
      summary=summary)

  end = time()

  summary.scalar('elapse/total', end - start)

  # generate dataset for surrogate metrics
  if hparams.surrogate_ds:
    utils.generate_dataset(hparams, gan=gan, num_samples=2 * 10**6)

  if return_metrics:
    return test(validation_ds, gan)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default='dataset/tfrecords')
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--num_units', default=32, type=int)
  parser.add_argument('--kernel_size', default=24, type=int)
  parser.add_argument('--strides', default=2, type=int)
  parser.add_argument('--m', default=2, type=int, help='phase shuffle m')
  parser.add_argument('--n', default=2, type=int, help='phase shuffle n')
  parser.add_argument('--epochs', default=20, type=int)
  parser.add_argument('--dropout', default=0.2, type=float)
  parser.add_argument('--learning_rate', default=0.0001, type=float)
  parser.add_argument('--noise_dim', default=32, type=int)
  parser.add_argument('--gradient_penalty', default=10.0, type=float)
  parser.add_argument('--model', default='wavegan', type=str)
  parser.add_argument('--activation', default='leakyrelu', type=str)
  parser.add_argument('--batch_norm', action='store_true')
  parser.add_argument('--layer_norm', action='store_true')
  parser.add_argument('--algorithm', default='wgan-gp', type=str)
  parser.add_argument(
      '--n_critic',
      default=5,
      type=int,
      help='number of steps between each generator update')
  parser.add_argument('--clear_output_dir', action='store_true')
  parser.add_argument(
      '--save_generated', default="", choices=["", "last", "all"], type=str)
  parser.add_argument('--plot_weights', action='store_true')
  parser.add_argument('--skip_checkpoints', action='store_true')
  parser.add_argument('--mixed_precision', action='store_true')
  parser.add_argument(
      '--profile', action='store_true', help='enable TensorBoard profiling')
  parser.add_argument('--dpi', default=120, type=int)
  parser.add_argument('--verbose', default=1, type=int)
  params = parser.parse_args()

  params.global_step = 0
  params.surrogate_ds = True if 'surrogate' in params.input_dir else False

  main(params)
