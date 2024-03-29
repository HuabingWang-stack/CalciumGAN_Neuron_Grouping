import os
import pickle
import random
import platform
import argparse
import warnings
import numpy as np
import pandas as pd
from time import time
import multiprocessing

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from gan.utils import utils
from gan.utils import h5_helper
from gan.utils import spike_metrics
from gan.utils import spike_helper
from gan.utils.summary_helper import Summary
from gan.utils.cascade.cascade import deconvolve_batch

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


def load_info(hparams):
  filename = os.path.join(hparams.generated_dir, 'info.pkl')
  with open(filename, 'rb') as file:
    info = pickle.load(file)
  return info


def deconvolve_neuron(hparams, filename, neuron):
  signals = h5_helper.get(filename, name='signals', neuron=neuron)
  signals = utils.set_array_format(signals, data_format='NW', hparams=hparams)
  return spike_helper.deconvolve_signals(signals, threshold=0.5)


def deconvolve_from_file(hparams, filename, return_spikes=False):
  if h5_helper.contains(filename, name='spikes'):
    fake_spikes = h5_helper.get(filename, name='spikes')
  else:
    if hparams.verbose:
      print('\tDeconvolve {}'.format(filename))

    pool = multiprocessing.Pool(hparams.num_processors)
    fake_spikes = pool.starmap(
        deconvolve_neuron,
        [(hparams, filename, n) for n in range(hparams.num_neurons)])
    pool.close()

    fake_spikes = utils.set_array_format(
        np.array(fake_spikes, dtype=np.int8), data_format='NWC', hparams=hparams)

    h5_helper.write(filename, {'spikes': fake_spikes})

  if return_spikes:
    return fake_spikes

def deconvolve_from_file_by_cascade(hparams, filename, return_spikes=False):
  
  #infer cascade spikes on synthetic signals
  if hparams.verbose:
    print('\tDeconvolve {}'.format(filename))
  if not os.path.exists(filename):
    raise FileNotFoundError(f"{filename} not found")
  signals = h5_helper.get(filename, name='signals')
  # convert to (num. samples, num. neurons, time-steps)
  # signals = np.transpose(signals, axes=[0, 2, 1])
  signals = utils.set_array_format(signals, data_format='NCW', hparams=hparams)
  if h5_helper.contains(filename, name='cascade'):
    spike_trains = h5_helper.get(filename, name='cascade')
    print('{} contains cascade inferred spike trains'.format(filename))
  else:
    spike_trains = deconvolve_batch(
              signals=signals, num_processors=hparams.num_processors
          )
    # back to (num. samples, time-steps, num. neurons)
    spike_trains = utils.set_array_format(spike_trains, data_format='NWC', hparams=hparams)       
    h5_helper.write(filename, {'cascade': spike_trains})


  #infer cascade spikes on validation signals
  if h5_helper.contains(hparams.validation_cache, name='cascade'):
    validation_spike_trains = h5_helper.get(filename, name='cascade')
    print('{} contains cascade inferred spike trains'.format(filename))
  else:
    if hparams.verbose:
      print('\tDeconvolve {}'.format(hparams.validation_cache))
    if not os.path.exists(filename):
      raise FileNotFoundError(f"{hparams.validation_cache} not found")
    validation_signals = h5_helper.get(hparams.validation_cache, name='signals')
    validation_signals = utils.set_array_format(validation_signals, data_format='NCW', hparams=hparams)
    validation_spike_trains = deconvolve_batch(
              signals=validation_signals, num_processors=hparams.num_processors
          )
    validation_spike_trains = utils.set_array_format(validation_spike_trains, data_format='NWC', hparams=hparams)
    h5_helper.write(hparams.validation_cache, {'cascade': validation_spike_trains}) 

  with open(hparams.output_dir+'/metrics/plots/stats.txt','w') as f:
      f.write('spike counts: {:.06f}\n'
              .format(np.sum(spike_trains)))
      
  if return_spikes:
    return spike_trains

def get_neo_trains(hparams,
                   filename,
                   neuron=None,
                   trial=None,
                   data_format=None,
                   num_trials=None):
  assert data_format and (neuron is not None or trial is not None)

  spikes = h5_helper.get(filename, name=hparams.spike_metric, neuron=neuron, trial=trial)
  spikes = utils.set_array_format(spikes, data_format, hparams)

  if num_trials is not None:
    assert data_format[0] == 'N'
    spikes = spikes[:num_trials]
  
  spikes = np.zeros_like(spikes)

  return spike_helper.trains_to_neo(spikes)


def mse(x, y):
  return np.nanmean(np.square(x - y), dtype=np.float32)


def kl_divergence(p, q):
  # replace entries with 0 probability with 1e-10
  p = np.where(p == 0, 1e-10, p)
  q = np.where(q == 0, 1e-10, q)
  return np.sum(p * np.log(p / q))


def pairs_kl_divergence(pairs):
  kl = np.zeros((len(pairs),), dtype=np.float32)
  # for each trial
  for i in range(len(pairs)):
    real, fake = pairs[i]

    df = pd.DataFrame({
        'data': np.concatenate([real, fake]),
        'is_real': [True] * len(real) + [False] * len(fake)
    })
    # cut concatenated data into 30 bins, label each bin with its bin number
    num_bins = 30
    df['bins'] = pd.cut(df.data, bins=num_bins, labels=np.arange(num_bins))
    #get the ratio of (real data in ith bin/ total # of real data)
    real_pdf = np.array([
        len(df[(df.bins == i) & (df.is_real == True)]) for i in range(num_bins)
    ],
                        dtype=np.float32) / len(real)
    fake_pdf = np.array([
        len(df[(df.bins == i) & (df.is_real == False)]) for i in range(num_bins)
    ],
                        dtype=np.float32) / len(fake)

    kl[i] = kl_divergence(real_pdf, fake_pdf)
  return kl


def plot_signals(hparams, summary, filename, epoch):
  # trial = random.randint(0, hparams.num_samples)
  trial = 0

  if hparams.verbose:
    print('\tPlotting traces for trial #{}'.format(trial))

  real_signals = h5_helper.get(
      hparams.validation_cache, name='signals', trial=trial)
  real_spikes = h5_helper.get(
      hparams.validation_cache, name=hparams.spike_metric, trial=trial)

  real_signals = utils.set_array_format(
      real_signals, data_format='CW', hparams=hparams)
  real_spikes = utils.set_array_format(
      real_spikes, data_format='CW', hparams=hparams)

  fake_signals = h5_helper.get(filename, name='signals', trial=trial)
  fake_spikes = h5_helper.get(filename, name=hparams.spike_metric, trial=trial)

  fake_signals = utils.set_array_format(
      fake_signals, data_format='CW', hparams=hparams)
  fake_spikes = utils.set_array_format(
      fake_spikes, data_format='CW', hparams=hparams)

  # get the y axis range for each neuron
  assert real_signals.shape == fake_signals.shape
  ylims = []
  for i in range(len(real_signals)):
    # ylims.append([
    #     np.min([np.min(real_signals[i]),
    #             np.min(fake_signals[i])]),
    #     np.max([np.max(real_signals[i]),
    #             np.max(fake_signals[i])])
    # ])
        ylims.append([
        np.min([np.min(real_signals),
                np.min(fake_signals)]),
        np.max([np.max(real_signals),
                np.max(fake_signals)])
    ])

  summary.plot_traces(
      'real_traces',
      real_signals,
      real_spikes,
      indexes=hparams.neurons[:hparams.num_neuron_plots],
      ylims=ylims,
      step=epoch,
      is_real=True,
      signal_label='recorded signal',
      spike_label='inferred spike',
      plots_per_row=hparams.plots_per_row)

  summary.plot_traces(
      'fake_traces',
      fake_signals,
      fake_spikes,
      indexes=hparams.neurons[:hparams.num_neuron_plots],
      ylims=ylims,
      step=epoch,
      is_real=False,
      signal_label='synthetic signal',
      spike_label='inferred spike',
      plots_per_row=hparams.plots_per_row)


def raster_plots(hparams, summary, filename, epoch, trial=0): # 100
  if hparams.verbose:
    print('\tPlotting raster plot for trial #{}'.format(trial))

  real_spikes = h5_helper.get(
      hparams.validation_cache, name=hparams.spike_metric, trial=trial)
  real_spikes = utils.set_array_format(real_spikes, 'CW', hparams)
  fake_spikes = h5_helper.get(filename, name=hparams.spike_metric, trial=trial)
  fake_spikes = utils.set_array_format(fake_spikes, 'CW', hparams)

  summary.raster_plot(
      'raster_plot',
      real_spikes=real_spikes,
      fake_spikes=fake_spikes,
      xlabel='Time (s)',
      ylabel='Neuron',
      legend_labels=['recorded', 'synthetic'],
      step=epoch)


def firing_rate(hparams, filename, neuron, num_trials=100):#200
  if hparams.verbose == 2:
    print('\tComputing firing rate for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
      num_trials=num_trials)
  fake_spikes = get_neo_trains(
      hparams,
      filename,
      neuron=neuron,
      data_format='NW',
      num_trials=num_trials,
  )

  real_firing_rate = spike_metrics.mean_firing_rate(real_spikes)
  fake_firing_rate = spike_metrics.mean_firing_rate(fake_spikes)

  return (real_firing_rate, fake_firing_rate)


def firing_rate_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing firing rate')

  pool = multiprocessing.Pool(hparams.num_processors)
  # 102,2,1000
  firing_rate_pairs = pool.starmap(firing_rate,
                                   [(hparams, filename, n, hparams.num_samples)
                                    for n in range(hparams.num_neurons)])
  pool.close()

  summary.plot_histograms_grid(
      'firing_rate',
      data=[firing_rate_pairs[n] for n in hparams.neurons],
      xlabel='Hz',
      ylabel='Count',
      titles=['Neuron #{:03d}'.format(n) for n in hparams.neurons],
      step=epoch,
      legend_labels=['recorded', 'synthetic'],
      plots_per_row=hparams.plots_per_row)

  kl_divergence = pairs_kl_divergence(firing_rate_pairs)
  summary.plot_distribution(
      'firing_rate_kl',
      data=kl_divergence,
      xlabel='KL divergence',
      ylabel='Count',
      title='Firing Rate',
      step=epoch)

  with open(hparams.output_dir+'/metrics/plots/stats.txt','w') as f:
      f.write('firing rate KL mean: {:.06f}, kL std {:.06f}\n'
              .format(np.mean(kl_divergence),np.std(kl_divergence)))

  if hparams.verbose:
    message = '\t\tKL mean: {:.04f}\n'.format(np.mean(kl_divergence))
    for n in hparams.neurons:
      message += '\t\tneuron {:03d}: {:.02f}\n'.format(n, kl_divergence[n])
    print(message)


def covariance(hparams, filename, trial):
  if hparams.verbose == 2:
    print('\t\tComputing covariance for sample #{}'.format(trial))

  diag_indices = np.triu_indices(hparams.num_neurons, k=1)

  real_spikes = get_neo_trains(
      hparams, hparams.validation_cache, trial=trial, data_format='CW')
  real_covariance = spike_metrics.covariance(real_spikes, None)
  real_covariance = utils.remove_nan(real_covariance[diag_indices])

  fake_spikes = get_neo_trains(hparams, filename, trial=trial, data_format='CW')
  fake_covariance = spike_metrics.covariance(fake_spikes, None)
  fake_covariance = utils.remove_nan(fake_covariance[diag_indices])

  return (real_covariance, fake_covariance)


def covariance_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing covariance')

  pool = multiprocessing.Pool(hparams.num_processors)
  covariances = pool.starmap(
      covariance, [(hparams, filename, i) for i in range(hparams.num_samples)])
  pool.close()

  summary.plot_histograms_grid(
      'covariance_histogram',
      data=[covariances[i] for i in hparams.trials],
      xlabel='Covariance',
      ylabel='Count',
      titles=['Sample #{:03d}'.format(i) for i in hparams.trials],
      step=epoch,
      legend_labels=['recorded', 'synthetic'],
      plots_per_row=hparams.plots_per_row)

  kl_divergence = pairs_kl_divergence(covariances)
  summary.plot_distribution(
      'covariance_kl',
      data=kl_divergence,
      xlabel='KL divergence',
      ylabel='Count',
      title='Covariance',
      step=epoch)

  if hparams.verbose:
    print(
        '\tmin: {:.04f}, max: {:.04f}, mean: {:.04f}, num below 1.5: {}'.format(
            np.min(kl_divergence), np.max(kl_divergence),
            np.mean(kl_divergence), np.count_nonzero(kl_divergence < 1.5)))


def correlation_coefficient(hparams, filename, trial):
    # get correlation_coefficient matrix of 102 neurons of a trial
  if hparams.verbose == 2:
    print('\t\tComputing correlation coefficient for sample #{}'.format(trial))

  diag_indices = np.triu_indices(hparams.num_neurons, k=1)

  real_spikes = get_neo_trains(
      hparams, hparams.validation_cache, trial=trial, data_format='CW')
  real_corrcoef = spike_metrics.correlation_coefficients(real_spikes, None)
  real_corrcoef = utils.remove_nan(real_corrcoef[diag_indices])

  fake_spikes = get_neo_trains(hparams, filename, trial=trial, data_format='CW')
  fake_corrcoef = spike_metrics.correlation_coefficients(fake_spikes, None)
  fake_corrcoef = utils.remove_nan(fake_corrcoef[diag_indices])

  return (real_corrcoef, fake_corrcoef)


def correlation_coefficient_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing correlation coefficient')

  pool = multiprocessing.Pool(hparams.num_processors)
  # 1000,2,1000~2000
  correlations = pool.starmap(
      correlation_coefficient,
      [(hparams, filename, i) for i in range(hparams.num_samples)])
  pool.close()

  summary.plot_histograms_grid(
      'correlation',
      data=[correlations[i] for i in hparams.trials],
      xlabel='Correlation',
      ylabel='Count',
      titles=['Sample #{:03d}'.format(i) for i in hparams.trials],
      step=epoch,
      legend_labels=['recorded', 'synthetic'],
      plots_per_row=hparams.plots_per_row)

  kl_divergence = pairs_kl_divergence(correlations)
  summary.plot_distribution(
      'correlation_kl',
      data=kl_divergence,
      xlabel='KL divergence',
      ylabel='Count',
      title='Correlation',
      step=epoch)
  with open(hparams.output_dir+'/metrics/plots/stats.txt','a') as f:
      f.write('correlation coefficient KL mean: {:.06f}, kl std {:.06f}\n'
              .format(np.mean(kl_divergence),np.std(kl_divergence)))
  if hparams.verbose:
    print('\t\tmean: {:.04f}'.format(np.mean(kl_divergence)))


def sort_heatmap(matrix):
  ''' sort the given matrix where the top left corner is the minimum'''
  num_trials = len(matrix)

  # create a copy of distances matrix for modification
  matrix_copy = np.copy(matrix)

  heatmap = np.full(matrix.shape, fill_value=np.nan, dtype=np.float32)

  # get the index with the minimum value
  min_index = np.unravel_index(np.argmin(matrix), matrix.shape)

  # row and column order for the sorted matrix
  row_order = np.full((num_trials,), fill_value=-1, dtype=np.int)
  row_order[0] = min_index[0]
  column_order = np.argsort(matrix[min_index[0]])

  for i in range(num_trials):
    if i != 0:
      row_order[i] = np.argsort(matrix_copy[:, column_order[i]])[0]
    heatmap[i] = matrix[row_order[i]][column_order]
    matrix_copy[row_order[i]][:] = np.inf

  return heatmap, row_order, column_order


def neuron_van_rossum(hparams, filename, neuron, num_trials=50):
  ''' compute van rossum heatmap for neuron with num_trials '''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum heatmap for neuron #{}'.format(neuron))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      neuron=neuron,
      data_format='NW',
      num_trials=num_trials)
  fake_spikes = get_neo_trains(
      hparams, filename, neuron=neuron, data_format='NW', num_trials=num_trials)

  distances = spike_metrics.van_rossum_distance(real_spikes, fake_spikes)
  heatmap, row_order, column_order = sort_heatmap(distances)

  return {
      'heatmap': heatmap,
      'xticklabels': row_order,
      'yticklabels': column_order
  }


def trial_van_rossum(hparams, filename, trial):
  ''' compute van rossum distance for a given trial '''
  if hparams.verbose == 2:
    print('\t\tComputing van-rossum histograms for trial #{}'.format(trial))

  real_spikes = get_neo_trains(
      hparams,
      hparams.validation_cache,
      trial=trial,
      data_format='CW',
  )
  real_van_rossum = spike_metrics.van_rossum_distance(real_spikes, None)

  fake_spikes = get_neo_trains(
      hparams,
      filename,
      trial=trial,
      data_format='CW',
  )
  fake_van_rossum = spike_metrics.van_rossum_distance(fake_spikes, None)

  assert real_van_rossum.shape == fake_van_rossum.shape

  diag_indices = np.triu_indices(len(real_van_rossum), k=1)

  real_van_rossum = real_van_rossum[diag_indices]
  fake_van_rossum = fake_van_rossum[diag_indices]

  return (real_van_rossum, fake_van_rossum)


def van_rossum_metrics(hparams, summary, filename, epoch):
  if hparams.verbose:
    print('\tComputing van-rossum distance')

  # compute van-Rossum distance heatmap
  pool = multiprocessing.Pool(hparams.num_processors)
  results = pool.starmap(neuron_van_rossum,
                         [(hparams, filename, n, 45) for n in hparams.neurons])
  
  pool.close()

  heatmaps, xticklabels, yticklabels, titles = [], [], [], []
  heatmap_means = []
  for i in range(len(results)):
    heatmaps.append(results[i]['heatmap'])
    xticklabels.append(results[i]['xticklabels'])
    yticklabels.append(results[i]['yticklabels'])
    titles.append('Neuron #{:03d}'.format(hparams.neurons[i]))
    heatmap_means.append(np.mean(results[i]['heatmap']))

  with open(hparams.output_dir+'/metrics/plots/stats.txt','a') as f:
      f.write('average van-Rossum mean: {:.06f}, van-Rossum mean std {:.06f}\n'
              .format(np.mean(heatmap_means),np.std(heatmap_means)))  
      
  summary.plot_heatmaps_grid(
      'van_rossum',
      matrix=heatmaps,
      xlabel='synthetic trial',
      ylabel='recorded trial',
      xticklabels=xticklabels,
      yticklabels=yticklabels,
      titles=titles,
      step=epoch,
      plots_per_row=hparams.plots_per_row)

  # compute van rossum distance KL divergence
  pool = multiprocessing.Pool(hparams.num_processors)
  van_rossum_pairs = pool.starmap(
      trial_van_rossum,
      [(hparams, filename, i) for i in range(hparams.num_samples)])
  pool.close()

  kl_divergence = pairs_kl_divergence(van_rossum_pairs)
  
  summary.plot_histograms_grid(
      'van-Rossum distance',
      data=[van_rossum_pairs[i] for i in hparams.trials],
      xlabel='van-Rossum distances',
      ylabel='Count',
      titles=['Sample #{:03d}'.format(i) for i in hparams.trials],
      step=epoch,
      legend_labels=['recorded', 'synthetic'],
      plots_per_row=hparams.plots_per_row)
  
  summary.plot_distribution(
      'van_rossum_kl',
      data=kl_divergence,
      xlabel='KL divergence',
      ylabel='Count',
      title='van-Rossum distance',
      step=epoch)
  with open(hparams.output_dir+'/metrics/plots/stats.txt','a') as f:
      f.write('van rossum KL mean: {:.06f}, kl std {:.06f}\n'
              .format(np.mean(kl_divergence),np.std(kl_divergence)))
  if hparams.verbose:
    print('\t\tmean: {:.04f}'.format(np.mean(kl_divergence)))


def compute_epoch_spike_metrics(hparams, summary, filename, epoch):
  # if not h5_helper.contains(filename, 'spikes'):
  if hparams.spike_metric == 'cascade':
    deconvolve_from_file_by_cascade(hparams, filename)
  elif hparams.spike_metric == 'spikes':
    deconvolve_from_file(hparams, filename)
  plot_signals(hparams, summary, filename, epoch)

  raster_plots(hparams, summary, filename, epoch)

  firing_rate_metrics(hparams, summary, filename, epoch)

  # covariance_metrics(hparams, summary, filename, epoch)

  correlation_coefficient_metrics(hparams, summary, filename, epoch)

  van_rossum_metrics(hparams, summary, filename, epoch)


def main(hparams):
  os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
  if not os.path.exists(hparams.output_dir):
    print('{} not found'.format(hparams.output_dir))
    exit()

  set_seed(hparams.seed)

  utils.load_hparams(hparams)
  info = load_info(hparams)

  hparams.num_samples = min(
      h5_helper.get_dataset_length(hparams.validation_cache, 'signals'), 1000)

  # randomly select neurons and trials to plot
  # hparams.neurons = list(
  #     range(hparams.num_neurons
  #          ) if hparams.num_neuron_plots >= hparams.num_neurons else np.random.
  #     choice(hparams.num_neurons, hparams.num_neuron_plots))
  hparams.neurons = list(range(hparams.num_neuron_plots))
  hparams.trials = list(np.random.choice(hparams.num_samples, hparams.num_trial_plots))
      # np.random.choice(hparams.num_samples, hparams.num_trial_plots))

  summary = Summary(hparams, spike_metrics=True)

  epochs = sorted(list(info.keys()))

  # only compute metrics for the last generated file
  if not hparams.all_epochs:
    epochs = [epochs[-1]]

  for epoch in epochs:
    start = time()
    if hparams.verbose:
      print('\nCompute metrics for {}'.format(info[epoch]['filename']))
      if os.name == "posix":
        info[epoch]['filename'] = info[epoch]['filename'].replace('\\', '/')
    compute_epoch_spike_metrics(
        hparams, summary, filename=info[epoch]['filename'], epoch=epoch)
    end = time()

    summary.scalar('elapse/spike_metrics', end - start, step=epoch)

    if hparams.verbose:
      print('{} took {:.02f} mins'.format(info[epoch]['filename'],
                                          (end - start) / 60))
  if hparams.spike_metric == 'cascade':
    os.rename(os.path.join(hparams.output_dir, 'metrics','plots'),
    os.path.join(hparams.output_dir, 'metrics','plots_cascade'))
  
  if hparams.spike_metric == 'spikes':
    os.rename(os.path.join(hparams.output_dir, 'metrics','plots'),
    os.path.join(hparams.output_dir, 'metrics','plots_spikes'))


if __name__ == '__main__':


  if platform.system() == 'Darwin':
    multiprocessing.set_start_method('spawn')

  # use CPU only
  # os.environ["CUDA_VISIBLE_DEVICES"] = ""

  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', default='runs')
  parser.add_argument('--num_processors', default=10, type=int)
  parser.add_argument('--all_epochs', action='store_true')
  # just cascade for showing the difference between oasis and cascade
   # using oasis input 'spikes', cascade input 'cascade' 
  parser.add_argument('--spike_metric', default='spikes') 
  parser.add_argument('--num_neuron_plots', default=102, type=int)
  parser.add_argument('--num_trial_plots', default=10, type=int)
  parser.add_argument('--plots_per_row', default=3, type=int)
  parser.add_argument('--dpi', default=120, type=int)
  parser.add_argument('--format', default='pdf', choices=['pdf', 'png'])
  parser.add_argument('--verbose', default=1, type=int)
  parser.add_argument('--seed', default=12, type=int)
  hparams = parser.parse_args()

  main(hparams)
