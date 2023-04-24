import os
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from functools import partial
from tqdm.contrib import concurrent
from multiprocessing import cpu_count

# from nlacgan.utils import h5 as h5
from gan.utils.cascade.cascade2p import cascade, utils_discrete_spikes

tf.get_logger().setLevel("ERROR")

MODEL_FOLDER = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "pretrained_models"
)


def signals2probs(
    signals: np.ndarray, model_name: str = "Global_EXC_25Hz_smoothing100ms"
):
    """Deconvolve signals and return spike probabilities
    Args:
      signals: np.ndarray, signals in format (num. neurons, time-steps)
      model_name: str, Cascade model name
    Returns:
      spike_probs: np.ndarray, spike probabilities in (num. neurons, time-steps)
    """
    assert len(signals.shape) == 2
    return cascade.predict(
        model_name=model_name,
        traces=signals,
        model_folder=MODEL_FOLDER,
        clear_backend=True,
    )


def probs2trains(
    spike_probs: np.ndarray, model_name: str = "Global_EXC_25Hz_smoothing100ms"
):
    """Infer discrete spike trains from spike probabilities
    Args:
      spike_probs: np.ndarray, spike probabilities in (num. neurons, time-steps)
      model_name: str, Cascade model name
    Returns:
      spike_trains: np.ndarray, discrete spike trains in
                                (num. neurons, time-steps)
    """
    assert len(spike_probs.shape) == 2
    _, spike_times = utils_discrete_spikes.infer_discrete_spikes(
        spike_rates=spike_probs, model_name=model_name, model_folder=MODEL_FOLDER
    )
    spike_trains = np.zeros_like(spike_probs, dtype=np.int8)
    for n in range(len(spike_times)):
        if len(spike_times[n]) > 0:
            spike_trains[n, spike_times[n]] = 1.0
    return spike_trains


def deconvolve_batch(
    signals: np.ndarray,
    model_name: str = "Global_EXC_25Hz_smoothing100ms",
    num_processors: int = cpu_count() - 2,
):
    """Deconvolve batch of signals and return discrete spike trains
    Args:
      signals: np.ndarray, signals in format
                          (num. samples, num. neurons, time-steps)
      model_name: str, the model name in Cascade
      num_processors: int, the number of processors to use in Pool
    Returns:
      spike_trains: np.ndarray, discrete spike trains in format
                                (num. samples, num. neurons, time-steps)
    """
    assert len(signals.shape) == 3
    if not os.path.isdir(os.path.join(MODEL_FOLDER, model_name)):
        cascade.download_model(model_name=model_name, model_folder=MODEL_FOLDER)
    num_samples = signals.shape[0]
    gpus = tf.config.list_physical_devices('GPU')
    spike_probs = []
    if len(gpus) > 0:
    # deconvolve signals and obtain spike probabilities
        for i in tqdm(range(num_samples), desc="signals2probs"):
            spike_probs.append(signals2probs(signals[i], model_name=model_name))
    else:
        spike_probs = concurrent.process_map(partial(signals2probs, model_name=model_name), 
        [signals[i] for i in range(num_samples)],max_workers=num_processors,desc="signals2probs")

    # convert spike probabilities to discrete spike trains
    spike_trains = concurrent.process_map(
        partial(probs2trains, model_name=model_name),
        [spike_probs[i] for i in range(num_samples)],
        max_workers=num_processors,
        desc="probs2spikes",
    )
    return np.array(spike_trains, dtype=np.int8)


# def deconvolve_file(
#     signals_filename: str,
#     spikes_filename: str,
#     model_name: str = "Global_EXC_25Hz_smoothing100ms",
#     num_processors: int = cpu_count() - 2,
# ):
#     if not os.path.exists(signals_filename):
#         raise FileNotFoundError(f"{signals_filename} not found")
#     if os.path.exists(spikes_filename):
#         os.remove(spikes_filename)

#     tf.keras.backend.clear_session()

#     print(f"deconvolve file {signals_filename}...")
#     for key in ["x", "y", "fake_x", "fake_y", "cycle_x", "cycle_y"]:
#         print(f"\ndeconvolve {key}...")
#         signals = h5.get(signals_filename, key=key)
#         # convert to (num. samples, time-steps, num. neurons)
#         signals = np.transpose(signals, axes=[0, 2, 1])
#         spike_trains = deconvolve_batch(
#             signals=signals, model_name=model_name, num_processors=num_processors
#         )
#         # convert to (num. samples, num. neurons, time-steps)
#         spike_trains = np.transpose(spike_trains, axes=[0, 2, 1])
#         h5.write(spikes_filename, data={key: spike_trains})


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--signals_filename", type=str, required=True)
#     parser.add_argument("--spikes_filename", type=str, required=True)
#     parser.add_argument("--num_processors", type=str, default=6)
#     args = parser.parse_args()
    # deconvolve_file(
    #     signals_filename=args.signals_filename,
    #     spikes_filename=args.spikes_filename,
    #     num_processors=args.num_processors,
    # )
