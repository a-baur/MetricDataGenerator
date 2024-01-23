import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import csv
import toml


def load_config(path: str):
    if not os.path.isabs(path):
        path = f"{os.getcwd()}/{path}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file {path} not found.")
    with open(path, 'r') as f:
        config = toml.load(f)
    return config


def generate_signals(metrics: dict, anomalies: dict, seed: int, n_samples: int):
    """
    Generate metrics.

    :param metrics: Dictionary of metrics to generate.
    :param anomalies: Dictionary of anomalies to inject.
    :param seed: Seed for the random number generator.
    :param n_samples: Number of samples to generate.
    :return: A list of metrics.
    """
    signals = {}

    for metric, params in metrics.items():
        value = params.pop('value')
        signals[metric] = generate_base_signal(value, n_samples)

        anomaly = anomalies.get(metric)
        if anomaly:
            at = anomaly.pop('at')
            anomaly_factor = anomaly.pop('anomaly_factor')
            pattern = anomaly.pop('pattern')
            recovery_time = anomaly.pop('recovery_time')
            pattern_args = anomaly.pop('pattern_args', {})
            signals[metric] = inject_anomaly(signals[metric], at, anomaly_factor, pattern, recovery_time, **pattern_args)

        noise = params.pop('noise', None)
        if noise:
            noise_args = params.pop('noise_args')
            signals[metric] = inject_noise(signals[metric], seed=seed, distribution=noise, **noise_args)

        step_size = params.pop('step_size', None)
        if step_size:
            signals[metric] = discretize_signal(signals[metric], step_size)

        signals[metric] = signals[metric].round(6)

    return signals


def generate_base_signal(value: int, n_samples: int):
    """
    Generate a signal from a distribution.

    :param value: Value to generate the signal around.
    :param n_samples: Number of samples to generate.
    :return: A signal.
    """
    return np.full(n_samples, float(value))


def discretize_signal(signal, step_size: float):
    """
    Discretize a signal into bins.

    :param signal: Signal to discretize.
    :param step_size: Size of the bins.
    :return: A discretized signal.
    """
    return np.round(signal / step_size) * step_size


def inject_noise(signal, seed: int, distribution: str, **noise_kwargs):
    """
    Inject noise into a signal using a random walk model.

    :param signal: Signal to inject noise into.
    :param seed: Seed for the random number generator.
    :param distribution: Distribution to sample noise from.
    :param noise_kwargs: Keyword arguments for the noise distribution.
    :return: A signal with noise.
    """
    random_generator = np.random.default_rng(seed=seed)

    smoothing_window = noise_kwargs.pop('smoothing_window', None)

    match distribution:
        case "gaussian":
            noise = random_generator.normal(**noise_kwargs, size=len(signal))
        case "uniform":
            noise = random_generator.uniform(**noise_kwargs, size=len(signal))
        case "exponential":
            noise = random_generator.exponential(**noise_kwargs, size=len(signal))
        case "logistic":
            noise = random_generator.logistic(**noise_kwargs, size=len(signal))
        case "laplace":
            noise = random_generator.laplace(**noise_kwargs, size=len(signal))
        case _:
            raise ValueError(f"Noise distribution {distribution} not supported.")

    if smoothing_window is not None:
        noise = np.convolve(noise, np.ones(smoothing_window) / smoothing_window, mode='same')

    return signal + noise


def inject_anomaly(signal, at: int, anomaly_factor: float, pattern: str, recovery_time: int, **pattern_kwargs):
    """
    Inject an anomaly into a signal.

    :param signal: Signal to inject anomaly into.
    :param at: Index to inject anomaly at.
    :param anomaly_factor: Magnitude of the anomaly.
    :param pattern: Pattern of the anomaly.
    :param recovery_time: Time to recover from the anomaly.
    :param pattern_kwargs: Pattern of the anomaly.
    :return: A signal with an anomaly.
    """

    if at + recovery_time > len(signal):
        raise ValueError(f"Anomaly injection goes beyond signal length ({at + recovery_time} > {len(signal)}).")

    match pattern:
        case "linear":
            response = np.linspace(1, 0, recovery_time)
        case "exponential":
            response = np.logspace(1, 0, base=recovery_time, num=recovery_time) - 1
            response = response / np.max(response)
        case "oscillating":
            frequency = pattern_kwargs.pop('frequency', 3)
            response = (
                    (np.logspace(1, 0, base=recovery_time, num=recovery_time) - 1)
                    / recovery_time * np.sin(frequency * np.arange(recovery_time))
            )
        case "step":
            response = np.ones(recovery_time)
        case _:
            return signal

    response = response * (anomaly_factor - 1) + 1
    signal[at:at+recovery_time] = signal[at:at+recovery_time] * response

    return signal


def create_dir(output_dir: str):
    # check if output directory is relative path
    if not os.path.isabs(output_dir):
        output_dir = f"{os.getcwd()}/{output_dir}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def plot_signal(metrics: dict, output_dir: str):
    """
    Plot a signal.

    :param metrics: Dictionary of metrics to plot.
    :param output_dir: Output directory.
    """
    output_dir = create_dir(output_dir)

    fig, axes = plt.subplots(nrows=len(metrics), ncols=1)
    fig.set_size_inches(10, 5 * len(metrics))
    for i, row in enumerate(metrics.items()):
        metric, signal = row
        if len(metrics) == 1:
            axes.plot(signal, label=metric)
            axes.set_title(metric)
        else:
            axes[i].plot(signal, label=metric)
            axes[i].set_title(metric)

    #  set size
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics.png")


def write_to_csv(metrics: dict, output_dir: str):
    """
    Write metrics to csv file.

    :param metrics: Dictionary of metrics to write.
    :param output_dir: Output directory.
    """
    output_dir = create_dir(output_dir)

    with open(f"{output_dir}/metrics.csv", 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerows(zip(*metrics.values()))


def process(config_path: str, output_dir: str, write: bool = True, plot: bool = True):
    config = load_config(config_path)

    n_samples = config['general']['num_samples']
    seed = config['general'].get('seed', None) or np.random.randint(0, 1000)
    metrics = generate_signals(config['metrics'], config["anomalies"], seed=seed, n_samples=n_samples)

    # create csv file with metrics
    if plot:
        plot_signal(metrics, output_dir)
    if write:
        write_to_csv(metrics, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic metrics.')
    parser.add_argument('--config', '-c', type=str, help='Path to config file.', required=True)
    parser.add_argument('--output', '-o', type=str, help='Output directory.', required=False, default='output')
    args = parser.parse_args()

    process(args.config, output_dir=args.output)
