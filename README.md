# Chaos Experiment Data Generator

## Usage
    
```bash
python generate.py -c configs/config.toml -o output
```

## Config File

The config file is a json file with the following structure:

### Structure

```
[general]
seed = <int>                       # seed for random number generator
num_samples = <int>                # number of samples to generate

[metrics]

    [metrics.<metric_name>]
    value = <float>                # base value of metric
    step_size = <float>            # step size of metric
    noise = <str>                  # name of noise function
    noise_args = <dict>            # arguments for the noise function

[anomalies]
    
        [anomalies.<metric_name>]
        at = <int>                 # time at which anomaly occurs
        anomaly_factor = <float>   # factor of signal at anomaly
        pattern = <str>            # pattern of the anomaly
        recovery_time = <int>      # time to recover from anomaly
        pattern_args = <dict>      # arguments for the pattern
```

### Noise Functions
Can be one of the following:
- gaussian
- uniform
- exponential
- logistic
- laplace

Parameters for the numpy noise functions are passed as a dictionary to the `noise_args` field.

### Anomaly Patterns
Can be one of the following:
- linear
- exponential
- oscillating
- step

Parameters for the anomaly patterns are passed as a dictionary to the `pattern_args` field.
The `frequency` parameter is only used for the `oscillating` pattern.



### Example

```toml
# Configuration for chaos experiment data generator

[general]
num_samples = 150
seed = 42

[metrics]

    [metrics.instances]
    value = 4
    step_size=1
    noise = "exponential"
    noise_args = {"scale" = 0.48, "smoothing_window" = 30}

    [metrics.cpu_usage]
    value = 60
    step_size=0.1
    noise = "gaussian"
    noise_args = {"loc" = 0, "scale" = 1, "smoothing_window" = 10}

    [metrics.memory_usage_mb]
    value = 2000
    step_size=0.01
    noise = "logistic"
    noise_args = {"loc" = 0, "scale" = 10, "smoothing_window" = 60}

    [metrics.response_time_s]
    value = 0.006
    step_size=0.00001
    noise = "exponential"
    noise_args = {"scale" = 0.0001, "smoothing_window" = 1}

[anomalies]

    [anomalies.instances]
    at = 40
    anomaly_factor = 2
    pattern = "step"
    recovery_time = 20

    [anomalies.cpu_usage]
    at = 35
    anomaly_factor = 1.3
    pattern = "exponential"
    recovery_time = 30

    [anomalies.memory_usage_mb]
    at = 10
    anomaly_factor = 0.4
    pattern = "oscillating"
    recovery_time = 40
    pattern_args = {"frequency" = 0.3}

    [anomalies.response_time_s]
    at = 12
    anomaly_factor = 2
    pattern = "oscillating"
    recovery_time = 50
    pattern_args = {"frequency" = 0.25}
```