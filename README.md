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

### Example

```toml
# Configuration for chaos experiment data generator

[general]
seed = 1234
num_samples = 1000

[metrics]

    [metrics.metric_1]
    value = 30
    noise = "gaussian"
    noise_args = {"loc" = 0, "scale" = 0.5}

    [metrics.metric_2]
    value = 100
    noise = "uniform"
    noise_args = {"low" = 0, "high" = 0.1}

    [metrics.metric_3]
    value = 5
    noise = "exponential"
    noise_args = {"scale" = 0.1}

    [metrics.metric_4]
    value = 10000
    noise = "gaussian"
    noise_args = {"loc" = 0, "scale" = 100}

    [metrics.metric_5]
    value = 100
    noise = "logistic"
    noise_args = {"loc" = 0, "scale" = 0.2}

[anomalies]

    [anomalies.metric_1]
    at = 400
    anomaly_factor = 2
    pattern = "linear"
    recovery_time = 100

    [anomalies.metric_2]
    at = 200
    anomaly_factor = 1.1
    pattern = "exponential"
    recovery_time = 300

    [anomalies.metric_3]
    at = 300
    anomaly_factor = 0.5
    pattern = "exponential"
    recovery_time = 200

    [anomalies.metric_4]
    at = 300
    anomaly_factor = 0.8
    pattern = "oscillating"
    recovery_time = 500
    pattern_args = {"frequency" = 0.5}

    [anomalies.metric_5]
    at = 100
    anomaly_factor = 1.2
    pattern = "step"
    recovery_time = 30



```