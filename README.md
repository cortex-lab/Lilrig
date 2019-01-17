# Lilrig
code specific to Lilrig

dependencies: [widefield repository](https://github.com/cortex-lab/widefield/)

# Load data from lilrig
```animal = 'animal_name';```

```day = 'yyyy-mm-dd';```

```experiment = experiment number;```

```lilrig_load_experiment;```

# Get visual field sign retinotopy
After loading data: 

```lilrig_retinotopy```

# Troubleshooting
- if timestamps are not saved after preprocessing, troubleshoot with the script AP_parse_temporal_component_fix (this can make the .timestamps.npy files if they're missing although they're not used anymore, it can also separate the svdTemporalComponents by experiment if they were parsed incorrectly)
