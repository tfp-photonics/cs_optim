# CSOptim

Gradient-based optimization of core-shell particles with distinct material classes. For detailed information please check the [article]()


## Requirements

Setup a suitable [conda](https://docs.conda.io/en/latest/) environment named `cs_optim` with:

```
conda env create -f environment.yaml
conda activate cs_optim
```

## Run Optimization

You can run an Optimization with the `main.py` file. The code includes the generation of target spectra of bare core and core-single-shell particles with [PyMieScatt](https://pymiescatt.readthedocs.io/en/latest/). To target single or mutliple frequencies add pass them as `-enhancement` and add linewidths with `-width`.

## Model Training

