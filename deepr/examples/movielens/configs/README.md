# MovieLens Experiments

## Data

Follow the instructions and use the notebook from [the Mult-VAE paper](https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb) to create the `.csv` files.

You can then update the `path_dataset` in `common/macros.jsonnet` to the directory containing the `.csv` files.

## Install

You need to install dependencies that are not listed in `deepr` (specific to the movielens example).

```
pandas
sklearn
faiss-cpu
```

## Average Model

To launch training

```
cd average
deepr run config.jsonnet macros.jsonnet
```
