# MPNN Training

Scripts for training a baseline MPNN using the [nfp](https://github.com/NREL/nfp) package.
Designed to simplify testing the networks on different systems before integrating them into our
[multi-site molecular design application](https://github.com/exalearn/multi-site-campaigns).

We provide a testing script that allows you to change basic parameters of an architecture that relies on 
a global state variable. For example, you can also turn on a "maximum atom count" for the data loader (`--padded-size`)
to ensure all batches have the same input shape.

## Installation 

The environment is described in [`environment-minimal.yml`](./environment-minimal.yml) and can be installed using Anaconda.

```bash
conda env create --file enviroment-minimal.yml --force
```
The installation script will install a reduced set of the utilities used in our application, the `moldesign` package. 
If you install the environment using another mechanism, those utilities can be installed using `pip install -e .`

## Directory layout

TBD

## Running the Example

TBD
