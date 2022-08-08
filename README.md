# MPNN Training

Scripts for training a baseline MPNN using the [nfp](https://github.com/NREL/nfp) package.
Designed to simplify testing the networks on different systems before integrating them into our
[multi-site molecular design application](https://github.com/exalearn/multi-site-campaigns).

We provide a testing script that allows you to change basic parameters of an architecture that relies on
a global state variable. For example, you can also turn on a "maximum atom count" for the data loader (`--padded-size`)
to ensure all batches have the same input shape.

## Installation

The environment is described in [`environment-minimal.yml`](./environment-minimal.yml) and can be installed using
Anaconda.

```bash
conda env create --file enviroment-minimal.yml --force
```

The installation script will install a reduced set of the utilities used in our application, the `moldesign` package.
If you install the environment using another mechanism, those utilities can be installed using `pip install -e .`

## Directory layout

The most important file in this repository is [`run_test.py`](./run_test.py), which holds
the code which trains the model and outputs the results in a form that is easy to evaluate later.

The training datasets are stored in [`data`](./data) and we have one directory per dataset source.

The results from model training are stored in [`networks`](./networks) with one directory per training
and names that are automatically generated based on the inputs.

## Running the Example

The `run_test.py` file trains a machine learning model and stores the results of the performance of the
model on a test set in the `networks` directory.

Run it with the default options by calling `python run_test.py`

There are configurable options, such as changing the batch size, that can be changed by providing command line arguments
(e.g., `python run_test.py --batch-size 32`).
Changing the inputs will result in a different save directory for the outputs.

`run_test.py` is built using argparse, so calling `python run_test.py` will produce all available options.

> Avoid changing the hard-coded parameters to change experiments
>
> `run_test.py` provides the ability for us to track the changes over time and doing so manually will inhibit our
> ability to reproduce results later.

### Adding more adjustable parameters

There will certainly be settings that are hard-coded now that we'll want to vary as we develop the network.
First add them to the argument parse class at the beginning.
They will then be included in the experiment hash and saved to `config.json`.

If the parameter changes the network architecture, include it as an argument to `make_model`. 

The goal is to make everything changeable without altering the source code between runs.
