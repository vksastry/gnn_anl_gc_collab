"""Train the model on a pre-defined training set and options provided by a user"""
from argparse import ArgumentParser
from pathlib import Path
from typing import List
from time import perf_counter
import hashlib
import json

try:
    from tensorflow.python import ipu
except ImportError:
    print('Cannot find the `ipu` library for Tensorflow')
    ipu = None

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks as cb
from tensorflow.keras import layers
import tensorflow as tf
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import numpy as np
import nfp

from moldesign.nfp import make_data_loader


def build_fn(atom_features: int = 64,
             message_steps: int = 8,
             output_layers: List[int] = (512, 256, 128)):
    """Construct a Keras model using the settings provided by a user

    Args:
        atom_features: Number of features used per atom and bond
        message_steps: Number of message passing steps
        output_layers: Number of neurons in the readout layers

    Returns:

    """
    atom = layers.Input(shape=[None], dtype=tf.int32, name='atom')
    bond = layers.Input(shape=[None], dtype=tf.int32, name='bond')
    connectivity = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity')

    # Convert from a single integer defining the atom state to a vector
    # of weights associated with that class
    atom_state = layers.Embedding(36, atom_features, name='atom_embedding', mask_zero=True)(atom)

    # Ditto with the bond state
    bond_state = layers.Embedding(5, atom_features, name='bond_embedding', mask_zero=True)(bond)

    # Here we use our first nfp layer. This is an attention layer that looks at
    # the atom and bond states and reduces them to a single, graph-level vector.
    # mum_heads * units has to be the same dimension as the atom / bond dimension
    global_state = nfp.GlobalUpdate(units=4, num_heads=1, name='problem')([atom_state, bond_state, connectivity])

    for _ in range(message_steps):  # Do the message passing
        new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
        bond_state = layers.Add()([bond_state, new_bond_state])

        new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
        atom_state = layers.Add()([atom_state, new_atom_state])

        new_global_state = nfp.GlobalUpdate(units=4, num_heads=1)(
            [atom_state, bond_state, connectivity, global_state]
        )
        global_state = layers.Add()([global_state, new_global_state])

    # Pass the global state through an output
    output = atom_state
    for shape in output_layers:
        output = layers.Dense(shape, activation='relu')(output)
    output = layers.Dense(1)(output)
    output = layers.Dense(1, activation='linear', name='scale')(output)
    output = layers.Lambda(tf.math.reduce_sum, arguments={'axis': 1})(output)

    # Construct the tf.keras model
    return tf.keras.Model([atom, bond, connectivity], [output])


if __name__ == "__main__":
    # Define the command line arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--atom-features', help='Number of atomic features', type=int, default=32)
    arg_parser.add_argument('--num-messages', help='Number of message-passing layers', type=int, default=8)
    arg_parser.add_argument('--output-layers', help='Number of hidden units of the output layers', type=int,
                            default=(512, 256, 128), nargs='*')
    arg_parser.add_argument('--batch-size', help='Number of molecules per batch', type=int, default=16)
    arg_parser.add_argument('--num-epochs', help='Number of epochs to run', type=int, default=64)
    arg_parser.add_argument('--padded-size', help='Maximum number of atoms per molecule', type=int, default=None)
    arg_parser.add_argument('--dataset', choices=['qm9', 'redox'], help='Which dataset to use for training', default='qm9')
    arg_parser.add_argument('--system', choices=['gpu', 'ipu'], help='Which system to use for training', default='gpu')
    arg_parser.add_argument('--lr-start', default=1e-3, help='Learning rate at start of training', type=float)

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]

    # Determine the output directory
    test_dir = Path('networks') / f'{args.dataset}-{args.system}_b{args.batch_size}_n{args.num_epochs}_{params_hash}'
    test_dir.mkdir(parents=True)
    with open(test_dir / 'config.json', 'w') as fp:
        json.dump(run_params, fp)

    # Configuration he
    if args.system == 'ipu':
        #  Configure the IPU system and define the strategy
        cfg = ipu.config.IPUConfig()
        cfg.auto_select_ipus = 1
        cfg.configure_ipu_system()

        strategy = ipu.ipu_strategy.IPUStrategy(enable_dataset_iterators=True)
    elif args.system == 'gpu':
        # Distribute over all available GPUs
        strategy = tf.distribute.MirroredStrategy()

        # Print the GPU list
        device_details = [
            tf.config.experimental.get_device_details(x)
            for x in tf.config.get_visible_devices('GPU')
        ]
        with open(test_dir / 'gpus.json', 'w') as fp:
            json.dump(device_details, fp)
    else:
        raise ValueError(f'System {args.system} not supported yet')

    # Making the data loaders
    data_dir = Path('data') / args.dataset
    train_data = pd.read_csv(data_dir / 'train.csv')
    train_loader = make_data_loader(train_data['smiles'], train_data['output'], shuffle_buffer=32768, repeat=True,
                                    batch_size=args.batch_size, max_size=args.padded_size, drop_last_batch=True)
    steps_per_epoch = len(train_data) // args.batch_size

    test_data = pd.read_csv(data_dir / 'test.csv')
    test_loader = make_data_loader(test_data['smiles'], test_data['output'], batch_size=args.batch_size,
                                   max_size=args.padded_size, drop_last_batch=True)

    valid_data = pd.read_csv(data_dir / 'valid.csv')
    valid_loader = make_data_loader(valid_data['smiles'], valid_data['output'], batch_size=args.batch_size,
                                    max_size=args.padded_size, drop_last_batch=True)

    # Determine the amount of scaling to provide
    y_train = train_data['output']
    y_scale = y_train / train_data['n_atom']  # Output layer scales atomic contributions for atomic-contrib networks

    y_scale_mean = y_scale.mean()
    y_scale_std = y_scale.std()

    with strategy.scope():
        # Make the model
        model = build_fn(atom_features=args.atom_features, message_steps=args.num_messages,
                         output_layers=args.output_layers)

        # Set the scale for the output parameter
        model.get_layer('scale').set_weights([np.array([[y_scale_mean]]), np.array([y_scale_std])])

        # Configure the LR schedule
        init_learn_rate = args.lr_start
        final_learn_rate = init_learn_rate * 1e-3
        decay_rate = (final_learn_rate / init_learn_rate) ** (1. / (args.num_epochs - 1))


        def lr_schedule(epoch, lr):
            return lr * decay_rate


        # Compile the model then train
        model.compile(Adam(init_learn_rate), 'mean_squared_error', metrics=['mean_absolute_error'])
        start_time = perf_counter()
        history = model.fit(
            train_loader, validation_data=valid_loader, epochs=args.num_epochs, verbose=True,
            shuffle=False,
            callbacks=[
                cb.LearningRateScheduler(lr_schedule),
                cb.ModelCheckpoint(test_dir / 'best_model.h5', save_best_only=True),
                # We restart the best weights, but do not halt early to simplify timing across
                cb.EarlyStopping(patience=args.num_epochs, restore_best_weights=True),
                cb.CSVLogger(test_dir / 'train_log.csv'),
                cb.TerminateOnNaN()
            ],
            steps_per_epoch=steps_per_epoch
        )
        run_time = perf_counter() - start_time

        # Run on the validation set and assess statistics
        y_true = np.hstack([np.squeeze(x[1].numpy()) for x in iter(test_loader)])
        y_pred = np.squeeze(model.predict(test_loader))

        pd.DataFrame({'true': y_true, 'pred': y_pred}).to_csv(test_dir / 'test_results.csv', index=False)

        with open(test_dir / 'test_summary.json', 'w') as fp:
            json.dump({
                'runtime': run_time,
                'r2_score': float(np.corrcoef(y_true, y_pred)[1, 0] ** 2),  # float() converts from np.float32
                'spearmanr': float(spearmanr(y_true, y_pred)[0]),
                'kendall_tau': float(kendalltau(y_true, y_pred)[0]),
                'mae': float(np.mean(np.abs(y_pred - y_true))),
                'rmse': float(np.sqrt(np.mean(np.square(y_pred - y_true))))
            }, fp, indent=2)
