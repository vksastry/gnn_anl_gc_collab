from threading import Event, Semaphore
from typing import List
from functools import partial, update_wrapper
from pathlib import Path
from datetime import datetime
from queue import Queue
import argparse
import logging
import hashlib
import json
import sys
import os

import rdkit
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from colmena.task_server import ParslTaskServer
from colmena.models import Result
from colmena.redis.queue import ClientQueues, make_queue_pairs
from colmena.thinker import BaseThinker, agent, result_processor, event_responder, task_submitter
from colmena.thinker.resources import ResourceCounter

from config import local_config
from moldesign.score.mpnn import evaluate_mpnn, update_mpnn, retrain_mpnn, MPNNMessage, custom_objects
from moldesign.store.models import MoleculeData
from moldesign.store.recipes import apply_recipes
from moldesign.utils import get_platform_info
from sim import run_simulation

# Disable all GPUs for the planning process
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'


class Thinker(BaseThinker):
    """ML-enhanced optimization loop for molecular design"""

    def __init__(self, queues: ClientQueues,
                 num_workers: int,
                 database: List[MoleculeData],
                 search_space: List[str],
                 n_to_evaluate: int,
                 n_complete_before_retrain: int,
                 retrain_from_initial: bool,
                 mpnns: List[tf.keras.Model],
                 inference_chunk_size: int,
                 output_dir: str,
                 beta: float,
                 random_seed: int,
                 output_property: str):
        """
        Args:
            queues: Queues used to communicate with the method server
            database: Link to the MongoDB instance used to store results
            search_space: List of InChI strings which define the search space
            n_complete_before_retrain: Number of simulations to complete before retraining
            retrain_from_initial: Whether to update the model or retrain it from initial weights
            mpnns: List of MPNNs to use for selecting samples
            output_dir: Where to write the output files
            beta: Amount to weight uncertainty in the activation function
            random_seed: Random seed for the model (re)trainings
            output_property: Name of the property to be computed
        """
        super().__init__(queues, ResourceCounter(num_workers, ['training', 'inference', 'simulation']), daemon=True)

        # Configuration for the run
        self.inference_chunk_size = inference_chunk_size
        self.n_complete_before_retrain = n_complete_before_retrain
        self.retrain_from_initial = retrain_from_initial
        self.mpnns = mpnns.copy()
        self.output_dir = Path(output_dir)
        self.beta = beta
        self.nodes_per_qc = 1
        self.random_seed = random_seed
        self.output_property = output_property

        # Get the initial database
        self.database = database
        self.logger.info(f'Populated an initial database of {len(self.database)} entries')

        # Get the target database size
        self.n_to_evaluate = n_to_evaluate
        self.target_size = n_to_evaluate + len(self.database)
        
        # List the molecules that have already been searched
        self.already_searched = set([d.identifier['inchi'] for d in self.database])

        # Prepare search space
        self.mols = search_space.copy()
        self.inference_chunks = np.array_split(self.mols, max(len(self.mols) // self.inference_chunk_size, 1))
        self.logger.info(f'Split {len(self.mols)} molecules into {len(self.inference_chunks)} chunks for inference')

        # Inter-thread communication stuff
        self.start_inference = Event()  # Mark that inference should start
        self.inference_finished = Event()  # Mark that inference has finished
        self.inference_slots = Semaphore(value=0)  # Number of slots available for inference tasks
        self.start_training = Event()   # Mark that retraining should start
        self.all_training_started = Event()  # Mark that training has begun
        self.task_queue = Queue()  # Holds a list of tasks for inference
        self.max_ml_workers = 1
        self.inference_batch = 0

        # Allocate maximum number of ML workers to inference, remainder to simulation
        self.rec.reallocate(None, 'inference', self.max_ml_workers)
        self.rec.acquire('inference', self.max_ml_workers)
        if num_workers > self.max_ml_workers:
            self.rec.reallocate(None, 'simulation', num_workers - self.max_ml_workers)
        
        # Mark the number of inference slots we have available
        for _ in range(self.rec.allocated_slots('inference') * 2):
            self.inference_slots.release()

    @agent
    def allocator(self):
        """Allocates resources to different tasks"""

        # Start off by running inference loop
        self.start_inference.set()
        
        # Wait until the inference tasks event to finish
        self.inference_finished.wait()
        self.inference_finished.clear()
        self.logger.info(f'Finished first round of inference')

        while not self.done.is_set():
            # Reallocate all resources from inference to QC tasks
            self.rec.reallocate('inference', 'simulation', self.rec.allocated_slots('inference'))
            self.inference_slots = Semaphore(value=0)  # Reset the number of inference slots to zero

            # Wait until QC tasks complete
            retrain_size = len(self.database) + self.n_complete_before_retrain
            self.logger.info(f'Waiting until database reaches {retrain_size}. Current size: {len(self.database)}')
            while len(self.database) < retrain_size:
                if self.done.wait(15):
                    return

            # Start the training process
            self.logger.info('Triggered retraining process. Beginning to allocate nodes from simulation to training')
            self.start_training.set()

            # Gather nodes for training until either training finishes or we have 1 node per model.
            n_allocated = 0
            self.all_training_started.clear()
            while not (self.all_training_started.is_set() or n_allocated >= self.max_ml_workers
                    or self.done.is_set()):
                if self.rec.reallocate('simulation', 'training', self.nodes_per_qc,
                                       cancel_if=self.all_training_started):
                    n_allocated += self.nodes_per_qc
            self.logger.info('All training tasks have been submitted.'
                             ' Waiting for them to finish before deallocating to inference')
            while not self.all_training_started.wait(timeout=15):
                if self.done.is_set():  # Exit if we have finished
                    return
            self.all_training_started.clear()

            # Allocate initial nodes for inference
            self.logger.info('Waiting for training tasks to complete.')
            n_to_reallocate = self.rec.allocated_slots('training')
            self.rec.reallocate('training', 'inference', n_to_reallocate)
            for _ in range(n_to_reallocate * 2):
                self.inference_slots.release()
            
            # Trigger inference
            self.logger.info('Beginning inference process. Will gradually scavenge nodes from simulation tasks')
            self.start_inference.set()
            while not (self.inference_finished.is_set() or self.done.is_set() or
                       self.rec.allocated_slots("inference") == self.max_ml_workers):
                # Request a block of nodes for inference
                acq_success = self.rec.reallocate('simulation', 'inference', self.nodes_per_qc,
                                                  cancel_if=self.inference_finished)
                self.rec.acquire('inference', self.nodes_per_qc)
                
                # Make them available to the task submission thread
                if acq_success:
                    self.logger.info(f'Allocated {self.nodes_per_qc} more nodes to inference')
                    for _ in range(self.nodes_per_qc * 2):  # 1 execution slot per worker
                        self.inference_slots.release()
            while not self.inference_finished.wait(timeout=15):
                if self.done.is_set():
                    return
            self.inference_finished.clear()
            self.rec.release('inference', self.rec.allocated_slots('inference'))
            self.logger.info(f'Completed inference and task selection')

    @task_submitter(task_type='simulation', n_slots=1)
    def launch_qc(self):

        # Submit the next task
        inchi, info = self.task_queue.get()
        mol = Chem.MolFromInchi(inchi)
        smiles = Chem.MolToSmiles(mol)
        self.logger.info(f'Submitted {smiles} to simulate with NWChem')
        self.already_searched.add(inchi)
        self.queues.send_inputs(smiles, task_info=info,
                                method='run_simulation', keep_inputs=True,
                                topic='simulate')

    @result_processor(topic='simulate')
    def record_qc(self, result: Result):
        # Get basic task information
        smiles, = result.args
        
        # Release nodes for use by other processes
        self.rec.release("simulation", self.nodes_per_qc)

        # If successful, add to the database
        if result.success:
            # Store the data in a molecule data object
            data = MoleculeData.from_identifier(smiles=smiles)
            opt_records, hess_records = result.value
            for r in opt_records:
                data.add_geometry(r)
            for r in hess_records:
                data.add_single_point(r)
            apply_recipes(data)  # Compute the IP

            # Add to database
            with open(self.output_dir.joinpath('moldata-records.json'), 'a') as fp:
                print(json.dumps([datetime.now().timestamp(), data.json()]), file=fp)
            self.database.append(data)
            
            # If the database is complete, set "done"
            if len(self.database) >= self.target_size:
                self.logger.info(f'Database has reached target size of {len(self.database)}. Exiting')
                self.done.set()

            # Write to disk
            with open(self.output_dir.joinpath('qcfractal-records.json'), 'a') as fp:
                for r in opt_records + hess_records:
                    print(r.json(), file=fp)
            self.logger.info(f'Added complete calculation for {smiles} to database.')
        else:
            self.logger.info(f'Computations failed for {smiles}. Check JSON file for stacktrace')

        # Write out the result to disk
        with open(self.output_dir.joinpath('simulation-results.json'), 'a') as fp:
            print(result.json(exclude={'value'}), file=fp)

    @event_responder(event_name='start_training')
    def train_models(self):
        """Train machine learning models"""
        self.start_training.clear()
        self.logger.info('Started retraining')

        for mid, model in enumerate(self.mpnns):
            # Wait until we have nodes
            if not self.rec.acquire('training', 1, cancel_if=self.done):
                # If unsuccessful, exit because we are finished
                return 

            # Make the database
            train_data = dict(
                (d.identifier['smiles'], d.oxidation_potential[self.output_property])
                for d in self.database
                if self.output_property in d.oxidation_potential
            )

            # Make the MPNN message
            if self.retrain_from_initial:
                self.queues.send_inputs(model.get_config(), train_data, method='retrain_mpnn', topic='train',
                                        task_info={'model_id': mid}, # , 'molecules': list(train_data.keys())}, 
                                        keep_inputs=False,
                                        input_kwargs={'random_state': mid + self.random_seed})
            else:
                model_msg = MPNNMessage(model)
                self.queues.send_inputs(model_msg, train_data, method='update_mpnn', topic='train',
                                        task_info={'model_id': mid}, #'molecules': list(train_data.keys())}, 
                                        keep_inputs=False,
                                        input_kwargs={'random_state': mid + self.random_seed})
            self.logger.info(f'Submitted model {mid} to train with {len(train_data)} entries')
        self.all_training_started.set()

    @result_processor(topic='train')
    def update_weights(self, result: Result):
        """Process the results of the saved model"""
        self.rec.release('training', 1)

        # Save results to disk
        with open(self.output_dir.joinpath('training-results.json'), 'a') as fp:
            print(result.json(exclude={'inputs', 'value'}), file=fp)

        # Make sure the run completed
        model_id = result.task_info['model_id']
        if not result.success:
            self.logger.warning(f'Training failed for {model_id}')
            return

        # Update weights
        weights, history = result.value
        self.mpnns[model_id].set_weights(weights)
        
        # Print out some status info
        self.logger.info(f'Model {model_id} finished training.')
        with open(self.output_dir.joinpath('training-history.json'), 'a') as fp:
            print(repr(history), file=fp)

    @event_responder(event_name='start_inference')
    def launch_inference(self):
        """Submit inference tasks for the yet-unlabelled samples"""

        self.logger.info('Beginning to submit inference tasks')
        # Make a folder for the models
        model_folder = self.output_dir.joinpath('models')
        model_folder.mkdir(exist_ok=True)

        # Submit the chunks to the workflow engine
        for mid, model in enumerate(self.mpnns):
            # Save the current model to disk
            model_path = model_folder.joinpath(f'model-{mid}-{self.inference_batch}.h5')
            model.save(model_path)
            
            # Read the model in
            for cid, chunk in enumerate(self.inference_chunks):
                self.inference_slots.acquire()  # Wait to get a slot
                self.queues.send_inputs([str(model_path)], chunk['smiles'].tolist(),
                                        topic='infer', method='evaluate_mpnn',
                                        keep_inputs=False,
                                        task_info={'chunk_id': cid, 'chunk_size': len(chunk), 'model_id': mid})
                self.logger.info(f'Submitted chunk {cid+1}/{len(self.inference_chunks)} for model {mid+1}/{len(self.mpnns)}')
        self.logger.info('Finished submitting molecules for inference')

    @event_responder(event_name='start_inference')
    def record_inference(self):
        """Re-prioritize the machine learning tasks"""
        self.start_inference.clear()
        
        #  Make arrays that will hold the output results from each run
        y_pred = [np.zeros((len(x), len(self.mpnns)), dtype=np.float32) for x in self.inference_chunks]

        # Collect the inference runs
        n_tasks = len(self.inference_chunks) * len(self.mpnns)
        for i in range(n_tasks):
            # Wait for a result
            result = self.queues.get_result(topic='infer')
            self.logger.info(f'Received inference task {i + 1}/{n_tasks}')

            # Free up resources to submit another
            self.inference_slots.release()
            
            # Save the inference information to disk
            with open(self.output_dir.joinpath('inference-records.json'), 'a') as fp:
                print(result.json(exclude={'value'}), file=fp)

            # Determine the data
            if not result.success:
                raise ValueError('Result failed! Check the JSON')

            # Store the outputs
            chunk_id = result.task_info.get('chunk_id')
            model_id = result.task_info.get('model_id')
            y_pred[chunk_id][:, model_id] = np.squeeze(result.value)
        self.logger.info('All inference tasks are complete')

        # Compute the mean and std for each prediction
        y_pred = np.concatenate(y_pred, axis=0)
        self._select_molecules(y_pred)

        # Free up resources
        self.rec.release('inference', self.rec.allocated_slots('inference'))

        # Mark that inference is complete
        self.inference_finished.set()
        self.inference_batch += 1

    def _select_molecules(self, y_pred):
        """Select a list of molecules given the predictions from each model

        Adds them to the task queue

        Args:
            y_pred: List of predictions for each molecule in self.search_space
        """
        # Compute the average and std of predictions
        y_mean = y_pred.mean(axis=1)
        y_std = y_pred.std(axis=1)

        # Rank compounds according to the upper confidence bound
        molecules = self.mols['inchi'].values
        ucb = y_mean + self.beta * y_std
        sort_ids = np.argsort(ucb)
        best_list = list(zip(molecules[sort_ids].tolist(),
                             y_mean[sort_ids], y_std[sort_ids], ucb[sort_ids]))

        # Make a temporary copy as a List, which means we can easily judge its size
        task_queue = []
        while len(task_queue) < self.n_to_evaluate * 2:
            # Pick a molecule
            mol, mean, std, ucb = best_list.pop()

            # Add it to list if not in database or not already in queue
            if mol not in self.already_searched and mol not in task_queue:
                # Note: converting to float b/c np.float32 is not JSON serializable
                task_queue.append((mol, {'mean': float(mean), 'std': float(std), 'ucb': float(ucb),
                                         'batch': self.inference_batch}))

        # Clear out the current queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get(False)
            except Empty:
                continue

        # Put new tasks in the queue
        for i in task_queue:
            self.task_queue.put(i)

        self.logger.info('Updated task list')

   
if __name__ == '__main__':
    # User inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--redishost", default="127.0.0.1",
                        help="Address at which the redis server can be reached")
    parser.add_argument("--redisport", default="6379",
                        help="Port on which redis is available")
    parser.add_argument('--mpnn-config-directory', help='Directory containing the MPNN-related JSON files',
                        required=True)
    parser.add_argument('--mpnn-model-files', nargs="+", help='Path to the MPNN h5 files', required=True)
    parser.add_argument('--init-dataset', default='init_dataset.json', help='Path to the initial dataset.')
    parser.add_argument('--retrain-from-scratch', action='store_true', help='Whether to retrain models from scratch')
    parser.add_argument('--search-space', help='Path to molecules to be screened', required=True)
    parser.add_argument("--search-size", default=1000, type=int,
                        help="Number of new molecules to evaluate during this search")
    parser.add_argument('--retrain-frequency', default=50, type=int,
                        help="Number of completed computations that will trigger a retraining")
    parser.add_argument("--molecules-per-ml-task", default=200000, type=int,
                        help="Number molecules per inference task")
    parser.add_argument("--ml-prefetch", default=1, help="Number of ML tasks to prefech on each node", type=int)
    parser.add_argument("--beta", default=1, help="Degree of exploration for active learning. "
                                                  "This is the beta from the UCB acquistion function", type=float)
    parser.add_argument("--learning-rate", default=3e-4, help="Initial Learning rate for re-training the models", type=float)
    parser.add_argument('--num-epochs', default=512, type=int, help='Maximum number of epochs for the model training')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for model training')
    parser.add_argument('--random-seed', default=0, type=int, help='Random seed for model (re)trainings')
    parser.add_argument('--dilation-factor', default=1, type=float,
                        help='Factor by which to artificially increase simulation time')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--train-timeout', default=None, type=float, help='Timeout for training operation (s)')
    parser.add_argument('--train-patience', default=None, type=int, help='Patience for training operation (epochs)')
    parser.add_argument('--solvent', default=None, choices=[None, 'acetonitrile'],
                        help='Whether to compute solvation energy in a solvent')

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Load in the models, initial dataset, agent and search space
    models = [tf.keras.models.load_model(path, custom_objects=custom_objects) for path in args.mpnn_model_files]
    with open(os.path.join(args.mpnn_config_directory, 'atom_types.json')) as fp:
        atom_types = json.load(fp)
    with open(os.path.join(args.mpnn_config_directory, 'bond_types.json')) as fp:
        bond_types = json.load(fp)

    # Load in the initial dataset
    with open(args.init_dataset) as fp:
        database = [MoleculeData.parse_raw(line.strip()) for line in fp]

    # Load in the search space
    def _only_known_elements(inchi: str):
        mol = rdkit.Chem.MolFromInchi(inchi)
        if mol is None:
            return False
        return all(
            e.GetAtomicNum() in atom_types for e in mol.GetAtoms()
        )
    full_search = pd.read_csv(args.search_space, delim_whitespace=True)
    search_space = full_search[full_search['inchi'].apply(_only_known_elements)]

    # Create an output directory with the time and run parameters
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = os.path.join('runs', f'ensemble-{start_time.strftime("%d%b%y-%H%M%S")}-{params_hash}')
    os.makedirs(out_dir, exist_ok=False)

    # Save the run parameters to disk
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)
    with open(os.path.join(out_dir, 'environment.json'), 'w') as fp:
        json.dump(dict(os.environ), fp, indent=2)

    # Save the platform information to disk
    host_info = get_platform_info()
    with open(os.path.join(out_dir, 'host_info.json'), 'w') as fp:
        json.dump(host_info, fp, indent=2)

    # Set up the logging
    handlers = [logging.FileHandler(os.path.join(out_dir, 'runtime.log')),
                logging.StreamHandler(sys.stdout)]

    class ParslFilter(logging.Filter):
        """Filter out Parsl debug logs"""

        def filter(self, record):
            return not (record.levelno == logging.DEBUG and '/parsl/' in record.pathname)

    for h in handlers:
        h.addFilter(ParslFilter())

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)

    # Write the configuration
    # ML nodes: N for updating models, 1 for MolDQN, 1 for inference runs
    config = local_config(os.path.join(out_dir, 'run-info'), args.num_workers)

    # Save Parsl configuration
    with open(os.path.join(out_dir, 'parsl_config.txt'), 'w') as fp:
        print(str(config), file=fp)

    # Connect to the redis server
    client_queues, server_queues = make_queue_pairs(args.redishost, args.redisport,
                                                    serialization_method="pickle",
                                                    topics=['simulate', 'infer', 'train'],
                                                    keep_inputs=False)

    # Apply wrappers to functions to affix static settings
    #  Update wrapper changes the __name__ field, which is used by the Method Server
    my_evaluate_mpnn = partial(evaluate_mpnn, atom_types=atom_types, bond_types=bond_types,
                               batch_size=32, cache=False)
    my_evaluate_mpnn = update_wrapper(my_evaluate_mpnn, evaluate_mpnn)
    
    my_update_mpnn = partial(update_mpnn, num_epochs=args.num_epochs, atom_types=atom_types, bond_types=bond_types,
                             learning_rate=args.learning_rate, bootstrap=True, batch_size=args.batch_size,
                             patience=args.train_patience, timeout=args.train_timeout)
    my_update_mpnn = update_wrapper(my_update_mpnn, update_mpnn)
    
    my_retrain_mpnn = partial(retrain_mpnn, num_epochs=args.num_epochs, atom_types=atom_types, bond_types=bond_types,
                              learning_rate=args.learning_rate, bootstrap=True, batch_size=args.batch_size,
                              patience=args.train_patience, timeout=args.train_timeout)

    my_retrain_mpnn = update_wrapper(my_retrain_mpnn, retrain_mpnn)

    my_run_simulation = partial(run_simulation, dilation_factor=args.dilation_factor, solvent=args.solvent)
    my_run_simulation = update_wrapper(my_run_simulation, run_simulation)

    # Get the name of the output property
    output_prop = {
        None: 'xtb-vacuum',
        'acetonitrile': 'xtb-acn'
    }[args.solvent]

    # Create the method server and task generator
    inf_cfg = {'executors': ['ml-worker']}
    tra_cfg = {'executors': ['ml-worker']}
    dft_cfg = {'executors': ['qc-worker']}
    doer = ParslTaskServer([(my_evaluate_mpnn, inf_cfg), (my_run_simulation, dft_cfg),
                            (my_update_mpnn, tra_cfg), (my_retrain_mpnn, tra_cfg)],
                           server_queues, config)

    # Configure the "thinker" application
    thinker = Thinker(client_queues,
                      args.num_workers,
                      database,
                      search_space,
                      args.search_size,
                      args.retrain_frequency,
                      args.retrain_from_scratch,
                      models,
                      args.molecules_per_ml_task,
                      out_dir,
                      args.beta,
                      args.random_seed,
                      output_prop)
    logging.info('Created the task server and task generator')

    try:
        # Launch the servers
        #  The method server is a Thread, so that it can access the Parsl DFK
        #  The task generator is a Thread, so that all debugging methods get cast to screen
        doer.start()
        thinker.start()
        logging.info(f'Running on {os.getpid()}')
        logging.info('Launched the servers')

        # Wait for the task generator to complete
        thinker.join()
        logging.info('Task generator has completed')
    finally:
        client_queues.send_kill_signal()

    # Wait for the method server to complete
    doer.join()
