import pickle as pkl
import argparse
import hashlib
import json
import sys
import logging
import os
from collections import deque
from random import sample, choice, shuffle, random
from datetime import datetime
from functools import partial, update_wrapper
from queue import Queue, Empty
from threading import Event, Lock
from typing import List, Dict

import tensorflow as tf
from pydantic import BaseModel
from moldesign.sample.rl.agents.moldqn import DQNFinalState
from moldesign.sample.rl.envs.rewards.mpnn import MPNNReward
from moldesign.score.mpnn.layers import custom_objects
from moldesign.score.mpnn import evaluate_mpnn, update_mpnn, MPNNMessage
from moldesign.config import theta_nwchem_config, theta_xtb_config
from moldesign.sample.rl import generate_molecules
from moldesign.simulate.functions import compute_atomization_energy
from moldesign.simulate.specs import lookup_reference_energies, get_qcinput_specification
from moldesign.utils import get_platform_info

from colmena.thinker import BaseThinker, agent
from colmena.method_server import ParslMethodServer
from colmena.redis.queue import ClientQueues, make_queue_pairs


class Thinker(BaseThinker):
    """ML-enhanced optimization loop for molecular design"""

    def __init__(self, queues: ClientQueues,
                 initial_training_set: Dict[str, float],
                 initial_search_space: List[str],
                 initial_moldqn: DQNFinalState,
                 initial_mpnns: List[tf.keras.Model],
                 output_dir: str,
                 n_parallel: int = 1,
                 n_parallel_updating: int = 1,
                 n_molecules: int = 10,
                 queue_length: int = None,
                 random_frac: float = 0.1,
                 greedy_frac: float = 0.8):
        """
        Args:
            queues (ClientQueues): Queues to use to communicate with server
            initial_training_set: List of molecules and atomization energies from the original search
            initial_search_space: List of molecules to use in the initial search space
            initial_moldqn: Pre-trained version of the MolDQN agent
            initial_mpnns: An ensemble of pre-trained MPNNs
            output_dir (str): Path to the run directory
            n_parallel (int): Maximum number of QC calculations to perform in parallel
            n_parallel_updating (int): Maximum number of model updates to perform in parallel
            n_molecules: Number of molecules to evaluate
            queue_length (int): Number of tasks to store in the queue at most
            random_frac: Number of molecules to pick at random
            greedy_frac: Number of molecules to pick greedly
        """
        super().__init__(queues, daemon=True)

        # Generic stuff: logging, communication to Method Server
        self.queues = queues
        self.output_dir = output_dir
        
        # The ML components
        self.moldqn = initial_moldqn
        init_mols = list(initial_training_set.keys())
        self.mpnns: List[List[tf.keras.Model, List[str]]] = list([x, init_mols] for x in initial_mpnns)
        
        # Active learning settings
        self.random_frac = random_frac
        self.greedy_frac = greedy_frac

        # Attributes associated with quantum chemistry calculations
        # TODO (wardlt): Use QCFractal or another database system instead of volatile in-memory databases
        self.database = initial_training_set.copy()
        self.search_space = initial_search_space

        # Attributes associated with the parallelism/problem size
        self.n_evals = n_molecules + len(self.database)
        self.n_parallel = n_parallel
        self.n_parallel_updating = n_parallel_updating

        # Synchronization between the threads
        if queue_length is None:
            queue_length = n_parallel * 2
        self.queue_length = queue_length
        self._task_queue = Queue(maxsize=queue_length)
        self._update_lock = Lock()  # Prevent models from being used while updating one
        self._done = Event()

    def _write_result(self, result: BaseModel, filename: str, keep_inputs: bool = True, keep_outputs: bool = True):
        """Write result to a log file

        Args:
            result: Result to be written
            filename: Name of the log file
            keep_inputs: Whether to write the function inputs
            keep_outputs: Whether to write the function outputs
        """

        # Determine which fields to dumb
        exclude = set()
        if not keep_inputs:
            exclude.add('inputs')
        if not keep_outputs:
            exclude.add('value')

        # Write it out
        with open(os.path.join(self.output_dir, filename), 'a') as fp:
            print(result.json(exclude=exclude), file=fp)

    @agent
    def simulation_dispatcher(self):
        """Submit and process simulation tasks"""

        self.logger.info('Simulation dispatcher waiting for work')
        for i in range(self.n_parallel):
            smiles, task_info = self._task_queue.get(block=True)
            if smiles in self.search_space:
                self.search_space.remove(smiles)
            self.queues.send_inputs(smiles, topic='simulate', method='compute_atomization_energy', keep_inputs=True,
                                    task_info=task_info)
        self.logger.info('Sent out first set of tasks')

        # As they come back submit new ones
        self.logger.info(f'Running until database has {self.n_evals} entries')
        while len(self.database) < self.n_evals and not self._done.is_set():
            # Get the task and store its content
            result = self.queues.get_result(topic='simulate')
            self.logger.info('Retrieved completed QC task')

            # Get a new one from the priority queue and submit it
            smiles, task_info = self._task_queue.get()
            if smiles in self.search_space:
                self.search_space.remove(smiles)
            self.logger.info(f'Submitted {smiles} from batch {task_info["batch"]}')
            self.queues.send_inputs(smiles, topic='simulate', method='compute_atomization_energy', keep_inputs=True,
                                    task_info=task_info)

            # Store the content from the previous run
            if result.success:
                # Store the result in the database
                self.database[result.args[0]] = result.value[0]  # First arg is the energy
                
                # Save the data
                self._write_result(result.value[1], 'qcfractal_records.jsonld')
                if result.value[2] is not None:
                    self._write_result(result.value[2], 'qcfractal_records.jsonld')
                result.value = result.value[0]  # Do not store the full results in the database
            else:
                self.logger.warning('Calculation failed! See simulation outputs and Parsl log file')
            self._write_result(result, 'simulation_records.jsonld', keep_outputs=True)

        # Mark that we are done (no longer submitting new simulations)
        self._done.set()

        # Waiting for the still-ongoing tasks to complete
        self.logger.info('Collecting the last molecules')
        for i in range(self.n_parallel):
            # Get the task and store its content
            result = self.queues.get_result(topic='simulate')
            self.logger.info(f'Retrieved {i+1}/{self.n_parallel} on-going tasks')
            if result.success:
                # Store the result in the database
                self.database[result.args[0]] = result.value[0]  # First arg is the energy

                self._write_result(result.value[1], 'qcfractal_records.jsonld')
                if result.value[2] is not None:
                    self._write_result(result.value[2], 'qcfractal_records.jsonld')
                result.value = result.value[0]  # Do not store the full results in the database
            else:
                self.logger.warning('Calculation failed! See simulation outputs and Parsl log file')
            self._write_result(result, 'simulation_records.jsonld', keep_outputs=True)

    @agent
    def model_updater(self):
        """Handle updating the ML models"""

        # Randomly order the MPNNs
        ready_to_retrain = list(range(len(self.mpnns)))
        shuffle(ready_to_retrain)
        ready_to_retrain = deque(ready_to_retrain)

        # Launch the first models to be updated
        for _ in range(self.n_parallel_updating):
            ind = ready_to_retrain.popleft()
            mpnn = self.mpnns[ind]
            self.queues.send_inputs(MPNNMessage(mpnn[0]), self.database, 4,
                                    method='update_mpnn', topic='update',
                                    task_info={'index': ind, 'training_molecules': list(self.database.keys())})
            self.logger.info(f'Submitted model {ind} to be updated')
            
        # Make a directory to store updated models
        model_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)

        # Continually wait for new models to come back
        result_ind = 0
        while not self._done.is_set():
            # Wait for a model to be returned
            result = self.queues.get_result(topic='update')

            # Update the weights
            complted_ind = result.task_info['index']
            if result.success:
                new_weights, _ = result.value
                with self._update_lock:
                    self.mpnns[complted_ind][0].set_weights(new_weights)
                    self.mpnns[complted_ind][1] = result.task_info['training_molecules']
                self.logger.info(f'Updated weights for model {complted_ind}')
            else:
                self.logger.info(f'Retraining failed for model {complted_ind}')

            # Mark the model as ready to be updated again
            ready_to_retrain.append(complted_ind)
            
            # Submit another model to be updated
            ind = ready_to_retrain.popleft()
            mpnn = self.mpnns[ind]
            self.queues.send_inputs(MPNNMessage(mpnn[0]), self.database, 4,
                                    method='update_mpnn', topic='update',
                                    task_info={'index': ind, 'training_molecules': list(self.database.keys())})
            self.logger.info(f'Submitted model {ind} to be updated')

            # Save the results
            self._write_result(result, 'update_records.jsonld', keep_inputs=False, keep_outputs=False)
            
            # If the updated model, if re-training was successful
            result_ind += 1
            if result.success:
                model_name = os.path.join(model_dir, f'{result_ind}_model_{complted_ind}.h5')
                self.logger.info(f'Saving model {complted_ind} to disk as {model_name}')
                self.mpnns[complted_ind][0].save(model_name, include_optimizer=False)
                self.logger.info('Model saved. Waiting for next update task to complete')

    @agent
    def task_ranker(self):
        """Prioritize list of available tasks"""

        # Submit some initial molecules so that the simulator gets started immediately
        num_to_seed = self.queue_length
        self.logger.info(f'Sending {num_to_seed} initial molecules')
        for smiles in sample(self.search_space, num_to_seed):
            # We send: (rank info), smiles, task_info
            self._task_queue.put((smiles, {'reason': 'initial', 'batch': -1, 'smiles': smiles}))

        # Perform the design loop iteratively
        batch_number = 0
        while not self._done.is_set():
            # Get the current copy of the search space
            search_space = self.search_space.copy()

            # Assign them scores
            with self._update_lock:
                self.queues.send_inputs([MPNNMessage(m) for m, _ in self.mpnns],
                                        search_space, method='evaluate_mpnn', topic='rank')

                # Capture the training set of models used in this inference run
                training_sets = [s.copy() for _, s in self.mpnns]

            self.logger.info(f'Submitted inference task')
            result = self.queues.get_result(topic='rank')
            scores = result.value
            result.task_info = {'training_sets': training_sets}  # Record the training sets
            self._write_result(result, 'screen_records.jsonld', keep_inputs=False, keep_outputs=False)
            self.logger.info(f'Assigned scores to all {len(scores)} molecules')

            # Assign scores to each SMILES
            mean_score = scores.mean(axis=1)
            std_score = scores.std(axis=1)
            task_options = [{'smiles': s, 'pred': float(m), 'pred_std': float(u), 'batch': batch_number}
                            for s, m, u in zip(search_space, mean_score, std_score)]

            # Rank according to different metrics. Best at the right end (so .pop works)
            random_selections = task_options.copy()
            shuffle(random_selections)
            greedy_selections = sorted(task_options, key=lambda x: -x['pred'])
            uq_selections = sorted(task_options, key=lambda x: x['pred_std'])
            self.logger.info('Sorted molecules by greedy, random and uncertainty selection.')

            # Pick enough to fill the queue
            already_picked = set()
            selections = []
            while len(already_picked) < self.queue_length:
                # Make sure none of the lists are empty
                if min(map(len, [greedy_selections, random_selections, uq_selections])) == 0:
                    self.logger.info('Ran out of molecules to select from')
                    break

                # Pick a task
                r = random()
                if r < self.greedy_frac:
                    task = greedy_selections.pop()
                    task['reason'] = 'greedy'
                elif r < self.greedy_frac + self.random_frac:
                    task = random_selections.pop()
                    task['reason'] = 'random'
                else:
                    task = uq_selections.pop()
                    task['reason'] = 'uq'

                # If it is not yet selected
                if (task['smiles'] not in already_picked
                   and task['smiles'] not in self.database):
                    already_picked.add(task['smiles'])
                    selections.append(task)
            self.logger.info(f'Selected {len(selections)} new molecules')

            # Clear out the queue
            while not self._task_queue.empty():
                try:
                    self._task_queue.get_nowait()
                except Empty:
                    break
            self.logger.info('Cleared out the current queue')

            # Add requested simulations to the queue
            for rank, task in enumerate(selections):
                self._task_queue.put((task['smiles'], task))
            batch_number += 1  # Increment the loop
            self.logger.info('Added all of them the task queue')

    @agent
    def task_generator(self):
        """Run RL to generate new candidates and use MPNNs to screen them"""
        while not self._done.is_set():
            # Use RL to generate new molecules
            with self._update_lock:
                self.moldqn.env.reward_fn.model = choice(self.mpnns)[0]
                self.queues.send_inputs(self.moldqn, method='generate_molecules', topic='generate')
            self.logger.info("Submitted task generator")

            # Record the result
            result = self.queues.get_result(topic='generate')
            self._write_result(result, 'generate_records.jsonld', keep_inputs=False, keep_outputs=False)
            
            # Update the list of molecules
            if result.success:
                new_molecules, self.moldqn = result.value  # Also update the RL agent
                self.logger.info(f'Generated {len(new_molecules)} candidate molecules')

                self.search_space = list(set(self.search_space).union(new_molecules).difference(self.database.keys()))
                self.logger.info(f'Search space now includes {len(self.search_space)} molecules')
            else:
                self.logger.info('Generation task failed. Resubmitting')


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
    parser.add_argument('--initial-agent', help='Path to the pickle file for the MolDQN agent', required=True)
    parser.add_argument('--initial-search-space', help='Path to an initial population of molecules', required=True)
    parser.add_argument('--initial-database', help='Path to the database used to train the MPNN', required=True)
    parser.add_argument('--qc-spec', help='Name of the QC specification', required=True,
                        choices=['normal_basis', 'xtb', 'small_basis'])
    parser.add_argument('--qc-parallelism', help='Degree of parallelism for QC tasks. For NWChem, number of nodes per task.'
                        ' For XTB, number of tasks per node.', default=1, type=int)
    parser.add_argument("--parallel-updating", default=2, type=int,
                        help="Number of model retraining to perform in parallel")
    parser.add_argument("--rl-episodes", default=10, type=int,
                        help="Number of episodes to run ing the reinforcement learning pipeline")
    parser.add_argument("--search-size", default=1000, type=int,
                        help="Number of new molecules to evaluate during this search")
    parser.add_argument('--queue-length', default=100, type=int, help="Number of molecules to hold in queue")
    parser.add_argument('--random-frac', default=0.1, type=float, help="Number of new molecules to pick randomly")
    parser.add_argument('--greedy-frac', default=0.8, type=float, help="Number of new molecules to pick greedly")

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Allocate nodes to ML and QC tasks
    nnodes = int(os.environ.get("COBALT_JOBSIZE", "1"))
    # ML nodes: N for updating models, 1 for MolDQN, 1 for inference runs
    ml_nodes = args.parallel_updating + 2
    # QC nodes: Whatever remains
    qc_nodes = nnodes - ml_nodes
    run_params["nnodes"] = nnodes
    run_params["ml_nodes"] = ml_nodes
    run_params["qc_nodes"] = qc_nodes

    # Determine the number of QC workers or threads per worker
    compute_config = {'nnodes': args.qc_parallelism, 'cores_per_rank': 2}
    if args.qc_spec == "xtb":
        qc_workers = nnodes * args.qc_parallelism
        compute_config["ncores"] = 64 // args.qc_parallelism
    else:
        qc_workers = qc_nodes // args.qc_parallelism
    run_params["qc_workers"] = qc_workers
    
    # Load in the models, initial dataset, agent and search space
    mpnns = [
        tf.keras.models.load_model(path, custom_objects=custom_objects)
        for path in args.mpnn_model_files
    ]                                 
    with open(os.path.join(args.mpnn_config_directory, 'atom_types.json')) as fp:
        atom_types = json.load(fp)
    with open(os.path.join(args.mpnn_config_directory, 'bond_types.json')) as fp:
        bond_types = json.load(fp)
    with open(args.initial_database) as fp:
        initial_database = json.load(fp)
    with open(args.initial_search_space) as fp:
        initial_search_space = json.load(fp)
    with open(args.initial_agent, 'rb') as fp:
        agent = pkl.load(fp)

    # Get QC specification
    qc_spec, code = get_qcinput_specification(args.qc_spec)
    if args.qc_spec != "xtb":
        qc_spec.keywords["dft__iterations"] = 150
        qc_spec.keywords["geometry__noautoz"] = True
    ref_energies = lookup_reference_energies(args.qc_spec)

    # Make the reward function
    agent.env.reward_fn = MPNNReward(mpnns[0], atom_types, bond_types, maximize=False)

    # Create an output directory with the time and run parameters
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = os.path.join('runs', f'{start_time.strftime("%d%b%y-%H%M%S")}-{params_hash}')
    os.makedirs(out_dir, exist_ok=True)

    # Save the run parameters to disk
    run_params['version'] = 'simple'
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)
    with open(os.path.join(out_dir, 'qc_spec.json'), 'w') as fp:
        print(qc_spec.json(), file=fp)
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
    if args.qc_spec == "xtb":
        config = theta_xtb_config(2, os.path.join(out_dir, 'run-info'), xtb_per_node=args.qc_parallelism,
                                  ml_tasks_per_node=2)
    else:

        config = theta_nwchem_config(ml_nodes, os.path.join(out_dir, 'run-info'),
                                     nodes_per_nwchem=args.qc_parallelism)

    # Save Parsl configuration
    with open(os.path.join(out_dir, 'parsl_config.txt'), 'w') as fp:
        print(str(config), file=fp)

    # Connect to the redis server
    client_queues, server_queues = make_queue_pairs(args.redishost, args.redisport,
                                                    serialization_method="pickle",
                                                    topics=['simulate', 'update', 'generate', 'rank'],
                                                    keep_inputs=False)

    # Apply wrappers to functions to affix static settings
    #  Update wrapper changes the __name__ field, which is used by the Method Server
    #  TODO (wardlt): Have users set the method name explicitly
    my_generate_molecules = partial(generate_molecules, episodes=args.rl_episodes)
    my_generate_molecules = update_wrapper(my_generate_molecules, generate_molecules)

    my_compute_atomization = partial(compute_atomization_energy, compute_hessian=args.qc_spec != "xtb",
                                     qc_config=qc_spec, reference_energies=ref_energies,
                                     compute_config=compute_config, code=code)
    my_compute_atomization = update_wrapper(my_compute_atomization, compute_atomization_energy)

    my_evaluate_mpnn = partial(evaluate_mpnn, atom_types=atom_types, bond_types=bond_types, batch_size=512)
    my_evaluate_mpnn = update_wrapper(my_evaluate_mpnn, evaluate_mpnn)

    my_update_mpnn = partial(update_mpnn, atom_types=atom_types, bond_types=bond_types)
    my_update_mpnn = update_wrapper(my_update_mpnn, update_mpnn)

    # Create the method server and task generator
    ml_cfg = {'executors': ['ml']}
    dft_cfg = {'executors': ['qc']}
    doer = ParslMethodServer([(my_generate_molecules, ml_cfg), (my_evaluate_mpnn, ml_cfg),
                              (my_update_mpnn, ml_cfg), (my_compute_atomization, dft_cfg)],
                             server_queues, config)

    # Configure the "thinker" application
    thinker = Thinker(client_queues,
                      initial_database,
                      initial_search_space,
                      agent,
                      mpnns,
                      output_dir=out_dir,
                      n_parallel=qc_workers,
                      n_parallel_updating=args.parallel_updating,
                      n_molecules=args.search_size,
                      queue_length=args.queue_length,
                      random_frac=args.random_frac,
                      greedy_frac=args.greedy_frac)
    logging.info('Created the method server and task generator')

    try:
        # Launch the servers
        #  The method server is a Thread, so that it can access the Parsl DFK
        #  The task generator is a Thread, so that all debugging methods get cast to screen
        doer.start()
        thinker.start()
        logging.info('Launched the servers')

        # Wait for the task generator to complete
        thinker.join()
        logging.info('Task generator has completed')
    finally:
        client_queues.send_kill_signal()

    # Wait for the method server to complete
    doer.join()
