import csv
import importlib
import json
import logging
import math
import os
import random
import shutil
import threading
from collections import deque
from datetime import datetime
from random import Random
from time import perf_counter

import numpy as np
import torch
from matplotlib import pyplot as plt

from asyncNodes.node.Node import Node
from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping


class Swift(Node):
    """
    This class defines the node on overlay graph

    """

    def get_neighbors(self, node=None):
        return set(self.rng.sample(list(self.my_neighbors), self.degree))

    def receive_and_queue(self):
        """
        Function that runs in parallel with training (main thread).
        Receives models from neighbors and appends them to a queue.

        If receives multiple models from the same neighbor, only the latest one is stored
        """

        while True:

            if self.stopper.is_set():  # main thread finished training
                logging.debug("receiver-thread: out of loop")
                break

            try:
                x = self.communication.receive(block=False)
                if x is None:
                    continue

                sender, data = x

                if data["CHANNEL"] == "model":

                    self.receiver_pauser.wait()  # pause if aggregation is ongoing

                    if not sender in self.in_queue:
                        self.in_queue[sender] = []

                    # Check if there exists a chunk from the same sender with the same indices in the queue
                    # If it exists, remove it and replace it with the new one

                    existing = False
                    start_index_new = data["start_index"]
                    sparsity_new = data["sparsity"]
                    random_generation_seed_new = data["random_generation_seed"]

                    for i, chunk in enumerate(self.in_queue[sender]):
                        start_index = chunk["start_index"]
                        sparsity = chunk["sparsity"]
                        random_generation_seed = chunk["random_generation_seed"]
                        if (
                            start_index == start_index_new
                            and sparsity == sparsity_new
                            and random_generation_seed == random_generation_seed_new
                        ):
                            self.in_queue[sender][i] = data
                            existing = True
                            break

                    if not existing:
                        self.in_queue[sender].append(data)

                    self.received += 1
                    logging.info(
                        "Received Model from {} of iteration {}".format(
                            sender, data["iteration"]
                        )
                    )

            except (TypeError, ValueError):  # in case of timeout
                continue

        logging.info("receiver-thread: finished")

    def queue_and_send(self):
        """
        Function that runs in parallel with training (main thread).
        Sends chunks of model to neighbors
        """

        while True:
            if self.stopper.is_set():
                logging.debug("sender-thread: out of loop")
                break

            if len(self.out_queue) == 0:
                self.sender_pauser.clear()  # Set pauser to false, meaning no sending
                while not self.sender_pauser.is_set() and not self.stopper.is_set():
                    self.sender_pauser.wait(
                        timeout=10.0
                    )  # If queue is empty, pause until new data is available
            else:
                # Get a neigh and a chunk of model to send
                neighbor, chunk = self.out_queue.pop(0)
                self.sender_pauser.wait()  # Pause if aggregation is ongoing
                self.communication.send(neighbor, chunk)
                self.msg_sent += 1
                logging.info(
                    "Sent Model to {}, {} messages still to send".format(
                        neighbor, len(self.out_queue)
                    )
                )

        logging.info("sender-thread: finished")

    def run(self):
        """
        Start the decentralized learning

        """
        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1
        self.rng = Random()
        self.rng.seed(self.dataset.random_seed + self.uid)

        self.stopper = threading.Event()
        self.sender_pauser = threading.Event()
        self.sender_pauser.set()  # Start sending
        self.receiver_pauser = threading.Event()
        self.receiver_pauser.set()  # Start receiving
        init_time = perf_counter()

        # Connect to neighbors
        self.connect_neighbors()
        logging.info("Connected to all neighbors")

        logging.info("Total number of neighbor: {}".format(len(self.my_neighbors)))

        logging.info(
            "Node should sent {} chunks to {} neighbors".format(
                int(1 / self.chunk_fraction), self.degree
            )
        )

        # create thread for receiving neighbors' models
        receiver_thread = threading.Thread(target=self.receive_and_queue)
        logging.debug("Receiver thread created")
        receiver_thread.start()

        # create thread for sending chunks of model to neighbors
        sender_thread = threading.Thread(target=self.queue_and_send)
        logging.debug("Sender thread created")
        sender_thread.start()

        # Main loop
        for iteration in range(self.iterations):

            # Stop training if timeout is reached
            if perf_counter() - init_time > self.timeout:
                logging.debug("Running time: %f", perf_counter() - init_time)
                logging.info("Timeout reached. Stopping training.")
                self.stopper.set()
                break

            # Local Phase : Train
            logging.info(
                "Starting training iteration: %d out of %d", iteration, self.iterations
            )
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1

            self.iteration = iteration
            start = perf_counter()
            self.trainer.train(self.dataset)
            end = perf_counter()
            self.cpu_time_train += end - start

            # Send Phase : replace queue with new data
            self.sender_pauser.clear()  # Pause sending
            logging.debug("Pausing sender and updating queue")

            self.out_queue_to_add = []

            for chunk in self.sharing.get_data_to_send(
                chunk_fraction=self.chunk_fraction, training_iteration=iteration
            ):
                for neighbor in self.get_neighbors():
                    self.out_queue_to_add.append((neighbor, chunk))

            # Shuffle the list of pairs (neighbor, chunk to send)
            self.rng.shuffle(self.out_queue_to_add)

            self.out_queue += self.out_queue_to_add

            # Aggregation Phase
            self.msg_aggr += self.received
            self.received_percentage = self.received / (
                self.degree * int(1 / self.chunk_fraction)
            )
            if self.received == 0:
                logging.info("No response received this round")
            else:
                self.receiver_pauser.clear()  # Pause receiving
                logging.info(
                    "Pausing receiver and starting aggregation with %d chunks",
                    self.received,
                )
                self.sharing.plain_avg_received_queue(self.in_queue)
                if len(self.in_queue) > 0:
                    logging.debug("Be careful, queue not empty")
                self.received = 0

            # Results storing
            # While results are computed, send and receiver are still paused
            # Indeed, test time is long so node could have the time to send everything and receive a lot
            # WARNING : The best option is to test at the end of the whole process, as it is done by default
            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            if iteration:
                with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                    "r",
                ) as inf:
                    results_dict = json.load(inf)
            else:
                results_dict = {
                    "test_acc": {},
                    "time_acc": {},
                    "not_sent_percentage": {},
                    "received_percentage": {},
                    "train_loss": {},
                    "test_loss": {},
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                    "cpu_time_train": {},
                    "comm_time": {},
                    "gradient_steps": {},
                    "received_this_round": {},
                }

            results_dict["received_percentage"][
                iteration + 1
            ] = self.received_percentage

            if rounds_to_train_evaluate == 0:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"][iteration + 1] = loss_after_sharing

            if self.dataset.__testing__ and rounds_to_test == 0:

                rounds_to_test = self.test_after
                results_dict["time_acc"][iteration + 1] = perf_counter() - init_time

                if self.eval_on_test_set:

                    logging.info("Evaluating on test set.")
                    ta, tl = self.dataset.test(
                        self.model, self.loss, self.trainer.device
                    )
                    results_dict["test_acc"][iteration + 1] = ta
                    results_dict["test_loss"][iteration + 1] = tl

                else:

                    logging.info("Saving model to test later.")
                    if not os.path.exists(os.path.join(self.log_dir, "models")):
                        os.makedirs(os.path.join(self.log_dir, "models"), exist_ok=True)

                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            self.log_dir,
                            "models/{}_model_{}_iter.pt".format(
                                self.uid, iteration + 1
                            ),
                        ),
                    )

                if global_epoch == 49:
                    change *= 2
                global_epoch += change

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)

            if self.model.shared_parameters_counter is not None:
                logging.info("Saving the shared parameter counts")
                with open(
                    os.path.join(
                        self.log_dir, "{}_shared_parameters.json".format(self.rank)
                    ),
                    "w",
                ) as of:
                    json.dump(self.model.shared_parameters_counter.numpy().tolist(), of)

            self.sender_pauser.set()  # Resume sending
            logging.debug("Resuming sender")
            self.receiver_pauser.set()  # Resume receiving
            logging.debug("Resuming receiver")

        self.stopper.set()

        if self.store_model:
            logging.info("Storing final weight")
            self.model.dump_weights(self.weights_store_dir, self.uid, iteration)

        if not self.eval_on_test_set:
            logging.info("Evaluating on test set the saved models")
            test_eval = {"test_acc": {}, "test_loss": {}}
            model_files = os.listdir(os.path.join(self.weights_store_dir, "models"))
            model_files = [f for f in model_files if f.endswith(".pt")]
            for file in model_files:
                logging.info("Now reading %s", file)
                self.model.load_state_dict(
                    torch.load(os.path.join(self.weights_store_dir, "models", file))
                )
                self.model.eval()
                ta, tl = self.dataset.test(self.model, self.loss, self.trainer.device)
                file_name = file.split("_")
                iteration = file_name[2]
                test_eval["test_acc"][iteration] = ta
                test_eval["test_loss"][iteration] = tl

            # Order the dictionaries by iteration
            test_eval["test_acc"] = {
                k: v
                for k, v in sorted(
                    test_eval["test_acc"].items(), key=lambda item: int(item[0])
                )
            }
            test_eval["test_loss"] = {
                k: v
                for k, v in sorted(
                    test_eval["test_loss"].items(), key=lambda item: int(item[0])
                )
            }

            with open(os.path.join(self.log_dir, "0_testset_results.json"), "w") as of:
                json.dump(test_eval, of)
            logging.info("Test results saved")

        logging.info(
            "Neighbors not disconnected because docker will shut down everything. Process complete!"
        )

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        graph,
        iterations,
        log_dir,
        weights_store_dir,
        test_after,
        train_evaluate_after,
        reset_optimizer,
    ):
        """
        Instantiate object field with arguments.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        """
        self.rank = rank
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
        self.iterations = iterations
        self.test_after = test_after
        self.train_evaluate_after = train_evaluate_after
        self.reset_optimizer = reset_optimizer
        self.sent_disconnections = False

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

    def init_comm(self, comm_configs):
        """
        Instantiate communication module from config.

        Parameters
        ----------
        comm_configs : dict
            Python dict containing communication config params

        """
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, self.graph.n_procs, **comm_params
        )

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        dataset_seed=97,
        chunk_fraction=1.0,
        *args
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, rank, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )
        self.init_dataset_model(config, dataset_seed)
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)
        self.message_queue = dict()
        self.message_queue["model"] = deque()
        self.message_queue["OK"] = 0

        asyncConfigs = config["ASYNC_PARAMS"]
        self.eval_on_train_set = asyncConfigs["eval_on_train_set"]
        self.eval_on_test_set = asyncConfigs["eval_on_test_set"]
        self.store_model = asyncConfigs["store_model"]
        if "chunk_fraction" in asyncConfigs:
            self.chunk_fraction = asyncConfigs["chunk_fraction"]
        else:
            self.chunk_fraction = chunk_fraction
        self.linger = asyncConfigs["linger"]

        self.init_sharing(config)
        self.in_queue = dict()
        self.out_queue = []
        self.received = 0
        self.connect_neighbors()

        self.iteration = 0

        self.sending_log = []

        self.msg_sent = 0
        self.msg_aggr = 0

        self.cpu_time_train = 0
        self.gradient_steps = 0

        self.stats = {
            "msg_sent": {},
            "msg_aggr": {},
            "cpu_time_train": {},
            "gradient_steps": {},
        }

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        dataset_seed=97,
        chunk_fraction=1.0,
        *args
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
                device = "cpu" or "mps" or "cuda"
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """

        print(args)

        total_threads = os.cpu_count()

        if config["NODE"]["threads_per_proc"]:
            self.threads_per_proc = config["NODE"]["threads_per_proc"]
        else:
            self.threads_per_proc = max(
                math.floor(total_threads / mapping.procs_per_machine), 1
            )
        torch.set_num_threads(self.threads_per_proc)
        torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_evaluate_after,
            reset_optimizer,
            dataset_seed,
            chunk_fraction,
            *args
        )

        nodeConfigs = config["NODE"]
        self.timeout = nodeConfigs["timeout"] if "timeout" in nodeConfigs else 1
        self.degree = (
            nodeConfigs["graph_degree"] if "graph_degree" in nodeConfigs else 2
        )

        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )
        self.run()
