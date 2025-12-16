import socket
import sys
from threading import Thread, Semaphore, Lock
from time import time
from typing import Optional

import torch

from common import sema_up, sema_down, get_random_params, WORKER_BATCH_SIZE, PARAMS_OUTPUT_PATH, TRAINING_STEPS
from message import Message, MessageType
from network import recv_msg, send_msg

HOST = "0.0.0.0"
PORT = 5000


class Server:
    def __init__(self, host: str, port: int, num_workers: int, training_steps: int, learning_rate: float = 0.01, throughput_logger_path: Optional[str] = None):
        """
        Initialises the gradient descent constants, data stores necessary for keeping track of active workers,
        and synchronisation primitives to ensure that all workers are working on the same training step
        :param host: host IP
        :param port: host port number
        :param num_workers: number of workers to which server will broadcast parameters
        :param training_steps: number of iterations of broadcasting parameters and receiving gradients
        :param learning_rate: learning rate applied to the averaged gradients received
        """
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.worker_sockets = []
        self.worker_sockets_lock = Lock()
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        self.workers_connected_sema = Semaphore(0)
        self.gradients_received_sema = Semaphore(0)
        self.gradients_from_workers = []
        self.gradients_from_workers_lock = Lock()
        self.model_structure = None
        self.model_structure_lock = Lock()
        self.params = {}
        self.running = False
        self.throughput_logger_path = throughput_logger_path

    def handle_worker_message(self, conn: socket.socket, msg: Message) -> None:
        """
        Callback for when a message from a worker is received
        :param conn: socket connection to the worker
        :param msg: message just received
        :return: None
        """
        #
        # If msg is a HELLO, push the socket to the list of worker sockets and assert
        # that the worker's declared model structure is the same as that already seen
        # and sema up to show that a connection to a worker has been newly formed
        #
        if msg.message_type == MessageType.HELLO:
            with self.worker_sockets_lock:
                assert len(self.worker_sockets) < self.num_workers, "Too many workers connecting"
                self.worker_sockets.append(conn)
            with self.model_structure_lock:
                if self.model_structure is None:
                    self.model_structure = msg.data
                else:
                    assert msg.data == self.model_structure, "Worker model structure does not match first one given"
            sema_up(self.workers_connected_sema)
        #
        # If msg is a GRADS, the worker has sent its gradients,
        # so atomically push them to the gradients list and sema up
        # to show gradients have been received
        #
        elif msg.message_type == MessageType.GRADS:
            grads = msg.data
            with self.gradients_from_workers_lock:
                self.gradients_from_workers.append(grads)
            sema_up(self.gradients_received_sema)

    def handle_worker_connection(self, conn: socket.socket, addr) -> None:
        """
        Callback for when a new connection is accepted. Remove from socket list and down
        workers_connected_sema if connection errors.
        :param conn: new socket connection
        :param addr: ip address of worker
        :return: None
        """
        with conn:
            print("Connected by", addr)
            try:
                while True:
                    self.handle_worker_message(conn, recv_msg(conn))
            except ConnectionError:
                print("Client disconnected:", addr)
                with self.gradients_from_workers_lock:
                    self.worker_sockets.remove(conn)
                sema_down(self.workers_connected_sema)

    def sgd_update(self):
        """
        Takes average of gradients received from workers and applies gradient descent.
        :return: None
        """
        average_grads = {key: sum(d[key] for d in self.gradients_from_workers) / self.num_workers for key in
                         self.gradients_from_workers[0].keys()}
        self.gradients_from_workers.clear()
        self.params = {k: p - self.learning_rate * average_grads[k] for k, p in self.params.items()}

    def run_training(self):
        """
        Training loop. Wait for `num_workers` workers to connect, then iterate: send parameters to all workers,
        wait for `num_workers` workers to send gradients, update parameters, repeat.
        Parameters are initially randomised.
        :return:
        """
        self.running = True
        for i in range(self.num_workers):
            print(f"Waiting for {i + 1}-th worker")
            sema_down(self.workers_connected_sema)

        print("All workers connected")

        self.params = get_random_params(self.model_structure)

        start_time = time()
        total_step_time = 0
        for i in range(self.training_steps):
            step_start_time = time()

            print(f"Starting training stage {i + 1}")
            with self.worker_sockets_lock:
                for conn in self.worker_sockets:
                    send_msg(conn, Message(MessageType.PARAMS, self.params))
            #
            # Will block until num_workers workers have sent gradients
            #
            for _ in range(self.num_workers):
                sema_down(self.gradients_received_sema)

            with self.gradients_from_workers_lock:
                self.sgd_update()

            total_step_time += time() - step_start_time

        print(f"Finished training in {round(time() - start_time, 6)} s")

        average_step_time: float = total_step_time / self.training_steps
        throughput: float = self.num_workers * WORKER_BATCH_SIZE / average_step_time
        if self.throughput_logger_path is not None:
            with open(self.throughput_logger_path, "a+") as file:
                file.write(f"{WORKER_BATCH_SIZE},{self.num_workers},{throughput}\n")

        print("Throughput (~ data points tested / s) :", throughput)

        #
        # Send STOP messages to all connected workers (if connections are still open)
        # as training has finished
        #
        with self.worker_sockets_lock:
            open_conns = self.worker_sockets.copy()
            for conn in open_conns:
                try:
                    send_msg(conn, Message(MessageType.STOP, None))
                except OSError:
                    pass

        torch.save(self.params, PARAMS_OUTPUT_PATH)
        print("Finished!")
        self.running = False

    def start(self):
        """
        Entry point for parameter server to become active. Handles worker messages concurrently
        and calls `run_training()`
        :return:
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((HOST, PORT))
            server.listen()
            server.settimeout(5.0)
            print(f"Listening on {HOST}:{PORT}")
            self.running = True

            def accept_loop():
                #
                # Will accept any connection made to the server.
                # Socket shutdown or keyboard interrupts are handled by terminating the loop
                #
                threads = []
                try:
                    while self.running:
                        try:
                            print("Accepting connections")
                            conn, addr = server.accept()
                        except socket.timeout:
                            break
                        print("Handling connection: ", conn, addr)
                        thread = Thread(target=self.handle_worker_connection, args=(conn, addr), daemon=True)
                        threads.append(thread)
                        thread.start()
                except KeyboardInterrupt:
                    pass
                except OSError:
                    pass
                for thread in threads:
                    thread.join(timeout=1.0)

            server_listen_thread = Thread(target=accept_loop, daemon=True)
            server_listen_thread.start()
            self.run_training()
            server_listen_thread.join()


if __name__ == "__main__":
    server = Server(HOST, PORT, int(sys.argv[1]), TRAINING_STEPS, throughput_logger_path="throughput_log.csv")
    server.start()
