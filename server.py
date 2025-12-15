import socket
from threading import Thread, Semaphore, Lock
from time import time

from message import Message, MessageType
from network import recv_msg, send_msg
from common import sema_up, sema_down, get_random_params, WORKER_BATCH_SIZE

HOST = "0.0.0.0"
PORT = 5000

class Server:
    def __init__(self, host: str, port: int, num_workers: int, training_steps: int, learning_rate: float = 0.01):
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

    def handle_worker_message(self, conn: socket.socket, msg: Message):
        print(f"Received {msg.message_type} from {conn.getpeername()}")
        if msg.message_type == MessageType.HELLO:
            with self.worker_sockets_lock:
                assert len(self.worker_sockets) < self.num_workers, "Too many workers connecting"
                with self.model_structure_lock:
                    if self.model_structure is None:
                        self.model_structure = msg.data
                    else:
                        assert msg.data == self.model_structure, "Worker model structure does not match first one given"
                self.worker_sockets.append(conn)
            sema_up(self.workers_connected_sema)
        elif msg.message_type == MessageType.GRADS:
            grads = msg.data
            with self.gradients_from_workers_lock:
                self.gradients_from_workers.append(grads)
            sema_up(self.gradients_received_sema)

    def handle_worker_connection(self, conn: socket.socket, addr):
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
        average_grads = {key : sum(d[key] for d in self.gradients_from_workers) / self.num_workers for key in self.gradients_from_workers[0].keys()}
        self.params = {k : p - self.learning_rate * average_grads[k] for k, p in self.params.items()}

    def run_training(self):
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
            for _ in range(self.num_workers):
                sema_down(self.gradients_received_sema)
            with self.gradients_from_workers_lock:
                self.sgd_update()
            total_step_time += time() - step_start_time

        print(f"Finished training in {round(time() - start_time, 6)} s")
        average_step_time : float = total_step_time / self.training_steps
        throughput : float = self.num_workers * WORKER_BATCH_SIZE / average_step_time
        print("Throughput (~ data points tested / s) :", throughput)

        with self.worker_sockets_lock:
            open_conns = self.worker_sockets.copy()
            for conn in open_conns:
                try:
                    send_msg(conn, Message(MessageType.STOP, None))
                except OSError:
                    pass

        print("Finished!")
        self.running = False

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((HOST, PORT))
            server.listen()
            server.settimeout(5.0)
            print(f"Listening on {HOST}:{PORT}")
            self.running = True
            def accept_loop():
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

            t = Thread(target=accept_loop, daemon=True)
            t.start()
            self.run_training()
            t.join()

if __name__ == "__main__":
    server = Server(HOST, PORT, 3, 64)
    server.start()