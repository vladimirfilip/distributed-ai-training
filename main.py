from threading import Thread

from mnist_model import WorkerMNistModel
from server import Server
from worker import Worker

#
# Disclaimer: this is merely an example script that creates the parameter server and workers in one place
# and is not performant when increasing the number of workers - possibly because of threading overhead.
# If you create the server and workers in separate processes, you will have a significant increase in
# throughput with increasing number of workers.
#

NUM_WORKERS = 1
TRAINING_STEPS = 64
LEARNING_RATE = 0.02

HOST = "0.0.0.0"
SERVER_IP = "127.0.0.1"
SERVER_PORT = 5000

workers = [Worker(SERVER_IP, SERVER_PORT, WorkerMNistModel(i, NUM_WORKERS)) for i in range(NUM_WORKERS)]
worker_threads = [Thread(target=(lambda w : w.start()), args=[worker], daemon=True) for worker in workers]
server_thread = Thread(target=lambda : Server(HOST, SERVER_PORT, NUM_WORKERS, TRAINING_STEPS, LEARNING_RATE).start())
for thread in worker_threads:
    thread.start()
server_thread.start()