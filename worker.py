import socket
import sys

from common import WORKER_BATCH_SIZE
from message import Message, MessageType
from mnist_model import WorkerMNistModel
from network import recv_msg, send_msg
from worker_model import WorkerModel

SERVER_IP = "127.0.0.1"
PORT = 5000

class Worker:
    def __init__(self, server_ip, server_port, worker_model: WorkerModel):
        self.server_ip = server_ip
        self.server_port = server_port
        self.worker_model = worker_model

    def handle_server_message(self, sock: socket, msg: Message) -> None:
        """
        Callback for when a message is received from the server. Should only be of type `MessageType.PARAMS`
        :param sock: socket connection with the parameter server
        :param msg: message received
        :return: None
        """
        #
        # If params are sent from the server, they will be run through the internal `WorkerModel`
        # to get the gradients
        #
        if msg.message_type == MessageType.PARAMS:
            send_msg(sock, Message(MessageType.GRADS, self.worker_model.get_loss_grads(msg.data)))
        else:
            assert False, f"Expected {MessageType.PARAMS} type message, got {msg.message_type}"

    def start(self) -> None:
        """
        Worker will forever try to open a connection to the parameter server and
        will continuously listen for messages if it succeeds
        :return: None
        """
        while True:
            print("Attempting to connect to server")
            try:
                with socket.create_connection((self.server_ip, self.server_port)) as sock:
                    #
                    # Sends HELLO message to register itself as a worker, sending its model
                    # structure for validation
                    #
                    send_msg(sock, Message(MessageType.HELLO, self.worker_model.get_model_structure()))
                    try:
                        while reply := recv_msg(sock):
                            if reply.message_type == MessageType.STOP:
                                break
                            self.handle_server_message(sock, reply)
                    except KeyboardInterrupt:
                        pass
                    break
            except ConnectionError:
                pass


if __name__ == '__main__':
    num_workers = int(sys.argv[1])
    worker_id = int(sys.argv[2])

    worker = Worker(SERVER_IP, PORT, WorkerMNistModel(worker_id, num_workers, WORKER_BATCH_SIZE))
    worker.start()
