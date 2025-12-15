import socket
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common import get_model_structure, get_random_params, WORKER_BATCH_SIZE
from message import Message, MessageType
from mnist_model import SimpleMNISTNet
from network import recv_msg, send_msg

SERVER_IP = "127.0.0.1"
PORT = 5000


def make_shard_loader(worker_id: int, num_workers: int) -> DataLoader:
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

    shard = Subset(ds, list(range(worker_id, len(ds), num_workers)))
    loader = DataLoader(shard, batch_size=WORKER_BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=False)

    return loader


class Worker:
    def __init__(self, server_ip, server_port, data_loader):
        self.server_ip = server_ip
        self.server_port = server_port
        self.model = self.init_model()
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def init_model(self):
        model = SimpleMNISTNet()
        print("Loading randomised params")
        self.load_params(model, get_random_params(get_model_structure(model)))
        return model

    @staticmethod
    def extract_grads(model: torch.nn.Module):
        grads = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                g = torch.zeros_like(p, device="cpu")
            else:
                g = p.grad.detach().cpu().clone()
            grads[name] = g
        return grads

    @staticmethod
    def load_params(model, params):
        with torch.no_grad():
            for k, p in model.named_parameters():
                p.copy_(params[k])

    def get_loss_grads(self, params):
        self.load_params(self.model, params)
        try:
            x, y = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            x, y = next(self.data_iter)
        assert torch.is_grad_enabled()
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y.long())
        loss.backward()

        print("loss", float(loss.detach().cpu()))
        return self.extract_grads(self.model)

    def start(self):
        while True:
            print("Attempting to connect to server")
            try:
                with socket.create_connection((self.server_ip, self.server_port)) as sock:
                    send_msg(sock, Message(MessageType.HELLO, get_model_structure(self.model)))
                    try:
                        while reply := recv_msg(sock):
                            if reply.message_type == MessageType.STOP:
                                break
                            elif reply.message_type == MessageType.PARAMS:
                                send_msg(sock, Message(MessageType.GRADS, self.get_loss_grads(reply.data)))
                            else:
                                assert False, "should not be here"
                    except KeyboardInterrupt:
                        pass
                    break
            except ConnectionError:
                pass


if __name__ == '__main__':
    num_workers = int(sys.argv[1])
    worker_id = int(sys.argv[2])

    worker = Worker(SERVER_IP, PORT, make_shard_loader(worker_id, num_workers))
    worker.start()
