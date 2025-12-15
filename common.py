from threading import Semaphore

import torch.nn

HELLO_TYPE = "HELLO"
STOP_TYPE = "STOP"
GRAD_TYPE = "GRAD"
PARAM_TYPE = "PARAM"

MODEL_CLASS = None
MODEL_PATH = ""

WORKER_BATCH_SIZE = 512

def sema_down(sema: Semaphore):
    sema.acquire()

def sema_up(sema: Semaphore):
    sema.release()

def get_model_structure(model: torch.nn.Module):
    return {name: (tuple(p.shape), str(p.dtype)) for name, p in model.named_parameters()}

def get_random_params(structure):
    DTYPE_MAP = {
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.bfloat16": torch.bfloat16,
        "torch.float64": torch.float64,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
    }
    return {
        name : torch.randn(meta[0],
                           dtype=DTYPE_MAP.get(meta[1]), device="cpu")
        for name, meta in structure.items()
    }