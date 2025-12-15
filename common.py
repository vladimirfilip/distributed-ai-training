from threading import Semaphore

import torch.nn
from torch import Tensor

MODEL_CLASS = None
PARAMS_OUTPUT_PATH = "./model_params.pt"

WORKER_BATCH_SIZE = 512


def sema_down(sema: Semaphore) -> None:
    sema.acquire()


def sema_up(sema: Semaphore) -> None:
    sema.release()


def get_random_params(structure: dict[str, (tuple, str)]) -> dict[str, Tensor]:
    DTYPE_MAP: dict[str, torch.dtype] = {"torch.float16": torch.float16, "torch.float32": torch.float32,
        "torch.bfloat16": torch.bfloat16, "torch.float64": torch.float64, "torch.int32": torch.int32,
        "torch.int64": torch.int64, }
    return {name: torch.randn(meta[0], dtype=DTYPE_MAP.get(meta[1]), device="cpu") for name, meta in structure.items()}
