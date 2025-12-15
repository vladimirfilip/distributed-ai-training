from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class WorkerModel:
    def __init__(self, model: nn.Module, loss_function: Callable[..., torch.Tensor], data_loader: DataLoader):
        self.model = model
        self.loss_function = loss_function
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

    def write_params(self, params: dict[str, Tensor]) -> None:
        """
        Overwrite the `self.model`'s parameters with `params`
        :param params: the params to write
        :return: None
        """
        with torch.no_grad():
            for k, p in self.model.named_parameters():
                p.copy_(params[k])

    @staticmethod
    def extract_grads(model: torch.nn.Module) -> dict[str, Tensor]:
        """
        Extracts the gradient tensors from `model` (zeroed tensor if `None`).
        :param model: model to extract gradients from
        :return: dictionary mapping parameter name to gradient tensor
        """
        grads = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                g = torch.zeros_like(p, device="cpu")
            else:
                g = p.grad.detach().cpu().clone()
            grads[name] = g
        return grads

    def get_model_structure(self) -> dict[str, (tuple, str)]:
        """
        :return: dictionary mapping parameter name to parameter shape and data type
        """
        return {name: (tuple(p.shape), str(p.dtype)) for name, p in self.model.named_parameters()}

    def get_loss_grads(self, params: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Overwrites model params with params given, loads a bach of data,
        calculates loss, back-propagates, and returns gradients.
        :param params: initial parameters intended
        :return: dictionary mapping parameter name to gradient tensor
        """
        self.write_params(params)
        try:
            x, y = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            x, y = next(self.data_iter)
        assert torch.is_grad_enabled()
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        loss = self.loss_function(logits, y.long())
        loss.backward()

        return self.extract_grads(self.model)