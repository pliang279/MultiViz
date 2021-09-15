"""
Class to handle Slow Influence Function on a PyTorch model.
Adapted from:
- https://github.com/allenai/allennlp/tree/main/allennlp/interpret/influence_interpreters
- https://github.com/xhan77/influence-function-analysis
"""

import re
from typing import List, Union

import numpy as np
import torch
from torch import autograd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ...utils.misc import set_seed


class SimpleInfluence:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        influence_on_decision: bool = True,
        params_to_freeze: List[str] = None,
        use_cuda: bool = False,
        lissa_batch_size: int = 8,
        damping: float = 3e-3,
        num_samples: int = 1,
        lissa_depth: Union[int, float] = 0.25,
        scale: float = 1e4,
        seed: int = 1234,
        last_state_key: str = "hidden_states",
    ):

        self.model = model
        self.train_dataset = train_dataset
        self.params_to_freeze = params_to_freeze
        self.influence_on_decision = influence_on_decision

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=False, num_workers=1
        )

        self.model.to(self.device)

        if params_to_freeze is not None:
            for name, param in self.model.named_parameters():
                if any([re.match(pattern, name) for pattern in params_to_freeze]):
                    param.requires_grad = False

        set_seed(seed)
        # LiSSA specific code
        self.lissa_loader = DataLoader(
            train_dataset, batch_size=lissa_batch_size, shuffle=True
        )

        if isinstance(lissa_depth, float) and lissa_depth > 0.0:
            self.lissa_recursion_depth = int(len(self.lissa_loader) * lissa_depth)
        elif isinstance(lissa_depth, int) and lissa_depth > 0:
            self.lissa_recursion_depth = lissa_depth
        else:
            raise ValueError(
                "'lissa_recursion_depth' should be a positive int or float"
            )

        self.damping = damping
        self.num_samples = num_samples
        # self.lissa_depth = lissa_depth
        self.scale = scale
        self.last_state_key = last_state_key

        self._train_instances = None
        self._used_param_names = None
        self._used_params = None

    @property
    def used_params(self):
        if self._used_params is None:
            self.compute_training_gradients()
        assert self._used_params is not None
        return self._used_params

    @property
    def used_param_names(self):
        if self._used_param_names is None:
            self.compute_training_gradients()
        assert self._used_param_names is not None
        return self._used_param_names

    @property
    def train_instances(self):
        """
        The training instances along with their corresponding loss and gradients.
        !!! Note
            Accessing this property requires calling `self._gather_train_instances_and_compute_gradients()`
            if it hasn't been called yet, which may take several minutes.
        """
        if self._train_instances is None:
            self.compute_training_gradients()
        assert self._train_instances is not None
        return self._train_instances

    def compute_training_gradients(self):
        self.model.train()
        self._train_instances = []

        for idx, batch in enumerate(
            tqdm(self.train_loader, desc="Computing Training Gradients")
        ):

            # Move input to device
            for key, value in batch.items():
                batch[key] = value.to(self.device)

            self.model.zero_grad()

            # Model should return the loss
            output = self.model(**batch)
            loss = output["loss"]

            # TODO: Check what is retain_graph for
            loss.backward(retain_graph=True)

            if self._used_params is None or self._used_param_names is None:
                self._used_params = []
                self._used_param_names = []
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self._used_params.append(param)
                        self._used_param_names.append(name)

            grads = autograd.grad(loss, self._used_params)

            self._train_instances.append(
                {
                    "idx": idx,
                    "batch": batch,
                    "loss": loss.detach().item(),
                    "grads": grads,
                }
            )

    def interpret_instances(self, test_instances, top_k=None):

        # self.compute_training_gradients()

        interpretations = []

        for test_idx, test_batch in enumerate(
            tqdm(test_instances, desc="Interpreting Test Instances")
        ):

            # Move input to device
            for key, value in test_batch.items():
                test_batch[key] = value.to(self.device)

            self.model.eval()
            self.model.zero_grad()

            if self.influence_on_decision:
                with torch.no_grad():
                    model_output = self.model(**test_batch)
                    assert self.last_state_key in model_output
                    logits = model_output[self.last_state_key]
                    logits = logits.detach().cpu().numpy()
                    outputs = np.argmax(logits, axis=1)

                    assert "labels" in test_batch
                    test_batch["labels"] = (
                        torch.from_numpy(outputs).long().to(self.device)
                    )

            test_output_dict = self.model(**test_batch)
            assert "loss" in test_output_dict
            test_loss = test_output_dict["loss"]
            test_loss_float = test_loss.detach().item()

            test_grads = autograd.grad(test_loss, self.used_params)

            influence_scores = torch.zeros(len(self.train_instances))
            for idx, score in enumerate(self._calculate_influence_scores(test_grads)):
                influence_scores[idx] = score

            if top_k is not None:
                top_k_scores, top_k_indices = torch.topk(influence_scores, top_k)

            interpretations.append(
                {
                    "idx": test_idx,
                    "batch": test_batch,
                    "loss": test_loss_float,
                    "influence_scores": influence_scores.numpy()
                    if top_k is None
                    else np.array(list(zip(top_k_indices, top_k_scores))),
                }
            )

        return interpretations

    def interpret(self, instance, top_k=None):

        return self.interpret_instances([instance], top_k)

    def get_inverse_hvp_lissa(
        self,
        vs,
    ):
        inverse_hvps = [torch.tensor(0) for _ in vs]
        for _ in tqdm(
            range(self.num_samples), total=self.num_samples, desc="Lissa Samples"
        ):

            cur_estimates = vs
            lissa_iterator = iter(self.lissa_loader)
            pbar = tqdm(
                range(self.lissa_recursion_depth),
                total=self.lissa_recursion_depth,
                desc="LiSSA Recursion Depth",
            )
            for j in pbar:
                try:
                    training_batch = next(lissa_iterator)
                except StopIteration:
                    lissa_iterator = iter(self.lissa_loader)
                    training_batch = next(lissa_iterator)

                # Move input to device
                for key, value in training_batch.items():
                    training_batch[key] = value.to(self.device)

                # NOTE: We are using train mode in the `interpret_instances` method
                # so no need to  set here
                self.model.zero_grad()
                train_output_dict = self.model(**training_batch)

                assert "loss" in train_output_dict

                hvps = self.get_hvp(
                    train_output_dict["loss"], self.used_params, cur_estimates
                )

                cur_estimates = [
                    v + (1 - self.damping) * cur_estimate - hvp / self.scale
                    for v, cur_estimate, hvp in zip(vs, cur_estimates, hvps)
                ]

                # Update Loss
                if (j % 50 == 0) or (j == len(self.lissa_loader) - 1):
                    norm = np.linalg.norm(
                        self._flatten_tensors(cur_estimates).cpu().numpy()
                    )
                    pbar.set_description(
                        desc=f"Calculating inverse HVP, norm = {norm:.5f}"
                    )

            inverse_hvps = [
                inverse_hvp + cur_estimate / self.scale
                for inverse_hvp, cur_estimate in zip(inverse_hvps, cur_estimates)
            ]

        return_ihvp = self._flatten_tensors(inverse_hvps)
        return_ihvp /= self.num_samples
        return return_ihvp

    def get_hvp(self, loss, params, vectors):
        assert len(params) == len(vectors)
        assert all(p.size() == v.size() for p, v in zip(params, vectors))
        grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
        hvp = autograd.grad(grads, params, grad_outputs=vectors)  # Caclulates HVP
        return hvp

    @staticmethod
    def _flatten_tensors(tensors):
        views = []
        for p in tensors:
            if p.data.is_sparse:
                view = p.data.to_dense().contiguous().view(-1)
            else:
                view = p.data.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _calculate_influence_scores(self, test_grads):

        inv_hvp = self.get_inverse_hvp_lissa(
            test_grads,
        )

        return [
            torch.dot(inv_hvp, self._flatten_tensors(x["grads"])).item()
            for x in tqdm(self.train_instances, desc="Influence Score Calculation")
        ]
