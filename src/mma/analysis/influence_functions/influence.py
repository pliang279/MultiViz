"""
Class to handle Slow Influence Function on a PyTorch model.
Adapted from:
- https://github.com/allenai/allennlp/tree/main/allennlp/interpret/influence_interpreters
- https://github.com/xhan77/influence-function-analysis
"""

import torch
from typing import List, Union
from torch import autograd
from torch.data.utils import Dataset, DataLoader
import re
from tqdm.auto import tqdm
import numpy as np


class Influence:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.data.utils.Dataset,
        test_dataset: torch.data.utils.Dataset,
        params_to_freeze: List[str] = None,
        use_cuda: bool = False,
        lissa_batch_size: int = 8,
        damping: float = 3e-3,
        lissa_repeat: int = 0.25,
        lissa_recursion_depth: Union[int, float] = 1.0,
        scale: float = 1e4,
    ):

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.params_to_freeze = params_to_freeze

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=False, num_workers=1
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1
        )

        self.model.to(self.device)

        if params_to_freeze is not None:
            for name, param in self.model.named_parameters():
                if any([re.match(pattern, name) for pattern in params_to_freeze]):
                    param.requires_grad = False

        # LiSSA specific code
        self.lissa_loader = DataLoader(
            train_dataset, batch_size=lissa_batch_size, shuffle=True
        )

        # TODO: Check how we can incorporate this into the ihvp
        # if isinstance(lissa_recursion_depth, float) and lissa_recursion_depth > 0.0:
        #     self.lissa_dataloader.batches_per_epoch = int(
        #         len(self.lissa_dataloader) * lissa_recursion_depth
        #     )
        # elif isinstance(lissa_recursion_depth, int) and lissa_recursion_depth > 0:
        #     self.lissa_dataloader.batches_per_epoch = lissa_recursion_depth
        # else:
        #     raise ValueError("'lissa_recursion_depth' should be a positive int or float")

        self.damping = damping
        self.lissa_repeat = lissa_repeat
        self.lissa_recursion_depth = lissa_recursion_depth
        self.scale = scale

    def compute_training_gradients(self):
        self.model.train()
        training_instances = []

        for idx, batch in enumerate(tqdm(self.train_loader)):

            # Move input to device
            for key, value in batch.items():
                batch[key] = value.to(self.device)

            self.model.zero_grad()

            # Model should return the loss
            output = self.model(**batch)
            loss = output["loss"]

            # Get the parameters
            # TODO: Check if we need to have used_params and used_param_names

            used_params = []

            # TODO: Check what is retain_graph for
            loss.backward(retain_graph=True)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    used_params.append(param)

            grads = autograd.grad(loss, used_params)

            training_instances.append(
                {
                    "idx": idx,
                    "batch": batch,
                    "loss": loss.detach().item(),
                    "grads": grads,
                }
            )

            return training_instances, used_params

    def interpret_instances(self, test_instances, top_k=None):

        training_instances, used_params = self.compute_training_gradients()

        outputs = []

        for test_idx, test_batch in enumerate(tqdm(test_instances)):

            # Move input to device
            for key, value in test_batch.items():
                test_batch[key] = value.to(self.device)

            self.model.eval()
            self.model.zero_grad()

            test_output_dict = self.model(**test_batch)
            test_loss = test_output_dict["loss"]
            test_loss_float = test_loss.detach().item()

            test_grads = autograd.grad(test_loss, used_params)

            influence_scores = torch.zeros(len(training_instances))
            for idx, score in enumerate(
                self._calculate_influence_scores(test_grads, used_params)
            ):
                influence_scores[idx] = score

            if top_k is not None:
                top_k_scores, top_k_indices = torch.topk(influence_scores, top_k)
            # Sort the influence scores

            outputs.append(
                {
                    "idx": test_idx,
                    "batch": test_batch,
                    "loss": test_loss_float,
                    "influence_scores": influence_scores.numpy()
                    if top_k is None
                    else np.array(list(zip(top_k_indices, top_k_scores))),
                }
            )

        return outputs

    def interpret(self, instance, top_k=None):

        return self.interpret_instances([instance], top_k)

    def get_inverse_hvp_lissa(
        self,
        vs,
        used_params,
    ):
        inverse_hvps = [torch.tensor(0) for _ in vs]
        for _ in tqdm(range(self.lissa_repeat), total=self.lissa_repeat):

            cur_estimates = vs

            pbar = tqdm(self.lissa_data_loader, total=len(self.lissa_data_loader))

            for j, training_batch in enumerate(pbar):
                # NOTE: We are using train mode in the `interpret_instances` method
                # so no need to  set here
                self.model.zero_grad()
                train_output_dict = self.model(**training_batch)

                hvps = self.get_hvp(
                    train_output_dict["loss"], used_params, cur_estimates
                )

                cur_estimates = [
                    v + (1 - self.damping) * cur_estimate - hvp / self.scale
                    for v, cur_estimate, hvp in zip(vs, cur_estimates, hvps)
                ]

                # Update Loss
                if (j % 50 == 0) or (j == len(self.lissa_data_loader) - 1):
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
        return_ihvp /= self.lissa_repeat
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
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _calculate_influence_scores(self, test_grads, used_params):

        inv_hvp = self.get_inverse_hvp_lissa(
            test_grads,
            used_params,
        )

        return [
            torch.dot(inv_hvp, self._flatten_tensors(x.grads)).item()
            for x in tqdm(self.train_loader)
        ]
