"""Implementation of the Emap.

Handles implementation of the EMAP (Empirically Multimodally-Additive Function Projection)
for the analysis of how cross-modal a model is. Higher (positive) gap in EMAP scores compared to
the original model scores indicate a more cross-modal model.
"""

import collections
from collections import OrderedDict
from itertools import product
from typing import Callable

import numpy as np


class Emap:
    def __init__(
        self,
        predictor_fn: Callable,
        dataset: "collections.OrderedDict[str, np.ndarray]",
    ):
        """Initializes the EMap class.

        Args:
            predictor_fn (Callable): A Callable that takes in a dictionary of input np.ndarrays and returns
                an np.ndarray containing predictions. The predictor_fn should take arguments from the dataset provided.
            dataset (collections.OrderedDict[str, np.ndarray): A dictionary of numpy array of the dataset to
                be used for the analysis. Each value is an np.ndarray over different mode inputs as needed by the predictor.
        """
        self.predictor_fn = predictor_fn
        self.dataset = dataset

    def get_cartesian_product_of_indices(self):
        """Computes the cartesian product of the indices of the dataset.

        Returns:
            np.ndarray: The cartesian product of index ranges of the dataset.
        """

        index_ranges = list(map(lambda x: list(range(len(x))), self.dataset.values()))
        cartesian_product = np.array(list(product(*index_ranges)))
        return cartesian_product

    def combination_generator(self, batch_size: int = 32):
        """Creates a generator to generate inputs with all combinations of different indices.

        Args:
            batch_size[int]: The batch size to use for the generator.
        Returns:
            collections.Ordered[str, List]: A batch of the new dataset which each batch a list of the original input.
        """

        cartesian_product = self.get_cartesian_product_of_indices()
        keys = list(self.dataset.keys())

        for i in range(0, cartesian_product.shape[0], batch_size):
            output_dict = OrderedDict(zip(keys, [[] for i in range(len(keys))]))
            batch_index_products = cartesian_product[i : i + batch_size]
            for index_product in batch_index_products:
                for j, key in enumerate(keys):
                    output_dict[key].append(self.dataset[key][index_product[j]])

            for key in keys:
                output_dict[key] = np.array(output_dict[key])
            yield output_dict

    def dataset_generator(self, batch_size: int = 32):
        """Computes the predictions on the given dataset for a given predictor.

        Returns:
            np.ndarray: A numpy array of predictions for a given predictor.
        """

        keys = list(self.dataset.keys())
        for i in range(0, len(self.dataset[keys[0]]), batch_size):
            yield OrderedDict(
                {key: self.dataset[key][i : i + batch_size] for key in keys}
            )

    def compute_predictions(self, typ: str = "emap", batch_size: int = 32):
        """Computes the predictions on the dataset of given type for a given predictor.

        Returns:
            np.ndarray: A numpy array of predictions for a given predictor.

        """

        assert typ in [
            "emap",
            "orig",
        ], "`typ` should be either `emap` or `orig` in prediction computation."

        output_predictions = []
        if typ == "emap":
            dataset = self.combination_generator(batch_size=batch_size)
        else:
            dataset = self.dataset_generator(batch_size=batch_size)
        for batch in dataset:
            output_predictions.append(self.predictor_fn(**batch))
        return np.vstack(output_predictions)

    def compute_emap_scores(self, batch_size: int = 32):
        """Computes the EMAP scores for the predictor and the dataset.

        Returns:
            np.ndarray: A numpy array of EMAP scores for the given predictor.
        """

        emap_predictions = self.compute_predictions("emap", batch_size=batch_size)

        logits_mean = np.mean(emap_predictions, axis=0)

        input_lengths = list(map(lambda x: len(x), self.dataset.values()))
        emap_predictions = emap_predictions.reshape(*input_lengths, -1)

        all_axes = list(range(emap_predictions.ndim - 1))
        projected_predictions = np.mean(
            emap_predictions, axis=tuple(all_axes[:0] + all_axes[1:])
        )

        for axis in all_axes[1:]:
            projected_predictions += np.mean(
                emap_predictions, axis=tuple(all_axes[:axis] + all_axes[axis + 1 :])
            )

        projected_predictions -= logits_mean

        return projected_predictions
