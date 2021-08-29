"""Implementation of the Emap.

Handles implementation of the EMAP (Empirically Multimodally-Additive Function Projection)
for the analysis of how cross-modal a model is. Higher (positive) gap in EMAP scores compared to
the original model scores indicate a more cross-modal model.
"""

from itertools import product
from typing import Callable, Union, Dict, Tuple

import numpy as np


class Emap:
    def __init__(
        self,
        predictor_fn: Callable,
        dataset: Dict[str, Union[Dict[str, np.ndarray], Tuple[np.ndarray], np.ndarray]],
    ):
        """Initializes the EMap class.

        Args:
            predictor_fn (Callable): A Callable that takes in a dictionary of input np.ndarrays and returns
                an np.ndarray containing predictions. The predictor_fn should take arguments from the dataset provided.
            dataset (Dict[str, Union[Dict[str, np.ndarray], Tuple[np.ndarray] ,np.ndarray]]): A dictionary of numpy array, or tuples,
                or dictionaries of the dataset to be used for the analysis. Each value is an np.ndarray over different mode
                inputs as needed by the predictor.
        """
        self.predictor_fn = predictor_fn
        self.dataset = dataset

    def get_input_lengths(self):
        """Returns the input lengths for each mode.

        Returns:
            List[int]: A list of the lengths for each mode.
        """
        # First mode, first value:
        test_value = list(self.dataset.values())[0]
        # If it is a batched numpy array, length is same as batch size
        if isinstance(test_value, np.ndarray):
            input_lengths = list(map(lambda x: len(x), self.dataset.values()))
        elif isinstance(test_value, Tuple):
            # Take the length of the first input in that mode
            input_lengths = list(map(lambda x: len(x[0]), self.dataset.values()))
        elif isinstance(test_value, Dict):
            # Take the length of the first key in the dictionary for that mode
            input_lengths = list(
                map(
                    lambda x: len(x[list(x.keys())[0]]),
                    self.dataset.values(),
                )
            )
        else:
            raise NotImplementedError(
                "No implementation found for types other than numpy array, tuple and dict."
            )
        return input_lengths

    def get_cartesian_product_of_indices(self):
        """Computes the cartesian product of the indices of the dataset.

        Returns:
            np.ndarray: The cartesian product of index ranges of the dataset.
        """

        input_lengths = self.get_input_lengths()
        index_ranges = list(map(lambda x: list(range(x)), input_lengths))
        cartesian_product = np.array(list(product(*index_ranges)))
        return cartesian_product

    def combination_generator(self, batch_size: int = 32):
        """Creates a generator to generate inputs with all combinations of different indices.

        Args:
            batch_size (int): The batch size to use for the generator.
        Returns:
            Dict[str, Union[Dict[str, np.ndarray], Tuple[np.ndarray] ,np.ndarray]]:
                A batch of the new dataset which each batch a list of the original input.
        """

        cartesian_product = self.get_cartesian_product_of_indices()
        keys = list(self.dataset.keys())

        for i in range(0, cartesian_product.shape[0], batch_size):
            if isinstance(self.dataset[keys[0]], np.ndarray):
                output_dict = dict(zip(keys, [[] for i in range(len(keys))]))
            elif isinstance(self.dataset[keys[0]], Tuple):
                # num modes = len(keys)
                # num inputs per mode = len(self.dataset[key])
                # need one more list for batch size
                output_dict = dict(
                    zip(
                        keys,
                        [
                            [[] for k in range(len(self.dataset[key]))]
                            for j, key in enumerate(keys)
                        ],
                    )
                )
            # We already check for types in cartesian product method
            else:
                output_dict = dict(
                    zip(
                        keys,
                        [
                            {k: [] for k in self.dataset[key].keys()}
                            for j, key in enumerate(keys)
                        ],
                    )
                )

            batch_index_products = cartesian_product[i : i + batch_size]
            for index_product in batch_index_products:
                for j, mode_key in enumerate(keys):
                    if isinstance(self.dataset[mode_key], np.ndarray):
                        output_dict[mode_key].append(
                            self.dataset[mode_key][index_product[j]]
                        )
                    elif isinstance(self.dataset[mode_key], Tuple):
                        # output_dict[key] will be list of tuples of numpy arrays
                        # Each numpy array will correspond to one kind of input for the `key` mode.
                        # We need to convert it to a tuple of numpy arrays with the same length.
                        # The numpy arrays in each have to be stacked togther.
                        for tuple_index in range(len(self.dataset[mode_key])):
                            output_dict[mode_key][tuple_index].append(
                                self.dataset[mode_key][tuple_index][index_product[j]]
                            )
                    else:
                        for dict_key in self.dataset[mode_key].keys():
                            output_dict[mode_key][dict_key].append(
                                self.dataset[mode_key][dict_key][index_product[j]]
                            )

            # Convert internal arrays to numpy arrays
            for mode_key in keys:
                if isinstance(self.dataset[mode_key], np.ndarray):
                    output_dict[mode_key] = np.array(output_dict[mode_key])
                elif isinstance(self.dataset[mode_key], Tuple):
                    for input_index in range(len(output_dict[mode_key])):
                        output_dict[mode_key][input_index] = np.array(
                            output_dict[mode_key][input_index]
                        )
                else:
                    for dict_key in self.dataset[mode_key].keys():
                        output_dict[mode_key][dict_key] = np.array(
                            output_dict[mode_key][dict_key]
                        )
            yield output_dict

    def dataset_generator(self, batch_size: int = 32):
        """Computes the predictions on the given dataset for a given predictor.

        Returns:
            np.ndarray: A numpy array of predictions for a given predictor.
        """

        keys = list(self.dataset.keys())
        first_value = list(self.dataset.values())[0]
        if isinstance(first_value, np.ndarray):
            for i in range(0, len(first_value), batch_size):
                yield {key: self.dataset[key][i : i + batch_size] for key in keys}
        elif isinstance(first_value, Tuple):
            for i in range(0, len(first_value[0]), batch_size):
                yield {
                    key: tuple(
                        [
                            input_array[i : i + batch_size]
                            for input_array in self.dataset[key]
                        ]
                    )
                    for key in keys
                }
        elif isinstance(first_value, Dict):
            first_value_value = list(first_value.values())[0]
            for i in range(0, len(first_value_value), batch_size):
                yield {
                    key: {
                        dict_key: self.dataset[key][dict_key][i : i + batch_size]
                        for dict_key in self.dataset[key].keys()
                    }
                    for key in keys
                }

    def compute_predictions(self, typ: str = "emap", batch_size: int = 32):
        """Computes the predictions on the dataset of given type for a given predictor.

        Args:
            typ (str): The type of predictions to compute. One of "emap" or "orig".
            batch_size (int): The batch size to use for the generator.
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

        input_lengths = self.get_input_lengths()
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
