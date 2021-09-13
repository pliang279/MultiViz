import unittest

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from mma.analysis.influence_functions.simple_influence import SimpleInfluence


def recursive_match(actual, expected):

    # Leaves
    if isinstance(actual, int):
        return actual == expected
    elif isinstance(actual, np.ndarray) and actual.dtype == np.float:
        return np.allclose(actual, expected, atol=1e-5)
    elif isinstance(actual, torch.Tensor) and actual.dtype == torch.float:
        return torch.allclose(actual, expected, atol=1e-5)
    elif isinstance(actual, torch.Tensor) and actual.dtype == torch.int:
        return actual == expected
    elif isinstance(actual, np.ndarray) and actual.dtype == np.int:
        return actual == expected

    # Sequences
    elif isinstance(actual, list):
        return all(
            recursive_match(actual[idx], expected[idx]) for idx in range(len(actual))
        )
    elif isinstance(actual, dict):
        keys_match = all(key in actual for key in expected.keys())
        values_match = all(
            recursive_match(actual[key], expected[key]) for key in actual.keys()
        )

        return keys_match and values_match


class TestSimpleInfluence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_input_1 = np.array(
            [
                [-0.38313667, -0.57625521, 0.64543348, -1.53118712, 0.00254321],
                [-1.7714859, 0.11143005, -0.02949927, 0.5507662, 1.38710581],
                [-0.94095595, -0.5021287, 1.15750038, 1.47322928, -1.26181978],
                [-0.18837869, 0.45918263, 1.79731464, 0.38474411, 0.13050071],
                [-0.86100136, -0.04482581, -0.06322346, 0.61815964, -1.38737267],
            ]
        )

        train_input_2 = np.array(
            [
                [0.44525277, 0.73597032, -0.72959245],
                [-1.79780742, -1.33908299, 1.05475906],
                [0.05604235, 0.98735651, 1.73243159],
                [0.69923864, -0.08010251, 1.70262097],
                [-0.26748404, -0.76923189, -0.14560712],
            ]
        )

        train_labels = np.array([2, 0, 2, 1, 2])

        test_input_1 = np.array(
            [
                [0.438961, -0.5618169, 0.02209749, 0.27288235, -2.02370362],
                [0.49510039, -0.13382932, 0.087042, -0.50757061, -0.60003024],
            ]
        )

        test_input_2 = np.array(
            [
                [-0.26086342, -1.72508324, -2.05845896],
                [0.99154996, 1.03019393, -1.26970934],
            ]
        )

        test_labels = np.array([0, 1])

        class DummyDataset(Dataset):
            def __init__(self, input_1, input_2, labels):
                self.input_1 = input_1
                self.input_2 = input_2
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {
                    "input_1": self.input_1[idx],
                    "input_2": self.input_2[idx],
                    "labels": self.labels[idx],
                }

        cls.train_dataset = DummyDataset(train_input_1, train_input_2, train_labels)
        cls.test_dataset = DummyDataset(test_input_1, test_input_2, test_labels)

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()

                self.dense_1 = nn.Linear(5, 3)
                self.dense_2 = nn.Linear(3, 3)

                self.init_weights()

            def forward(self, input_1, input_2, labels=None):
                x = self.dense_1(input_1)
                y = self.dense_2(input_2)
                z = x + y
                if labels is None:
                    return {"hidden_states": z}
                else:
                    loss_fn = nn.CrossEntropyLoss()
                    return {"hidden_states": z, "loss": loss_fn(z, labels)}

            def init_weights(self):
                weights_input_1 = np.array(
                    [
                        [-0.63405347, 1.20433563, -0.72497645],
                        [1.63702896, -0.94918872, -1.11057412],
                        [0.11731525, -0.52821813, -0.38398329],
                        [-0.00873864, 0.49997465, 0.62061593],
                        [0.93855677, 0.08329474, 0.93919091],
                    ]
                ).T
                bias_input_1 = np.array([-1.52406738, -0.76214741, 0.87666446])
                weights_input_2 = np.array(
                    [
                        [0.22966335, -0.23029136, 1.27847173],
                        [-1.92014287, -1.0555818, 0.37404504],
                        [0.0465951, 0.17966665, -0.26075708],
                    ]
                ).T
                bias_input_2 = np.array([0.50664227, 0.83077437, 0.21617283])

                with torch.no_grad():
                    self.dense_1.weight = nn.Parameter(
                        torch.from_numpy(weights_input_1)
                    )
                    self.dense_1.bias = nn.Parameter(torch.from_numpy(bias_input_1))
                    self.dense_2.weight = nn.Parameter(
                        torch.from_numpy(weights_input_2)
                    )
                    self.dense_2.bias = nn.Parameter(torch.from_numpy(bias_input_2))

        cls.model = DummyModel()

        cls.simple_influence = SimpleInfluence(cls.model, cls.train_dataset)

    def test_compute_training_gradients(self):

        (
            training_instances,
            used_params,
        ) = self.simple_influence.compute_training_gradients()

        # Training instances

        self.assertTrue(isinstance(training_instances, list))
        self.assertTrue(isinstance(training_instances[0], dict))
        self.assertTrue(isinstance(training_instances[0]["grads"], tuple))
        self.assertTrue(isinstance(training_instances[0]["grads"][0], torch.Tensor))
        self.assertTrue(isinstance(training_instances[0]["idx"], int))
        self.assertTrue(isinstance(training_instances[0]["loss"], float))
        self.assertTrue(isinstance(training_instances[0]["batch"], dict))
        self.assertTrue(
            isinstance(training_instances[0]["batch"]["input_1"], torch.Tensor)
        )
        self.assertTrue(
            isinstance(training_instances[0]["batch"]["input_2"], torch.Tensor)
        )
        self.assertTrue(
            isinstance(training_instances[0]["batch"]["labels"], torch.Tensor)
        )

        # Used params

        self.assertTrue(isinstance(used_params, list))
        self.assertTrue(isinstance(used_params[0], torch.Tensor))

        expected_instances = [
            {
                "idx": 0,
                "batch": {
                    "input_1": torch.tensor(
                        [[-0.3831, -0.5763, 0.6454, -1.5312, 0.0025]],
                        dtype=torch.float64,
                    ),
                    "input_2": torch.tensor(
                        [[0.4453, 0.7360, -0.7296]], dtype=torch.float64
                    ),
                    "labels": torch.tensor([2]),
                },
                "loss": 0.029712873456184914,
                "grads": (
                    torch.tensor(
                        [
                            [
                                -2.9975e-03,
                                -4.5084e-03,
                                5.0497e-03,
                                -1.1980e-02,
                                1.9897e-05,
                            ],
                            [
                                -8.2191e-03,
                                -1.2362e-02,
                                1.3846e-02,
                                -3.2847e-02,
                                5.4557e-05,
                            ],
                            [
                                1.1217e-02,
                                1.6870e-02,
                                -1.8896e-02,
                                4.4827e-02,
                                -7.4454e-05,
                            ],
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor([0.0078, 0.0215, -0.0293], dtype=torch.float64),
                    torch.tensor(
                        [
                            [0.0035, 0.0058, -0.0057],
                            [0.0096, 0.0158, -0.0157],
                            [-0.0130, -0.0215, 0.0214],
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor([0.0078, 0.0215, -0.0293], dtype=torch.float64),
                ),
            }
        ]

        self.assertTrue(recursive_match(training_instances, expected_instances))
        expected_params = [
            torch.tensor(
                [
                    [-0.6341, 1.6370, 0.1173, -0.0087, 0.9386],
                    [1.2043, -0.9492, -0.5282, 0.5000, 0.0833],
                    [-0.7250, -1.1106, -0.3840, 0.6206, 0.9392],
                ],
                dtype=torch.float64,
                requires_grad=True,
            ),
            torch.tensor(
                [-1.5241, -0.7621, 0.8767], dtype=torch.float64, requires_grad=True
            ),
            torch.tensor(
                [
                    [0.2297, -1.9201, 0.0466],
                    [-0.2303, -1.0556, 0.1797],
                    [1.2785, 0.3740, -0.2608],
                ],
                dtype=torch.float64,
                requires_grad=True,
            ),
            torch.tensor(
                [0.5066, 0.8308, 0.2162], dtype=torch.float64, requires_grad=True
            ),
        ]
