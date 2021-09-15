import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mma.analysis.influence_functions.simple_influence import SimpleInfluence


def recursive_match(actual, expected):

    # Leaves
    if isinstance(actual, int):
        return actual == expected
    elif isinstance(actual, float):
        return np.isclose(actual, expected, rtol=1e-4)
    elif isinstance(actual, np.ndarray) and np.issubdtype(actual.dtype, np.floating):
        return np.allclose(actual, expected, atol=1e-4)
    elif isinstance(actual, torch.Tensor) and actual.dtype.is_floating_point:
        return torch.allclose(actual, expected, atol=1e-4)
    elif isinstance(actual, torch.Tensor) and not actual.dtype.is_floating_point:
        return actual == expected
    elif isinstance(actual, np.ndarray) and np.issubdtype(actual.dtype, np.integer):
        return actual == expected

    # Sequences
    elif isinstance(actual, list):
        return all(
            recursive_match(actual[idx], expected[idx]) for idx in range(len(actual))
        )

    elif isinstance(actual, tuple):
        return all(
            recursive_match(actual[idx], expected[idx]) for idx in range(len(actual))
        )

    elif isinstance(actual, dict):
        keys_match = all(key in actual for key in expected.keys())
        values_match = all(
            recursive_match(actual[key], expected[key]) for key in actual.keys()
        )

        return keys_match and values_match
    else:
        return False


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

        cls.test_loader = DataLoader(cls.test_dataset, shuffle=False, batch_size=1)

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

        cls.simple_influence = SimpleInfluence(
            cls.model, cls.train_dataset, lissa_batch_size=2
        )

    def test_compute_training_gradients(self):

        self.simple_influence.compute_training_gradients()

        # Training instances
        self.assertTrue(
            len(self.simple_influence.train_instances), len(self.train_dataset)
        )
        self.assertTrue(isinstance(self.simple_influence.train_instances, list))
        self.assertTrue(isinstance(self.simple_influence.train_instances[0], dict))
        self.assertTrue(
            isinstance(self.simple_influence.train_instances[0]["grads"], tuple)
        )
        self.assertTrue(
            isinstance(
                self.simple_influence.train_instances[0]["grads"][0], torch.Tensor
            )
        )
        self.assertTrue(
            isinstance(self.simple_influence.train_instances[0]["idx"], int)
        )
        self.assertTrue(
            isinstance(self.simple_influence.train_instances[0]["loss"], float)
        )
        self.assertTrue(
            isinstance(self.simple_influence.train_instances[0]["batch"], dict)
        )
        self.assertTrue(
            isinstance(
                self.simple_influence.train_instances[0]["batch"]["input_1"],
                torch.Tensor,
            )
        )
        self.assertTrue(
            isinstance(
                self.simple_influence.train_instances[0]["batch"]["input_2"],
                torch.Tensor,
            )
        )
        self.assertTrue(
            isinstance(
                self.simple_influence.train_instances[0]["batch"]["labels"],
                torch.Tensor,
            )
        )

        # Used params

        self.assertTrue(isinstance(self.simple_influence.used_params, list))
        self.assertTrue(isinstance(self.simple_influence.used_params[0], torch.Tensor))

        # Integration Test

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
            },
            {
                "idx": 1,
                "batch": {
                    "input_1": torch.tensor(
                        [[-1.7715, 0.1114, -0.0295, 0.5508, 1.3871]],
                        dtype=torch.float64,
                    ),
                    "input_2": torch.tensor(
                        [[-1.7978, -1.3391, 1.0548]], dtype=torch.float64
                    ),
                    "labels": torch.tensor([0]),
                },
                "loss": 0.07809868048890224,
                "grads": (
                    torch.tensor(
                        [
                            [0.1331, -0.0084, 0.0022, -0.0414, -0.1042],
                            [-0.0477, 0.0030, -0.0008, 0.0148, 0.0374],
                            [-0.0854, 0.0054, -0.0014, 0.0265, 0.0669],
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor([-0.0751, 0.0269, 0.0482], dtype=torch.float64),
                    torch.tensor(
                        [
                            [0.1351, 0.1006, -0.0792],
                            [-0.0484, -0.0361, 0.0284],
                            [-0.0866, -0.0645, 0.0508],
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor([-0.0751, 0.0269, 0.0482], dtype=torch.float64),
                ),
            },
            {
                "idx": 2,
                "batch": {
                    "input_1": torch.tensor(
                        [[-0.9410, -0.5021, 1.1575, 1.4732, -1.2618]],
                        dtype=torch.float64,
                    ),
                    "input_2": torch.tensor(
                        [[0.0560, 0.9874, 1.7324]], dtype=torch.float64
                    ),
                    "labels": torch.tensor([2]),
                },
                "loss": 0.05573753938454816,
                "grads": (
                    torch.tensor(
                        [
                            [-0.0029, -0.0016, 0.0036, 0.0046, -0.0039],
                            [-0.0481, -0.0257, 0.0591, 0.0753, -0.0645],
                            [0.0510, 0.0272, -0.0628, -0.0799, 0.0684],
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor([0.0031, 0.0511, -0.0542], dtype=torch.float64),
                    torch.tensor(
                        [
                            [0.0002, 0.0031, 0.0054],
                            [0.0029, 0.0504, 0.0885],
                            [-0.0030, -0.0535, -0.0939],
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor([0.0031, 0.0511, -0.0542], dtype=torch.float64),
                ),
            },
            {
                "idx": 3,
                "batch": {
                    "input_1": torch.tensor(
                        [[-0.1884, 0.4592, 1.7973, 0.3847, 0.1305]], dtype=torch.float64
                    ),
                    "input_2": torch.tensor(
                        [[0.6992, -0.0801, 1.7026]], dtype=torch.float64
                    ),
                    "labels": torch.tensor([1]),
                },
                "loss": 2.583315583412059,
                "grads": (
                    torch.tensor(
                        [
                            [-0.0770, 0.1876, 0.7343, 0.1572, 0.0533],
                            [0.1742, -0.4245, -1.6616, -0.3557, -0.1206],
                            [-0.0972, 0.2369, 0.9272, 0.1985, 0.0673],
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor([0.4086, -0.9245, 0.5159], dtype=torch.float64),
                    torch.tensor(
                        [
                            [0.2857, -0.0327, 0.6956],
                            [-0.6464, 0.0741, -1.5740],
                            [0.3607, -0.0413, 0.8784],
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor([0.4086, -0.9245, 0.5159], dtype=torch.float64),
                ),
            },
            {
                "idx": 4,
                "batch": {
                    "input_1": torch.tensor(
                        [[-0.8610, -0.0448, -0.0632, 0.6182, -1.3874]],
                        dtype=torch.float64,
                    ),
                    "input_2": torch.tensor(
                        [[-0.2675, -0.7692, -0.1456]], dtype=torch.float64
                    ),
                    "labels": torch.tensor([2]),
                },
                "loss": 0.8579207256543936,
                "grads": (
                    torch.tensor(
                        [
                            [-0.1758, -0.0092, -0.0129, 0.1262, -0.2832],
                            [-0.3201, -0.0167, -0.0235, 0.2298, -0.5158],
                            [0.4959, 0.0258, 0.0364, -0.3560, 0.7991],
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor([0.2041, 0.3718, -0.5760], dtype=torch.float64),
                    torch.tensor(
                        [
                            [-0.0546, -0.1570, -0.0297],
                            [-0.0995, -0.2860, -0.0541],
                            [0.1541, 0.4430, 0.0839],
                        ],
                        dtype=torch.float64,
                    ),
                    torch.tensor([0.2041, 0.3718, -0.5760], dtype=torch.float64),
                ),
            },
        ]

        self.assertTrue(
            recursive_match(self.simple_influence.train_instances, expected_instances)
        )

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

        self.assertTrue(
            recursive_match(self.simple_influence.used_params, expected_params)
        )

    def test_flatten_tensors(self):
        tensors = [
            torch.tensor(
                [
                    [0.4352, 0.4047, -0.0724],
                    [-0.6185, 0.5521, -1.2561],
                    [-0.6518, 0.6810, 0.8875],
                ]
            ),
            torch.tensor(
                [
                    [0.7553, -1.3607, 0.8686],
                    [-1.7059, 0.1793, -1.3230],
                    [-1.0763, -0.2307, -0.5403],
                ]
            ),
            torch.tensor(
                [
                    [0.2088, 0.1012, -0.5556],
                    [0.3390, -0.1571, -0.1666],
                    [0.1620, -1.5969, 2.5991],
                ]
            ),
        ]

        flattened_tensors = self.simple_influence._flatten_tensors(tensors)

        self.assertTrue(isinstance(flattened_tensors, torch.Tensor))

        expected_tensor = torch.tensor(
            [
                0.4352,
                0.4047,
                -0.0724,
                -0.6185,
                0.5521,
                -1.2561,
                -0.6518,
                0.6810,
                0.8875,
                0.7553,
                -1.3607,
                0.8686,
                -1.7059,
                0.1793,
                -1.3230,
                -1.0763,
                -0.2307,
                -0.5403,
                0.2088,
                0.1012,
                -0.5556,
                0.3390,
                -0.1571,
                -0.1666,
                0.1620,
                -1.5969,
                2.5991,
            ]
        )

        self.assertTrue(torch.allclose(flattened_tensors, expected_tensor, atol=1e-4))

    def test_get_hvp(self):
        params = [
            torch.tensor(
                [
                    [-0.1283, -0.2313, 0.0322],
                    [0.2294, 0.7183, -0.3843],
                    [0.3333, 1.6089, 0.0283],
                ],
                requires_grad=True,
            ),
            torch.tensor(
                [
                    [0.2244, -0.1412, -0.0866],
                    [-0.2326, 0.7083, 1.0258],
                    [-0.3466, 0.2819, 0.1800],
                ],
                requires_grad=True,
            ),
            torch.tensor(
                [
                    [-0.8745, -0.8036, 0.0846],
                    [-0.6761, 0.7260, -1.5931],
                    [-0.7515, 2.1729, 0.2600],
                ],
                requires_grad=True,
            ),
        ]

        vectors = [
            torch.tensor(
                [
                    [0.3285, 0.6391, 0.7913],
                    [0.0600, 0.0991, -1.2980],
                    [0.8371, 0.5562, 0.6932],
                ],
                requires_grad=True,
            ),
            torch.tensor(
                [
                    [-4.5721e-01, 2.8054e00, 2.3653e-01],
                    [-6.0028e-01, 3.8541e-01, -1.1073e00],
                    [-1.3738e00, 5.8845e-01, -1.6985e-03],
                ],
                requires_grad=True,
            ),
            torch.tensor(
                [
                    [-1.4190, -2.2052, 0.6378],
                    [0.5115, 1.7654, -1.5282],
                    [-0.1182, 0.1098, -0.2531],
                ],
                requires_grad=True,
            ),
        ]

        loss = 2 * torch.sum(torch.cat(params)) * torch.sum(torch.cat(params))

        hvp = self.simple_influence.get_hvp(loss, params, vectors)

        self.assertTrue(isinstance(hvp, tuple))
        self.assertTrue(isinstance(hvp[0], torch.Tensor))

        self.assertTrue(
            recursive_match(
                hvp,
                (
                    torch.tensor(
                        [
                            [2.7312, 2.7312, 2.7312],
                            [2.7312, 2.7312, 2.7312],
                            [2.7312, 2.7312, 2.7312],
                        ]
                    ),
                    torch.tensor(
                        [
                            [2.7312, 2.7312, 2.7312],
                            [2.7312, 2.7312, 2.7312],
                            [2.7312, 2.7312, 2.7312],
                        ]
                    ),
                    torch.tensor(
                        [
                            [2.7312, 2.7312, 2.7312],
                            [2.7312, 2.7312, 2.7312],
                            [2.7312, 2.7312, 2.7312],
                        ]
                    ),
                ),
            )
        )

    def test_get_inverse_hvp_lissa(self):
        vs = [
            torch.tensor(
                [
                    [0.7359, 0.0716, 0.5422, -0.8484, 1.4143],
                    [-0.1642, 1.2630, 0.8645, 0.2743, -0.6648],
                    [0.7100, 0.2042, 0.5055, -0.5210, -1.2863],
                ]
            ),
            torch.tensor([-0.3059, -2.4745, 1.1289]),
            torch.tensor(
                [
                    [0.1843, -1.8002, -0.0373],
                    [0.1956, -0.6378, -0.4022],
                    [-0.1695, 0.2596, 0.9131],
                ]
            ),
            torch.tensor([-1.4436, 0.7668, 0.1103]),
        ]

        inv_hvp = self.simple_influence.get_inverse_hvp_lissa(vs)

        expected_out = torch.tensor(
            [
                7.3590e-05,
                7.1600e-06,
                5.4220e-05,
                -8.4840e-05,
                1.4143e-04,
                -1.6420e-05,
                1.2630e-04,
                8.6450e-05,
                2.7430e-05,
                -6.6480e-05,
                7.1000e-05,
                2.0420e-05,
                5.0550e-05,
                -5.2100e-05,
                -1.2863e-04,
                -3.0590e-05,
                -2.4745e-04,
                1.1289e-04,
                1.8430e-05,
                -1.8002e-04,
                -3.7300e-06,
                1.9560e-05,
                -6.3780e-05,
                -4.0220e-05,
                -1.6950e-05,
                2.5960e-05,
                9.1310e-05,
                -1.4436e-04,
                7.6680e-05,
                1.1030e-05,
            ]
        )

        self.assertTrue(torch.allclose(inv_hvp, expected_out, atol=1e-4))

    def test_calculate_influence_scores(self):
        test_grads = [
            torch.tensor(
                [
                    [0.7359, 0.0716, 0.5422, -0.8484, 1.4143],
                    [-0.1642, 1.2630, 0.8645, 0.2743, -0.6648],
                    [0.7100, 0.2042, 0.5055, -0.5210, -1.2863],
                ],
                dtype=torch.float64,
            ),
            torch.tensor([-0.3059, -2.4745, 1.1289], dtype=torch.float64),
            torch.tensor(
                [
                    [0.1843, -1.8002, -0.0373],
                    [0.1956, -0.6378, -0.4022],
                    [-0.1695, 0.2596, 0.9131],
                ],
                dtype=torch.float64,
            ),
            torch.tensor([-1.4436, 0.7668, 0.1103], dtype=torch.float64),
        ]

        influence_scores = self.simple_influence._calculate_influence_scores(test_grads)

        self.assertEqual(len(influence_scores), 5)

        self.assertTrue(
            np.allclose(
                influence_scores,
                [
                    -1.0423861639986597e-05,
                    -1.4525943204495106e-05,
                    -2.879593033265721e-05,
                    0.0001314994426739569,
                    -0.00017758964756792146,
                ],
                atol=1e-5,
            )
        )

    def test_interpret_instances(self):
        outputs = self.simple_influence.interpret_instances(self.test_loader)

        self.assertTrue(isinstance(outputs, list))

        self.assertTrue(
            recursive_match(
                outputs,
                [
                    {
                        "idx": 0,
                        "batch": {
                            "input_1": torch.tensor(
                                [[0.4390, -0.5618, 0.0221, 0.2729, -2.0237]],
                                dtype=torch.float64,
                            ),
                            "input_2": torch.tensor(
                                [[-0.2609, -1.7251, -2.0585]], dtype=torch.float64
                            ),
                            "labels": torch.tensor([1]),
                        },
                        "loss": 0.06068094329987219,
                        "influence_scores": np.array(
                            [
                                -3.7067454e-07,
                                1.8347779e-07,
                                2.0212188e-07,
                                -1.6458258e-05,
                                -2.2013332e-05,
                            ],
                            dtype=np.float32,
                        ),
                    },
                    {
                        "idx": 1,
                        "batch": {
                            "input_1": torch.tensor(
                                [[0.4951, -0.1338, 0.0870, -0.5076, -0.6000]],
                                dtype=torch.float64,
                            ),
                            "input_2": torch.tensor(
                                [[0.9915, 1.0302, -1.2697]], dtype=torch.float64
                            ),
                            "labels": torch.tensor([2]),
                        },
                        "loss": 0.04869496369818666,
                        "influence_scores": np.array(
                            [
                                1.1503371e-06,
                                5.8013416e-07,
                                2.8494145e-07,
                                -1.1485598e-06,
                                5.4407610e-06,
                            ],
                            dtype=np.float32,
                        ),
                    },
                ],
            )
        )

    def test_interpret(self):
        instance = {
            "input_1": torch.tensor(
                [[-1.6634, -1.0129, 0.6935, -0.8271, -0.1680]], dtype=torch.float64
            ),
            "input_2": torch.tensor([[-1.3537, 0.5030, -1.2318]], dtype=torch.float64),
            "labels": torch.tensor([1]),
        }
        interpretations = self.simple_influence.interpret(instance)

        self.assertTrue(isinstance(interpretations, list))
        self.assertEqual(len(interpretations), 1)
        self.assertTrue(
            recursive_match(
                interpretations,
                [
                    {
                        "idx": 0,
                        "batch": {
                            "input_1": torch.tensor(
                                [[-1.6634, -1.0129, 0.6935, -0.8271, -0.1680]],
                                dtype=torch.float64,
                            ),
                            "input_2": torch.tensor(
                                [[-1.3537, 0.5030, -1.2318]], dtype=torch.float64
                            ),
                            "labels": torch.tensor([2]),
                        },
                        "loss": 0.04378295178826923,
                        "influence_scores": np.array(
                            [
                                1.1168834e-06,
                                -1.0322286e-06,
                                8.3707306e-07,
                                1.4568662e-06,
                                1.2710483e-05,
                            ],
                            dtype=np.float32,
                        ),
                    }
                ],
            )
        )
