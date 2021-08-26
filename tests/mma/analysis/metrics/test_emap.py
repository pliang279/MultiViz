import unittest
from collections import OrderedDict

import numpy as np

from src.mma.analysis.metrics.emap import Emap


class TestEmap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.dataset = OrderedDict(
            {
                "arg0": np.array(
                    [
                        [1.12743584, 1.13448005],
                        [0.50889023, 0.46084774],
                        [-0.28415428, -0.07687285],
                        [0.15706401, 0.26843169],
                        [-1.70497695, 1.19067221],
                    ]
                ),
                "arg1": np.array(
                    [
                        [-0.75966783, 1.14106388, 0.27610688],
                        [0.66013527, 0.27800452, -0.37362844],
                        [0.96804656, 0.21056764, 0.38410439],
                        [0.52634001, -0.4904973, -0.2707183],
                        [0.50303166, -1.27711476, 0.45915383],
                    ]
                ),
            }
        )

        cls.w0 = np.array([[0.33023155], [1.40766588]])
        cls.w1 = np.array([[-0.89154714], [-0.78821395], [1.50791622]])
        cls.b = np.array([[0.17281638]])
        cls.predictor_fn = lambda arg0, arg1: np.concatenate(
            [(arg0 @ cls.w0) * (arg1 @ cls.w1), (arg0 @ cls.w0) + (arg1 @ cls.w1)],
            axis=1,
        )

        cls.batch_size = 4

        cls.emap_obj = Emap(cls.predictor_fn, cls.dataset)

    def test_get_cartesian_product_of_indices(self):
        cartesian_product = self.emap_obj.get_cartesian_product_of_indices()
        self.assertEqual(cartesian_product.shape, (25, 2))

        self.assertTrue(
            (
                cartesian_product
                == np.array(
                    [
                        [0, 0],
                        [0, 1],
                        [0, 2],
                        [0, 3],
                        [0, 4],
                        [1, 0],
                        [1, 1],
                        [1, 2],
                        [1, 3],
                        [1, 4],
                        [2, 0],
                        [2, 1],
                        [2, 2],
                        [2, 3],
                        [2, 4],
                        [3, 0],
                        [3, 1],
                        [3, 2],
                        [3, 3],
                        [3, 4],
                        [4, 0],
                        [4, 1],
                        [4, 2],
                        [4, 3],
                        [4, 4],
                    ]
                )
            ).all()
        )

    def test_combination_generator(self):
        comb_gen = self.emap_obj.combination_generator(self.batch_size)
        batch = comb_gen.__next__()
        self.assertEqual(batch.keys(), self.dataset.keys())

        # Integration Test
        expected_output = OrderedDict(
            [
                (
                    "arg0",
                    np.array(
                        [
                            [1.12743584, 1.13448005],
                            [1.12743584, 1.13448005],
                            [1.12743584, 1.13448005],
                            [1.12743584, 1.13448005],
                        ]
                    ),
                ),
                (
                    "arg1",
                    np.array(
                        [
                            [-0.75966783, 1.14106388, 0.27610688],
                            [0.66013527, 0.27800452, -0.37362844],
                            [0.96804656, 0.21056764, 0.38410439],
                            [0.52634001, -0.4904973, -0.2707183],
                        ]
                    ),
                ),
            ]
        )
        for key in batch.keys():
            self.assertTrue(np.allclose(batch[key], expected_output[key], atol=1e-7))

    def test_dataset_generator(self):
        data_gen = self.emap_obj.dataset_generator(self.batch_size)
        batch = data_gen.__next__()
        self.assertEqual(batch.keys(), self.dataset.keys())

        # Integration Test
        for key in batch.keys():
            self.assertTrue(
                np.allclose(batch[key], self.dataset[key][: self.batch_size], atol=1e-7)
            )

    def test_compute_predictions(self):
        orig_predictions = self.emap_obj.compute_predictions("orig", self.batch_size)
        self.assertEqual(orig_predictions.shape, (5, 2))

        self.assertTrue(
            np.allclose(
                orig_predictions,
                self.emap_obj.predictor_fn(**self.dataset),
                atol=1e-5,
            )
        )

        emap_predictions = self.emap_obj.compute_predictions("emap", self.batch_size)
        self.assertEqual(emap_predictions.shape, (25, 2))

        # Integration Test
        self.assertTrue(
            np.allclose(
                orig_predictions,
                np.array(
                    [
                        [0.3824807, 2.163507],
                        [-1.11984986, -0.55429789],
                        [0.09088811, -0.65188226],
                        [-0.21093736, -0.06113102],
                        [1.39187783, 2.3635602],
                    ]
                ),
                atol=1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                emap_predictions,
                np.array(
                    [
                        [0.3824807, 2.163507],
                        [-2.70002418, 0.59821461],
                        [-0.8858513, 1.51944949],
                        [-0.96664388, 1.47842311],
                        [2.46264593, 3.21981249],
                        [0.15863597, 1.01099451],
                        [-1.11984986, -0.55429789],
                        [-0.3674117, 0.36693699],
                        [-0.40092086, 0.32591062],
                        [1.02139593, 2.0673],
                        [-0.03924242, -0.00782475],
                        [0.27702178, -1.57311714],
                        [0.09088811, -0.65188226],
                        [0.09917741, -0.69290864],
                        [-0.25266683, 1.04848074],
                        [0.08346349, 0.62395288],
                        [-0.58918902, -0.94133952],
                        [-0.19330711, -0.02010464],
                        [-0.21093736, -0.06113102],
                        [0.53738924, 1.68025836],
                        [0.21617659, 1.30725471],
                        [-1.52604308, -0.25803769],
                        [-0.50067968, 0.66319719],
                        [-0.54634333, 0.62217082],
                        [1.39187783, 2.3635602],
                    ]
                ),
                atol=1e-5,
            )
        )

    def test_compute_emap_scores(self):
        emap_scores = self.emap_obj.compute_emap_scores(self.batch_size)
        self.assertEqual(emap_scores.shape, (5, 2))

        # Integration Test
        self.assertTrue(
            np.allclose(
                emap_scores,
                np.array(
                    [
                        [-0.03805738, 2.163507],
                        [-1.13012867, -0.55429789],
                        [-0.19311842, -0.65188226],
                        [-0.33653145, -0.06113102],
                        [0.98224439, 2.3635602],
                    ]
                ),
                atol=1e-5,
            )
        )
