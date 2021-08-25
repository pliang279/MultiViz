import unittest
from collections import OrderedDict

import numpy as np

from src.analysis.metrics.emap import Emap


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
            [(arg0 * cls.w0) * (arg1 * cls.w1), (arg0 * cls.w0) + (arg1 * cls.w1)],
            axis=1,
        )

        cls.batch_size = 4

        cls.emap_obj = Emap(cls.predictor_fn, cls.dataset)

    def test_get_cartesian_product_of_indices(self):
        cartesian_product = self.emap_obj.get_cartesian_product_of_indices()
        self.assertEqual(cartesian_product.shape, (25, 2))

        self.assertTrue(
            (cartesian_product==
            np.array(
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
            )).all()
        )

    def test_combination_generator(self):
        comb_gen = self.emap_obj.combination_generator(4)
        batch = comb_gen.__next__()
        self.assertEqual(batch.keys(), self.dataset.keys())
