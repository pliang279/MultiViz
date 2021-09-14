import unittest
from typing import List

import numpy as np
from lime.explanation import Explanation
from lime.lime_image import ImageExplanation

from mma.analysis.surrogates.lime.lime import Lime


class TestLime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lime = Lime()
        cls.image = np.random.randn(4, 4, 3)
        cls.text = "This is an example text"
        cls.text_to_token = {
            "this": 0,
            "is": 1,
            "an": 2,
            "example": 3,
            "text": 4,
            "": 5,
        }

    def test_explain_image_instance(self):
        explanation_params = {
            "top_labels": 1,
            "hide_color": 0,
            "num_samples": 4,
            "batch_size": 2,
        }

        def image_predictor_fn(input_arr: np.ndarray):
            flattened_img = input_arr.reshape(-1, 48)
            out = flattened_img @ np.array(
                [
                    [-0.46239574, -1.00840316],
                    [-0.42240455, -0.21117908],
                    [-0.21069564, -1.41805838],
                    [1.50885308, -0.18432689],
                    [-0.03385737, 0.69803818],
                    [-0.19933407, 2.01855091],
                    [0.41935625, 0.73155555],
                    [1.42863409, -0.9760189],
                    [1.06410404, -1.56751148],
                    [1.25512487, 0.24480091],
                    [1.87871926, -1.64382003],
                    [-0.41628892, 0.84749586],
                    [-0.10447285, -0.0045578],
                    [-1.75644721, 1.06660709],
                    [1.55842638, -0.34923455],
                    [-0.09480482, -0.86489727],
                    [1.55685448, -0.40840669],
                    [1.4483313, -0.09979141],
                    [0.45149348, 0.33204063],
                    [0.43450386, -1.44361579],
                    [-0.01622472, 0.45432919],
                    [0.68096675, -0.51783705],
                    [-1.38087113, -0.43861507],
                    [-0.25548623, 0.16771692],
                    [0.32190907, 0.68665366],
                    [1.10935884, 0.87016319],
                    [-1.96098897, -0.38803507],
                    [-0.02089441, -0.18649836],
                    [-0.74808197, 0.89777835],
                    [-2.74759371, 1.88853607],
                    [2.81505519, 1.54146173],
                    [-0.12090903, -0.72230394],
                    [1.04082037, -0.06557836],
                    [1.6938797, 1.60343131],
                    [0.32188712, -1.1895716],
                    [-0.21755283, -2.11117051],
                    [0.90423886, -0.68235335],
                    [-0.28474683, -1.73792945],
                    [1.24167549, 0.3152186],
                    [-2.20716953, 1.02690477],
                    [0.107402, 0.24978991],
                    [0.57063166, -0.71689151],
                    [-0.72398032, -0.90063213],
                    [0.60730169, -0.97122372],
                    [0.22247972, -0.78905264],
                    [-0.30092593, 0.47984969],
                    [1.76887006, -1.52377588],
                    [-0.26905585, 1.25123331],
                ]
            )
            softmax_out = np.exp(out) / (1 + np.exp(out))

            return softmax_out

        explanation = self.lime.explain_image_instance(
            image_predictor_fn, self.image, explanation_params=explanation_params
        )

        assert isinstance(explanation, ImageExplanation)

    def test_explain_text_instance(self):
        init_params = {"class_names": ["happy", "sad"]}
        explanation_params = {
            "num_features": 5,
            "num_samples": 100,
            "top_labels": 1,
        }

        def text_predictor_fn(input_texts: List[str]):
            samples = []
            for text in input_texts:
                if text.lower().split() == []:
                    samples.append([5])
                else:
                    samples.append(
                        list(map(self.text_to_token.get, text.lower().split()))
                    )
            embedding = np.array(
                [
                    [-1.879321, -0.37232661, 0.80240232, -0.19308255],
                    [-0.41636274, 0.87225002, 0.86812386, -1.03196489],
                    [-0.74629701, 1.70429363, 0.48370129, 2.32290763],
                    [1.02759878, -0.53779461, 0.21262688, -1.67208261],
                    [-1.26857806, 0.6205153, 0.31843821, 0.7938336],
                    [-0.89272226, 0.44016401, -0.12751301, -0.19996849],
                ]
            )

            out = []
            for sample in samples:
                out.append(embedding[sample[0]])
                for j in range(1, len(sample)):
                    out[-1] += embedding[sample[j]]

            out = np.array(out)

            softmax_out = np.exp(out) / (1 + np.exp(out))

            return softmax_out

        explanation = self.lime.explain_text_instance(
            text_predictor_fn,
            self.text,
            init_params=init_params,
            explanation_params=explanation_params,
        )

        assert isinstance(explanation, Explanation)
