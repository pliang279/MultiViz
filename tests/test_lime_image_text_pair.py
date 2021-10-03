import unittest

import numpy as np
from lime.lime_text import IndexedString
from mma.analysis.surrogates.lime.lime_image_text_pair import LimeImageTextPairExplainer


class TestLimeImageText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lime = LimeImageTextPairExplainer()

    def test_explain_instance(self):

        image = np.random.randn(4, 4, 3)

        def segmentation_fn(image):
            segments = np.ones(image.shape[:2])
            w, h = segments.shape
            segments[0 : w // 2, 0 : h // 2] = 0
            segments[0 : w // 2, h // 2 :] = 2
            segments[w // 2 :, h // 2 :] = 3
            return segments

        text = "This is an example text"
        text_to_token = {"this": 0, "is": 1, "an": 2, "example": 3, "text": 4, "": 5}

        def classifier_fn(images, texts):
            flattened_img = images.reshape(-1, 48)
            image_out = flattened_img @ np.array(
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

            samples = []
            for text in texts:
                if text.lower().split() == []:
                    samples.append([5])
                else:
                    samples.append(list(map(text_to_token.get, text.lower().split())))
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

            text_out = []
            for sample in samples:
                text_out.append(embedding[sample[0]])
                for j in range(1, len(sample)):
                    text_out[-1] += embedding[sample[j]]

            text_out = np.array(text_out)

            text_out = np.matmul(
                text_out,
                [
                    [1.44295072, -0.36536974],
                    [1.33581306, -0.27648498],
                    [-0.25141459, 1.78492958],
                    [1.09983086, -1.33080765],
                ],
            )

            out = image_out + text_out

            softmax_out = np.exp(out) / (1 + np.exp(out))

            return softmax_out

        (
            intercept,
            local_exp,
            scores,
            local_preds,
            image_instance,
            segments,
            indexed_string,
            new_indices,
        ) = self.lime.explain_instance(
            image,
            text,
            classifier_fn,
            labels=(0,),
            num_samples=5,
            num_features=4,
            segmentation_fn=segmentation_fn,
            random_seed=42,
        )

        self.assertTrue(isinstance(intercept[0], float))
        self.assertTrue(isinstance(local_exp[0], list))
        self.assertTrue(isinstance(local_exp[0][0], tuple))
        self.assertTrue(isinstance(local_exp[0][0][0], tuple))
        self.assertTrue(isinstance(local_exp[0][0][0][0], np.int64))
        self.assertTrue(isinstance(local_exp[0][0][1], np.float64))
        self.assertTrue(isinstance(scores[0], float))
        self.assertTrue(isinstance(image_instance, np.ndarray))
        self.assertTrue(isinstance(segments, np.ndarray))
        self.assertTrue(isinstance(indexed_string, IndexedString))

    def test_data_labels_distances(self):
        image = np.random.randn(4, 4, 3)

        segments = np.ones(image.shape[:2])
        w, h = segments.shape
        segments[0 : w // 2, 0 : h // 2] = 0
        segments[0 : w // 2, h // 2 :] = 2
        segments[w // 2 :, h // 2 :] = 3

        fudged_image = image.copy()

        for x in np.unique(segments):
            fudged_image[segments == x] = (
                np.mean(image[segments == x][:, 0]),
                np.mean(image[segments == x][:, 1]),
                np.mean(image[segments == x][:, 2]),
            )

        text = "This is an example text"
        indexed_string = IndexedString(text)
        text_to_token = {"this": 0, "is": 1, "an": 2, "example": 3, "text": 4, "": 5}

        def classifier_fn(images, texts):
            flattened_img = images.reshape(-1, 48)
            image_out = flattened_img @ np.array(
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

            samples = []
            for text in texts:
                if text.lower().split() == []:
                    samples.append([5])
                else:
                    samples.append(list(map(text_to_token.get, text.lower().split())))
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

            text_out = []
            for sample in samples:
                text_out.append(embedding[sample[0]])
                for j in range(1, len(sample)):
                    text_out[-1] += embedding[sample[j]]

            text_out = np.array(text_out)

            text_out = np.matmul(
                text_out,
                [
                    [1.44295072, -0.36536974],
                    [1.33581306, -0.27648498],
                    [-0.25141459, 1.78492958],
                    [1.09983086, -1.33080765],
                ],
            )

            out = image_out + text_out

            softmax_out = np.exp(out) / (1 + np.exp(out))

            return softmax_out

        num_samples = 5
        (
            data,
            labels,
            distances,
            total_num_image_features,
            total_num_text_features,
            new_indices,
        ) = self.lime.data_labels_distances(
            indexed_string, image, fudged_image, segments, classifier_fn, num_samples
        )

        self.assertTrue(isinstance(data, np.ndarray))
        self.assertTrue(isinstance(labels, np.ndarray))
        self.assertTrue(isinstance(distances, np.ndarray))
        self.assertTrue(isinstance(total_num_image_features, int))
        self.assertTrue(isinstance(total_num_image_features, int))

        self.assertEqual(
            data.shape,
            (num_samples, np.unique(segments).shape[0] * indexed_string.num_words()),
        )

        self.assertEqual(labels.shape, (num_samples, 2))

        self.assertEqual(distances.shape, (num_samples,))
