import unittest

import numpy as np
from lime.lime_text import IndexedString

from src.mma.analysis.surrogates.lime.lime_image_text import LimeImageTextExplainer


class TestLimeImageText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lime = LimeImageTextExplainer()

    def test_generate_lars_path(self):
        weighted_data = np.array(
            [
                [-0.29383316, -0.92318155, -0.33500992, 0.46049398, -0.47126854],
                [0.0901286, 0.21606975, -1.15251943, -1.15601711, -0.4509742],
                [-0.45757916, 0.80990318, -0.58236119, -0.71683723, 1.05838359],
                [-0.0766647, -0.37549353, 1.40454813, 0.30042609, -0.56223881],
            ]
        )

        weighted_labels = np.array([-0.68763455, 0.67477582, 0.85570835, -1.7565166])
        alphas, coefs = self.lime.generate_lars_path(weighted_data, weighted_labels)

        self.assertTrue(isinstance(coefs, np.ndarray))
        self.assertTrue(isinstance(alphas, np.ndarray))

        self.assertTrue(coefs.shape == (5, 5))
        self.assertTrue(alphas.shape == (5,))

    def test_forward_selection(self):

        labels = np.array([0.66162433, 0.55406658, 0.73188359, 0.36736308])
        data = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            ]
        )
        weights = np.array([0.29786048, 0.02227938, 0.58934327, 0.09051687])

        # Unimodal or Bimodal without Constraints
        num_features = 5
        used_features = self.lime.forward_selection(
            data, labels, weights, num_features=num_features
        )
        self.assertTrue(isinstance(used_features, np.ndarray))
        self.assertTrue(used_features.shape == (num_features,))

        # Bimodal Constraint Case
        num_image_features = 3
        num_text_features = 4
        split_index = 5

        used_features = self.lime.forward_selection(
            data,
            labels,
            weights,
            num_image_features=num_image_features,
            num_text_features=num_text_features,
            split_index=split_index,
            bimodal_constrained=True,
        )

        self.assertTrue(isinstance(used_features, np.ndarray))
        self.assertTrue(
            used_features.shape == (num_image_features + num_text_features,)
        )

        count_image_featues = np.where(used_features < split_index, 1, 0).sum()
        count_text_features = np.where(used_features >= split_index, 1, 0).sum()

        self.assertTrue(count_image_featues == num_image_features)
        self.assertTrue(count_text_features == num_text_features)

    def test_feature_selection(self):

        labels = np.array([0.66162433, 0.55406658, 0.73188359, 0.36736308])
        data = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            ]
        )
        weights = np.array([0.29786048, 0.02227938, 0.58934327, 0.09051687])

        # Unimodal or Bimodal without Constraints
        num_features = 5

        # auto
        method = "auto"
        used_features = self.lime.feature_selection(
            data, labels, weights, num_features=num_features, method=method
        )
        self.assertTrue(isinstance(used_features, np.ndarray))
        self.assertTrue(used_features.shape == (num_features,))

        # highest weights
        method = "highest_weights"
        used_features = self.lime.feature_selection(
            data, labels, weights, num_features=num_features, method=method
        )

        self.assertTrue(isinstance(used_features, np.ndarray))
        self.assertTrue(used_features.shape == (num_features,))

        # forward selection
        method = "forward_selection"
        used_features = self.lime.feature_selection(
            data, labels, weights, num_features=num_features, method=method
        )

        self.assertTrue(isinstance(used_features, np.ndarray))
        self.assertTrue(used_features.shape == (num_features,))

        # lasso path
        method = "lasso_path"
        used_features = self.lime.feature_selection(
            data, labels, weights, num_features=num_features, method=method
        )

        self.assertTrue(isinstance(used_features, np.ndarray))
        self.assertTrue(used_features.shape[0] <= num_features)

        # Bimodal with Constraints
        num_image_features = 3
        num_text_features = 4
        split_index = 5

        # auto
        method = "auto"
        used_features = self.lime.feature_selection(
            data,
            labels,
            weights,
            method=method,
            num_image_features=num_image_features,
            num_text_features=num_text_features,
            split_index=split_index,
            bimodal_constrained=True,
        )
        self.assertTrue(isinstance(used_features, np.ndarray))
        self.assertTrue(
            used_features.shape == (num_image_features + num_text_features,)
        )

        count_image_featues = np.where(used_features < split_index, 1, 0).sum()
        count_text_features = np.where(used_features >= split_index, 1, 0).sum()

        self.assertTrue(count_image_featues == num_image_features)
        self.assertTrue(count_text_features == num_text_features)

        # highest weights
        method = "highest_weights"
        used_features = self.lime.feature_selection(
            data,
            labels,
            weights,
            method=method,
            num_image_features=num_image_features,
            num_text_features=num_text_features,
            split_index=split_index,
            bimodal_constrained=True,
        )

        self.assertTrue(isinstance(used_features, np.ndarray))
        self.assertTrue(
            used_features.shape == (num_image_features + num_text_features,)
        )

        count_image_featues = np.where(used_features < split_index, 1, 0).sum()
        count_text_features = np.where(used_features >= split_index, 1, 0).sum()

        self.assertTrue(count_image_featues == num_image_features)
        self.assertTrue(count_text_features == num_text_features)

        # forward selection
        method = "forward_selection"
        used_features = self.lime.feature_selection(
            data,
            labels,
            weights,
            method=method,
            num_image_features=num_image_features,
            num_text_features=num_text_features,
            split_index=split_index,
            bimodal_constrained=True,
        )

        self.assertTrue(isinstance(used_features, np.ndarray))
        self.assertTrue(
            used_features.shape == (num_image_features + num_text_features,)
        )
        count_image_featues = np.where(used_features < split_index, 1, 0).sum()
        count_text_features = np.where(used_features >= split_index, 1, 0).sum()

        self.assertTrue(count_image_featues == num_image_features)
        self.assertTrue(count_text_features == num_text_features)

        # lasso path
        method = "lasso_path"
        used_features = self.lime.feature_selection(
            data,
            labels,
            weights,
            method=method,
            num_image_features=num_image_features,
            num_text_features=num_text_features,
            split_index=split_index,
            bimodal_constrained=True,
        )

        self.assertTrue(isinstance(used_features, np.ndarray))
        self.assertTrue(
            used_features.shape[0] <= (num_image_features + num_text_features)
        )

        count_image_featues = np.where(used_features < split_index, 1, 0).sum()
        count_text_features = np.where(used_features >= split_index, 1, 0).sum()

        self.assertTrue(count_image_featues <= num_image_features)
        self.assertTrue(count_text_features <= num_text_features)

    def test_explain_instance_with_data(self):
        neighborhood_labels = np.array(
            [[0.66162433], [0.55406658], [0.73188359], [0.36736308]]
        )
        data = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            ]
        )
        label = 0

        # Bimodal
        common_distances = np.array([0.0, 0.5421619, 0.46116378, 1.38440934])
        num_common_features = 5
        (
            intercept,
            coef,
            features,
            prediction_score,
            local_pred,
        ) = self.lime.explain_instance_with_data(
            neighborhood_labels,
            label,
            neighborhood_common_data=data,
            num_common_features=num_common_features,
            common_distances=common_distances,
            feature_selection_type="bimodal",
        )

        self.assertTrue(isinstance(intercept, float))
        self.assertTrue(isinstance(coef, np.ndarray))
        self.assertTrue(isinstance(features, np.ndarray))
        self.assertTrue(isinstance(prediction_score, float))
        self.assertTrue(isinstance(local_pred, np.ndarray))
        self.assertTrue(coef.shape, (num_common_features,))
        self.assertTrue(features.shape, (num_common_features,))
        self.assertTrue(local_pred.shape, (1, 1))

        # Bimodal with Constraints
        num_image_features = 3
        num_text_features = 4

        (
            intercept,
            coef,
            features,
            prediction_score,
            local_pred,
        ) = self.lime.explain_instance_with_data(
            neighborhood_labels,
            label,
            neighborhood_common_data=data,
            common_distances=common_distances,
            feature_selection_type="bimodal_constrained",
            num_image_features=num_image_features,
            num_text_features=num_text_features,
            split_index=5,
        )

        self.assertTrue(isinstance(intercept, float))
        self.assertTrue(isinstance(coef, np.ndarray))
        self.assertTrue(isinstance(features, np.ndarray))
        self.assertTrue(isinstance(prediction_score, float))
        self.assertTrue(isinstance(local_pred, np.ndarray))
        self.assertTrue(coef.shape, (num_image_features + num_text_features,))
        self.assertTrue(features.shape, (num_image_features + num_text_features,))
        self.assertTrue(local_pred.shape, (1, 1))

        # Unimodal
        image_distances = np.array([0.0, 0.57259863, -0.21909499, -1.16386149])
        text_distances = np.array([0.0, -0.77227573, 0.41658478, 0.27931626])

        neighborhood_image_data = np.array(
            [[1, 1, 1, 1, 1], [0, 1, 1, 0, 1], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0]]
        )
        neighborhood_text_data = np.array(
            [[1, 1, 1, 1, 1], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0], [1, 0, 1, 1, 0]]
        )

        (
            intercept,
            coef,
            features,
            prediction_score,
            local_pred,
        ) = self.lime.explain_instance_with_data(
            neighborhood_labels,
            label,
            neighborhood_image_data=neighborhood_image_data,
            neighborhood_text_data=neighborhood_text_data,
            image_distances=image_distances,
            text_distances=text_distances,
            feature_selection_type="unimodal",
            num_image_features=num_image_features,
            num_text_features=num_text_features,
            split_index=5,
        )

        self.assertTrue(isinstance(intercept, float))
        self.assertTrue(isinstance(coef, np.ndarray))
        self.assertTrue(isinstance(features[0], np.ndarray))
        self.assertTrue(isinstance(features[1], np.ndarray))
        self.assertTrue(isinstance(prediction_score, float))
        self.assertTrue(isinstance(local_pred, np.ndarray))
        self.assertTrue(coef.shape, (num_image_features + num_text_features,))
        self.assertTrue(features[0].shape, (num_image_features,))
        self.assertTrue(features[1].shape, (num_text_features,))
        self.assertTrue(local_pred.shape, (1, 1))

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
        text_to_token = {
            "this": 0,
            "is": 1,
            "an": 2,
            "example": 3,
            "text": 4,
            "": 5,
        }

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
            coefs,
            features,
            prediction_scores,
            local_predictions,
            split_index,
        ) = self.lime.explain_instance(
            image,
            text,
            classifier_fn,
            labels=(0,),
            num_image_samples=3,
            num_text_samples=3,
            num_image_features=3,
            num_text_features=4,
            segmentation_fn=segmentation_fn,
            random_seed=42,
        )

        self.assertTrue(isinstance(intercept[0], float))
        self.assertTrue(isinstance(coefs[0], np.ndarray))
        self.assertTrue(isinstance(features[0], tuple))
        self.assertTrue(isinstance(features[0][0], np.ndarray))
        self.assertTrue(isinstance(features[0][1], np.ndarray))
        self.assertTrue(isinstance(prediction_scores[0], float))
        self.assertTrue(isinstance(local_predictions[0], np.ndarray))
        self.assertTrue(isinstance(split_index, int))

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
        text_to_token = {
            "this": 0,
            "is": 1,
            "an": 2,
            "example": 3,
            "text": 4,
            "": 5,
        }

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

        num_image_samples = 3
        num_text_samples = 3

        (
            used_data,
            image_data,
            text_data,
            labels,
            distances,
            image_distances,
            text_distances,
        ) = self.lime.data_labels_distances(
            indexed_string,
            image,
            fudged_image,
            segments,
            classifier_fn,
            num_image_samples,
            num_text_samples,
        )

        self.assertTrue(isinstance(used_data, np.ndarray))
        self.assertTrue(isinstance(image_data, np.ndarray))
        self.assertTrue(isinstance(text_data, np.ndarray))
        self.assertTrue(isinstance(labels, np.ndarray))
        self.assertTrue(isinstance(distances, np.ndarray))
        self.assertTrue(isinstance(image_distances, np.ndarray))
        self.assertTrue(isinstance(text_distances, np.ndarray))

        self.assertEqual(
            used_data.shape,
            (
                num_image_samples * num_text_samples,
                np.unique(segments).shape[0] + indexed_string.num_words(),
            ),
        )
        self.assertEqual(
            image_data.shape,
            (num_image_samples * num_text_samples, np.unique(segments).shape[0]),
        )
        self.assertEqual(
            text_data.shape,
            (num_image_samples * num_text_samples, indexed_string.num_words()),
        )

        self.assertEqual(labels.shape, (num_image_samples * num_text_samples, 2))

        self.assertEqual(distances.shape, (num_image_samples * num_text_samples,))
        self.assertEqual(image_distances.shape, (num_image_samples * num_text_samples,))
        self.assertEqual(text_distances.shape, (num_image_samples * num_text_samples,))
