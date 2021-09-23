"""
Functions for explaining classifiers that use both image and text data.
"""
import copy
from functools import partial

import numpy as np
import scipy as sp
import sklearn
from lime import lime_base
from lime.lime_text import IndexedCharacters, IndexedString
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import gray2rgb
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from tqdm.auto import tqdm


class LimeImageTextPairExplainer:
    def __init__(
        self,
        kernel_width=0.25,
        kernel=None,
        verbose=False,
        class_names=None,
        feature_selection="auto",
        split_expression=r"\W+",
        bow=True,
        mask_string=None,
        random_state=None,
        char_level=False,
    ):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel for image.
            text_kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in `lime.lime_base.LimeBase`
                for details on what each of the options does.
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True (bag of words), will perturb input data by removing
                all occurrences of individual words or characters.
                Explanations will be in terms of these words. Otherwise, will
                explain in terms of word-positions, so that a word may be
                important the first time it appears and unimportant the second.
                Only set to false if the classifier uses word order in some way
                (bigrams, etc), or if you set char_level=True.
            mask_string: String used to mask tokens or characters if bow=False
                if None, will be 'UNKWORDZ' if char_level=False, chr(0)
                otherwise.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            char_level: an boolean identifying that we treat each character
                as an independent occurence in the string
        """
        kernel_width = float(kernel_width)

        if kernel is None:

            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width)

        self.random_state = check_random_state(random_state)

        self.base = lime_base.LimeBase(
            kernel_fn, verbose, random_state=self.random_state
        )
        self.feature_selection = feature_selection
        self.class_names = class_names
        self.vocabulary = None
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level

    def explain_instance(
        self,
        image_instance,
        text_instance,
        classifier_fn,
        labels=(1,),
        hide_color=None,
        top_labels=None,
        num_features=20,
        num_samples=10,
        batch_size=4,
        segmentation_fn=None,
        distance_metric="cosine",
        model_regressor=None,
        random_seed=None,
        progress_bar=True,
        ignore_words=None,
    ):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime.lime_base).

        Args:
            image_instance: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            text_instance: raw text string to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of image-text features present in explanation
            num_samples: size of the image-text neighborhood to learn the linear model
            batch_size: batch size for model predictions
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
                segmentation function
            distance_metric: the distance metric to use for sample weighting.
                Defaults to cosine similarity.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression in LimeBase. Must be a regressor
                with model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit().
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.
            ignore_words: list of words to ignore

        Returns:
            An ImageTextExplanation object with the corresponding explanations.
        """

        # Text
        indexed_string = (
            IndexedCharacters(text_instance, bow=self.bow, mask_string=self.mask_string)
            if self.char_level
            else IndexedString(
                text_instance,
                bow=self.bow,
                split_expression=self.split_expression,
                mask_string=self.mask_string,
            )
        )

        if self.class_names is None:
            self.class_names = [str(x) for x in range(len(labels))]

        # Image
        if len(image_instance.shape) == 2:
            image_instance = gray2rgb(image_instance)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm(
                "quickshift",
                kernel_size=4,
                max_dist=200,
                ratio=0.2,
                random_seed=random_seed,
            )
        segments = segmentation_fn(image_instance)

        fudged_image = image_instance.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image_instance[segments == x][:, 0]),
                    np.mean(image_instance[segments == x][:, 1]),
                    np.mean(image_instance[segments == x][:, 2]),
                )
        else:
            fudged_image[:] = hide_color

        (
            data,
            yss,
            distances,
            total_num_image_features,
            total_num_text_features,
            new_indices,
        ) = self.data_labels_distances(
            indexed_string,
            image_instance,
            fudged_image,
            segments,
            classifier_fn,
            num_samples,
            batch_size,
            progress_bar,
            distance_metric,
            ignore_words,
        )

        if top_labels:
            labels = np.argsort(labels)[-top_labels:]

        intercept = {}
        local_exp = {}
        scores = {}
        local_preds = {}

        for label in labels:
            (
                intercept_i,
                local_exp_i,
                score_i,
                local_preds_i,
            ) = self.base.explain_instance_with_data(
                data,
                yss,
                distances,
                label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection,
            )

            intercept[label] = intercept_i
            local_exp[label] = [
                ((idx // total_num_text_features, idx % total_num_text_features), val)
                for idx, val in local_exp_i
            ]
            scores[label] = score_i
            local_preds[label] = local_preds_i

        return (
            intercept,
            local_exp,
            scores,
            local_preds,
            image_instance,
            segments,
            indexed_string,
            new_indices,
        )

    def data_labels_distances(
        self,
        indexed_string,
        image,
        fudged_image,
        segments,
        classifier_fn,
        num_samples,
        batch_size=10,
        progress_bar=True,
        distance_metric="cosine",
        ignore_words=None,
    ):
        def distance_fn(x, mul=1):

            return (
                sklearn.metrics.pairwise.pairwise_distances(
                    x, x[0].reshape(1, -1), metric=distance_metric
                ).ravel()
                * mul
            )  # NOTE: Maybe a multiplier is required here. For text, multiplier is 100, for images it is 1.

        # Text

        new_indices = []
        if ignore_words is None:
            total_num_text_features = indexed_string.num_words()
        else:
            total_num_text_features = 0
            for idx, word in enumerate(indexed_string.inverse_vocab):
                if word in ignore_words:
                    continue
                else:
                    new_indices.append(idx)
                    total_num_text_features += 1

        new_indices = np.array(new_indices)

        # Image
        total_num_image_features = np.unique(segments).shape[0]

        data = self.random_state.randint(
            0, 2, num_samples * total_num_image_features * total_num_text_features
        ).reshape((num_samples, total_num_image_features * total_num_text_features))

        labels = []

        data[0, :] = 1

        def sample_to_excluded_tokens(sample):
            out = list(
                zip(
                    *[
                        (idx // total_num_text_features, idx % total_num_text_features)
                        for idx, val in enumerate(sample)
                        if val == 0
                    ]
                )
            )

            if out:
                excluded_segments, excluded_words = out
            else:
                excluded_segments = np.array([])
                excluded_words = np.array([])

            return np.unique(excluded_segments), np.unique(excluded_words)

        imgs = []
        txts = []
        rows = tqdm(data) if progress_bar else data
        for sample in rows:
            excluded_segments, excluded_words = sample_to_excluded_tokens(sample)
            temp = copy.deepcopy(image)
            zeros = excluded_segments
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if ignore_words is None:
                txts.append(indexed_string.inverse_removing(excluded_words))
            else:
                txts.append(
                    indexed_string.inverse_removing(new_indices[excluded_words])
                )
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs), txts)
                labels.extend(preds)
                imgs = []
                txts = []

        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs), txts)
            labels.extend(preds)

        distances = distance_fn(data)

        return (
            data,
            np.array(labels),
            distances,
            total_num_image_features,
            total_num_text_features,
            new_indices,
        )
