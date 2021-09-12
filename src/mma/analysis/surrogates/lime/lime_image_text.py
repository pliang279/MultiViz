"""
Functions for explaining classifiers that use both image and text data.
"""
import copy
from functools import partial

import numpy as np
import scipy as sp
import sklearn
from lime.lime_text import IndexedCharacters, IndexedString
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import gray2rgb
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from tqdm.auto import tqdm


class LimeImageTextExplainer:
    """Explains image (i.e. matrix) and text input classifiers.

    For images: If numerical features are present, perturb them by sampling
    from a Normal(0,1) and doing the inverse operation of mean-centering and
    scaling, according to the means and stds in the training data.
    For categorical features, perturb by sampling according to the training
    distribution, and making a binary feature that is 1 when the value is the
    same as the instance being explained.

    Currently, we are using an exponential kernel on cosine distance, and
    restricting explanations to words that are present in documents for text."""

    def __init__(
        self,
        image_kernel_width=0.25,  # TODO: Check defaults.
        text_kernel_width=0.25,
        common_kernel_width=0.25,
        image_kernel=None,
        text_kernel=None,
        common_kernel=None,
        verbose=False,
        class_names=None,
        feature_selection_method="auto",
        split_expression=r"\W+",
        bow=True,
        mask_string=None,
        random_state=None,
        char_level=False,
    ):
        """Init function.
        Args:
            image_kernel_width: kernel width for the exponential kernel for image.
            text_kernel_width: kernel width for the exponential kernel for text.
            image_kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
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
        image_kernel_width = float(image_kernel_width)
        text_kernel_width = float(text_kernel_width)
        common_kernel_width = float(common_kernel_width)

        if image_kernel is None:

            def image_kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.image_kernel_fn = partial(image_kernel, image_kernel_width)

        if text_kernel is None:

            def text_kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.text_kernel_fn = partial(text_kernel, text_kernel_width)

        if common_kernel is None:

            def common_kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.common_kernel_fn = partial(common_kernel, common_kernel_width)

        self.random_state = check_random_state(random_state)

        self.feature_selection_method = feature_selection_method
        self.class_names = class_names
        self.vocabulary = None
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level
        self.verbose = verbose

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.
        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel
        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(
            x_vector, weighted_labels, method="lasso", verbose=False
        )
        return alphas, coefs

    def forward_selection(
        self,
        data,
        labels,
        weights,
        num_features=None,
        num_image_features=None,
        num_text_features=None,
        split_index=None,
        bimodal_constrained=False,
    ):
        """Iteratively adds features to the model"""

        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []

        if bimodal_constrained:
            assert num_features is None
            assert num_image_features is not None
            assert num_text_features is not None
            assert split_index is not None
            num_features = num_image_features + num_text_features
            image_feat_count = 0
            text_feat_count = 0

        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            if bimodal_constrained:
                is_image = None
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                if bimodal_constrained:
                    if image_feat_count == num_image_features and feature < split_index:
                        continue
                    if text_feat_count == num_text_features and feature >= split_index:
                        continue
                clf.fit(
                    data[:, used_features + [feature]], labels, sample_weight=weights
                )
                score = clf.score(
                    data[:, used_features + [feature]], labels, sample_weight=weights
                )
                if score > max_:
                    best = feature
                    max_ = score
                    if bimodal_constrained:
                        if feature < split_index:
                            is_image = True
                        else:
                            is_image = False

            used_features.append(best)

            if bimodal_constrained:
                if is_image is True:
                    image_feat_count += 1
                elif is_image is False:
                    text_feat_count += 1
                else:
                    raise ValueError("is_image is not True or False")

        return np.array(used_features)

    def feature_selection(
        self,
        data,
        labels,
        weights,
        method,
        num_features=None,
        num_image_features=None,
        num_text_features=None,
        split_index=None,
        bimodal_constrained=False,
    ):
        """Selects features for the model. see explain_instance_with_data to
        understand the parameters."""

        if bimodal_constrained:
            assert num_features is None
            assert num_image_features is not None
            assert num_text_features is not None
            assert split_index is not None

        if method == "none":
            return np.array(range(data.shape[1]))
        elif method == "forward_selection":
            return self.forward_selection(
                data,
                labels,
                weights,
                num_features,
                num_image_features,
                num_text_features,
                split_index,
                bimodal_constrained,
            )
        elif method == "highest_weights":
            clf = Ridge(alpha=0.01, fit_intercept=True, random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                if not bimodal_constrained:
                    # Note: most efficient to slice the data before reversing
                    sdata = len(weighted_data.data)
                    argsort_data = np.abs(weighted_data.data).argsort()
                    # Edge case where data is more sparse than requested number of feature importances
                    # In that case, we just pad with zero-valued features
                    if sdata < num_features:
                        nnz_indexes = argsort_data[::-1]
                        indices = weighted_data.indices[nnz_indexes]
                        num_to_pad = num_features - sdata
                        indices = np.concatenate(
                            (indices, np.zeros(num_to_pad, dtype=indices.dtype))
                        )
                        indices_set = set(indices)
                        pad_counter = 0
                        for i in range(data.shape[1]):
                            if i not in indices_set:
                                indices[pad_counter + sdata] = i
                                pad_counter += 1
                                if pad_counter >= num_to_pad:
                                    break
                    else:
                        nnz_indexes = argsort_data[sdata - num_features : sdata][::-1]
                        indices = weighted_data.indices[nnz_indexes]
                    return indices
                else:

                    img_data = split_index
                    txt_data = data.shape[1] - img_data

                    argsort_image_data = np.abs(
                        weighted_data.data[:split_index]
                    ).argsort()
                    argsort_text_data = (
                        np.abs(weighted_data.data[split_index:]).argsort() + split_index
                    )

                    if img_data < num_image_features:
                        nnz_indexes = argsort_image_data[::-1]
                        img_indices = weighted_data.indices[nnz_indexes]
                        num_to_pad = num_image_features - img_data
                        img_indices = np.concatenate(
                            (img_indices, np.zeros(num_to_pad, dtype=img_indices.dtype))
                        )
                        indices_set = set(img_indices)
                        pad_counter = 0
                        for i in range(data.shape[1]):
                            if i not in indices_set:
                                img_indices[pad_counter + img_data] = i
                                pad_counter += 1
                                if pad_counter >= num_to_pad:
                                    break
                    else:
                        nnz_indexes = argsort_image_data[
                            img_data - num_image_features : img_data
                        ][::-1]
                        img_indices = weighted_data.indices[nnz_indexes]

                    if txt_data < num_text_features:
                        nnz_indexes = argsort_text_data[::-1]
                        text_indices = weighted_data.indices[nnz_indexes]
                        num_to_pad = num_text_features - txt_data
                        text_indices = np.concatenate(
                            (
                                text_indices,
                                np.ones(num_to_pad, dtype=text_indices.dtype)
                                * split_index,
                            )
                        )
                        indices_set = set(text_indices)
                        pad_counter = 0
                        for i in range(data.shape[1]):
                            if i not in indices_set:
                                text_indices[pad_counter + txt_data] = i
                                pad_counter += 1
                                if pad_counter >= num_to_pad:
                                    break
                    else:
                        nnz_indexes = argsort_text_data[
                            txt_data - num_text_features : txt_data
                        ][::-1]
                        text_indices = weighted_data.indices[nnz_indexes]

                    return img_indices + text_indices

            else:
                weighted_data = coef * data[0]

                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True,
                )
                if not bimodal_constrained:
                    return np.array([x[0] for x in feature_weights[:num_features]])
                else:
                    used_features = []
                    image_feat_count = 0
                    text_feat_count = 0
                    for i, (feature, weight) in enumerate(feature_weights):
                        if feature < split_index:
                            if image_feat_count == num_image_features:
                                continue
                            else:
                                used_features.append(feature)
                                image_feat_count += 1
                        else:
                            if text_feat_count == num_text_features:
                                continue
                            else:
                                used_features.append(feature)
                                text_feat_count += 1
                        if (
                            image_feat_count == num_image_features
                            and text_feat_count == num_text_features
                        ):
                            break
                    return np.array(used_features)

        elif method == "lasso_path":
            weighted_data = (
                data - np.average(data, axis=0, weights=weights)
            ) * np.sqrt(weights[:, np.newaxis])
            weighted_labels = (labels - np.average(labels, weights=weights)) * np.sqrt(
                weights
            )
            # NOTE: This has been modified to return an array instead of range
            nonzero = np.array(range(weighted_data.shape[1]))
            _, coefs = self.generate_lars_path(weighted_data, weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                if bimodal_constrained:
                    image_nonzero = coefs.T[i][:split_index].nonzero()[0]
                    text_nonzero = coefs.T[i][split_index:].nonzero()[0]
                    if (
                        len(image_nonzero) <= num_image_features
                        and len(text_nonzero) <= num_text_features
                    ):
                        break
                else:
                    nonzero = coefs.T[i].nonzero()[0]
                    if len(nonzero) <= num_features:
                        break
            if bimodal_constrained:
                used_features = np.concatenate(
                    (np.array(image_nonzero), np.array(text_nonzero) + split_index)
                )
            else:
                used_features = np.array(nonzero)
            return used_features

        elif method == "auto":
            if bimodal_constrained:
                if num_image_features + num_text_features <= 6:
                    n_method = "forward_selection"
                else:
                    n_method = "highest_weights"
            else:
                if num_features <= 6:
                    n_method = "forward_selection"
                else:
                    n_method = "highest_weights"
            return self.feature_selection(
                data,
                labels,
                weights,
                n_method,
                num_features,
                num_image_features,
                num_text_features,
                split_index,
                bimodal_constrained,
            )

    def explain_instance_with_data(
        self,
        neighborhood_labels,
        label,
        neighborhood_common_data=None,
        neighborhood_image_data=None,
        neighborhood_text_data=None,
        num_image_features=None,
        num_text_features=None,
        num_common_features=None,
        image_distances=None,
        text_distances=None,
        common_distances=None,
        feature_selection_method="auto",
        model_regressor=None,
        feature_selection_type="unimodal",
        split_index=None,
    ):

        if feature_selection_type == "unimodal":

            assert neighborhood_image_data is not None
            assert neighborhood_text_data is not None
            assert num_image_features is not None
            assert num_text_features is not None
            assert text_distances is not None
            assert image_distances is not None

            image_weights = self.image_kernel_fn(image_distances)

            labels_column = neighborhood_labels[:, label]
            used_image_features = self.feature_selection(
                neighborhood_image_data,
                labels_column,
                image_weights,
                feature_selection_method,
                num_image_features,
            )

            text_weights = self.text_kernel_fn(text_distances)

            labels_column = neighborhood_labels[:, label]
            used_text_features = self.feature_selection(
                neighborhood_text_data,
                labels_column,
                text_weights,
                feature_selection_method,
                num_text_features,
            )

            # Combine the two datasets
            image_data = neighborhood_image_data[:, used_image_features]
            text_data = neighborhood_text_data[:, used_text_features]

            used_data = np.hstack([image_data, text_data])

            # The sample weights are taken using the normalized sum of the image and text weights
            used_weights = image_weights + text_weights

            used_weights = used_weights / np.sum(used_weights)

        elif feature_selection_type == "bimodal":

            assert neighborhood_common_data is not None
            assert num_common_features is not None
            assert common_distances is not None

            used_weights = self.common_kernel_fn(common_distances)
            labels_column = neighborhood_labels[:, label]
            used_features = self.feature_selection(
                neighborhood_common_data,
                labels_column,
                used_weights,
                feature_selection_method,
                num_common_features,
            )

            # Combine the two datasets
            used_data = neighborhood_common_data[:, used_features]
        elif feature_selection_type == "bimodal_constrained":
            assert neighborhood_common_data is not None
            assert num_image_features is not None
            assert num_text_features is not None
            assert common_distances is not None
            assert split_index is not None

            used_weights = self.common_kernel_fn(common_distances)

            labels_column = neighborhood_labels[:, label]
            used_features = self.feature_selection(
                neighborhood_common_data,
                labels_column,
                used_weights,
                feature_selection_method,
                num_features=None,
                num_image_features=num_image_features,
                num_text_features=num_text_features,
                split_index=split_index,
                bimodal_constrained=True,
            )

            used_data = neighborhood_common_data[:, used_features]

        if model_regressor is None:
            model_regressor = Ridge(
                alpha=1, fit_intercept=True, random_state=self.random_state
            )

        easy_model = model_regressor
        easy_model.fit(used_data, labels_column, sample_weight=used_weights)
        prediction_score = easy_model.score(
            used_data, labels_column, sample_weight=used_weights
        )

        local_pred = easy_model.predict(used_data[0].reshape(1, -1))

        if self.verbose:
            print("Intercept", easy_model.intercept_)
            print(
                "Prediction_local",
                local_pred,
            )
            print("Right:", neighborhood_labels[0, label])
        return (
            easy_model.intercept_,
            easy_model.coef_,
            used_features
            if feature_selection_type != "unimodal"
            else (used_image_features, used_text_features),
            prediction_score,
            local_pred,
        )

    def explain_instance(
        self,
        image_instance,
        text_instance,
        classifier_fn,
        labels=(1,),
        hide_color=None,
        top_labels=None,
        num_image_features=10,
        num_text_features=10,
        num_common_features=20,
        num_image_samples=10,
        num_text_samples=10,
        batch_size=4,
        segmentation_fn=None,
        distance_metric="cosine",
        model_regressor=None,
        random_seed=None,
        progress_bar=True,
        feature_selection_type="unimodal",
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
            num_features: maximum number of image features present in explanation
            num_features: maximum number of text features present in explanation
            num_image_samples: size of the image neighborhood to learn the linear model
            num_text_samples: size of the text neighborhood to learn the linear model
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
            common_data,
            image_data,
            text_data,
            predictions,
            common_distances,
            image_distances,
            text_distances,
        ) = self.data_labels_distances(
            indexed_string,
            image_instance,
            fudged_image,
            segments,
            classifier_fn,
            num_image_samples,
            num_text_samples,
            batch_size,
            progress_bar,
            distance_metric,
        )

        if top_labels:
            labels = np.argsort(labels)[-top_labels:]

        intercept = {}
        coefs = {}
        features = {}
        prediction_scores = {}
        local_predictions = {}

        for label in labels:
            (
                intercept_i,
                coefs_i,
                features_i,
                prediction_score_i,
                local_prediction_i,
            ) = self.explain_instance_with_data(
                neighborhood_labels=predictions,
                label=label,
                neighborhood_common_data=common_data,
                neighborhood_image_data=image_data,
                neighborhood_text_data=text_data,
                num_image_features=num_image_features,
                num_text_features=num_text_features,
                num_common_features=num_common_features,
                image_distances=image_distances,
                text_distances=text_distances,
                common_distances=common_distances,
                feature_selection_method=self.feature_selection_method,
                model_regressor=model_regressor,
                feature_selection_type=feature_selection_type,
                split_index=image_data.shape[1],
            )

            intercept[label] = intercept_i
            coefs[label] = coefs_i
            features[label] = features_i
            prediction_scores[label] = prediction_score_i
            local_predictions[label] = local_prediction_i

        return (
            intercept,
            coefs,
            features,
            prediction_scores,
            local_predictions,
            image_data.shape[1],
            image_instance,
            segments,
            indexed_string,
        )

    def data_labels_distances(
        self,
        indexed_string,
        image,
        fudged_image,
        segments,
        classifier_fn,
        num_image_samples,
        num_text_samples,
        batch_size=10,
        progress_bar=True,
        distance_metric="cosine",
    ):
        def distance_fn(x, mul=1):

            return (
                sklearn.metrics.pairwise.pairwise_distances(
                    x, x[0].reshape(1, -1), metric=distance_metric
                ).ravel()
                * mul
            )  # NOTE: Maybe a multiplier is required here. For text, multiplier is 100, for images it is 1.

        # Text
        doc_size = indexed_string.num_words()
        sample = self.random_state.randint(1, doc_size + 1, num_text_samples - 1)
        text_data = np.ones((num_text_samples, doc_size))
        text_data[0] = np.ones(doc_size)
        features_range = range(doc_size)
        inverse_text_data = [indexed_string.raw_string()]
        for i, size in enumerate(sample, start=1):
            inactive = self.random_state.choice(features_range, size, replace=False)
            text_data[i, inactive] = 0
            inverse_text_data.append(indexed_string.inverse_removing(inactive))

        # Image
        n_features = np.unique(segments).shape[0]
        image_data = self.random_state.randint(
            0, 2, num_image_samples * n_features
        ).reshape((num_image_samples, n_features))
        labels = []
        image_data[0, :] = 1

        # Repeat the two datasets
        repeated_image_data = np.repeat(image_data, text_data.shape[0], axis=0)
        repeated_text_data = np.tile(text_data, (image_data.shape[0], 1))
        repeated_inverse_text_data = inverse_text_data * image_data.shape[0]

        imgs = []
        txts = []
        rows = (
            tqdm(zip(repeated_image_data, repeated_inverse_text_data))
            if progress_bar
            else zip(repeated_image_data, repeated_inverse_text_data)
        )
        for image_dat, text in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(image_dat == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            txts.append(text)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs), txts)
                labels.extend(preds)
                imgs = []
                txts = []

        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs), txts)
            labels.extend(preds)

        # Distances
        used_data = np.hstack([repeated_image_data, repeated_text_data])

        distances = distance_fn(used_data)

        image_distances = distance_fn(repeated_image_data)

        text_distances = distance_fn(repeated_text_data)

        return (
            used_data,
            repeated_image_data,
            repeated_text_data,
            np.array(labels),
            distances,
            image_distances,
            text_distances,
        )
