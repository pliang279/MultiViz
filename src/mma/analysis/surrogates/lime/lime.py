"""Implementation of a wrapper around LIME.

Handles implementation of the LIME (Local Interpretable Model-Agnostic Explanations)
wrapper for the analysis of how important various parts of input are. Current implementation
supports unimodal explanations with specified type for one sample, keeping other 
inputs constant.
"""

from typing import Callable, Dict

import numpy as np
from lime.lime_image import LimeImageExplainer
from lime.lime_text import LimeTextExplainer


class Lime:
    """Implementation of wrapper around LIME (Local Interpretable Model-Agnostic Explanations)."""

    @classmethod
    def explain_image_instance(
        cls,
        predictor_fn: Callable,
        input_arr: np.ndarray,
        init_params: Dict = {},
        explanation_params: Dict = {},
    ):
        """Explain image instance.

        Args:
            predictor_fn: Predictor function which takes in a batch of inputs (not single input).
            input_arr: Input image as an array without a batch dimension.
            init_params: Initialization parameters for LIME Explainer.
            explanation_params: Parameters for LIME Explainer.

        Returns:
            lime.lime_image.ImageExplanation:
                Explanation of the image with given arguments
        """

        # Create LIME explainer
        lime_explainer = LimeImageExplainer(**init_params)

        # Run LIME explainer
        exp = lime_explainer.explain_instance(
            input_arr, predictor_fn, **explanation_params
        )

        return exp

    @classmethod
    def explain_text_instance(
        cls,
        predictor_fn: Callable,
        text: str,
        init_params: Dict = {},
        explanation_params: Dict = {},
    ):
        """Explain text instance.

        Args:
            predictor_fn: Predictor function which takes in a batch of inputs (not single input).
            text: Input text as a string.
            init_params: Initialization parameters for LIME Explainer.
            explanation_params: Parameters for LIME Explainer.

        Returns:
            lime.lime_text.TextExplanation:
                Explanation of the text with given arguments
        """

        # Create LIME explainer
        lime_explainer = LimeTextExplainer(**init_params)

        # Run LIME explainer
        exp = lime_explainer.explain_instance(text, predictor_fn, **explanation_params)

        return exp
