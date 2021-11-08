from typing import Callable

from IPython.core.display import HTML, display
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np


class ZeroShotHeatmapText:
    """
    Implementation of the Zero-shot Text Retrieval Explanations using Heatmap.
    Reference: https://huggingface.co/spaces/clip-italian/clip-italian-demo
    """

    # Copied from the reference implementation
    def pad_to_square(self, image, size=224):
        old_size = image.size
        ratio = float(size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        image = image.resize(new_size, Image.ANTIALIAS)
        new_image = Image.new("RGB", size=(size, size), color=(128, 128, 128))
        new_image.paste(image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
        return new_image

    # Copied from the reference implementation
    def gen_text_batch(self, text: str, max_window_size=None):

        words = text.split()
        num_words = len(words)
        if max_window_size is None:
            max_window_size = num_words

        text_batch = []
        masks = []

        mask = np.ones(num_words)

        if max_window_size < num_words:
            # Entire Text
            text_batch.append(text)
            masks.append(mask)

        # Horizontal Pass
        for window_size in range(1, max_window_size):
            for start_idx in range(num_words):
                m = mask.copy()
                m[:start_idx] = 0
                m[start_idx + window_size :] = 0
                text_batch.append(" ".join(words[start_idx : start_idx + window_size]))
                masks.append(m)

        return text_batch, masks

    # Copied from the reference implementation
    def explain_instance(
        self,
        image: np.ndarray,
        text: str,
        image_encoder_fn: Callable,
        text_encoder_fn: Callable,
        image_preprocessor_fn: Callable = None,
        max_window_size: int = None,
        resize_size:int = 224,
        iter: int = 3,
        batch_size: int = 4,
    ):
        
        image = self.pad_to_square(image, resize_size)
        
        if image_preprocessor_fn is not None:
            image = image_preprocessor_fn(image)
        image_embedding = image_encoder_fn(image)

        texts, masks = self.gen_text_batch(text, max_window_size=max_window_size)

        text_embeddings = []
        for i in range(0, len(texts), batch_size):
            text_embeddings.append(text_encoder_fn(texts[i : i + batch_size]))

        text_embeddings = np.vstack(text_embeddings)

        sims = []
        scores = []
        mask_val = np.zeros_like(masks[0])

        for e, m in zip(text_embeddings, masks):
            sim = np.matmul(e, image_embedding.T)
            sims.append(sim)
            if len(sims) > 1:
                scores.append(sim * m)
                mask_val += 1 - m

        score = np.mean(np.clip(np.array(scores) - sims[0], 0, np.inf), axis=0)
        for i in range(iter):
            score = np.clip(score - np.mean(score), 0, np.inf)
        score = (score - np.min(score)) / (np.max(score) - np.min(score))
        return np.asarray(score)

    def _get_color(self, attr):
        # clip values to prevent CSS errors (Values should be from [-1,1])
        attr = max(-1, min(1, attr))
        if attr > 0:
            hue = 220
            sat = 100
            lig = 100 - int(127 * attr)
        else:
            hue = 220
            sat = 100
            lig = 100 - int(-125 * attr)
        return "hsl({}, {}%, {}%)".format(hue, sat, lig)

    def format_special_tokens(self, token):
        """Convert <> to # if there are any HTML syntax tags.
        Example: '<Hello>' will be converted to '#Hello' to avoid confusion
        with HTML tags.
        Args:
            token (str): The token to be formatted.
        Returns:
            (str): The formatted token.
        """
        if token.startswith("<") and token.endswith(">"):
            return "#" + token.strip("<>")
        return token

    def format_word_importances(self, words, importances):
        if np.isnan(importances[0]):
            importances = np.zeros_like(importances)

        assert len(words) <= len(importances)
        tags = ["<div>"]
        for word, importance in zip(words, importances[: len(words)]):
            word = self.format_special_tokens(word)
            for character in word:  ## Printing Weird Words
                if ord(character) >= 128:
                    print(word)
                    break
            color = self._get_color(importance)
            unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                        line-height:1.75"><font color="black"> {word}\
                        </font></mark>'.format(
                color=color, word=word
            )
            tags.append(unwrapped_tag)
        tags.append("</div>")
        return HTML("".join(tags))
