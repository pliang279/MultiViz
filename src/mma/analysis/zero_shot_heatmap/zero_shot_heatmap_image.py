from typing import Callable

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch


class ZeroShotHeatmapImage:
    """
    Implementation of the Zero-shot Image Retrieval Explanations using Heatmap.
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
    def gen_image_batch(self, image, resize_size=224, pixel_size=10):

        n_pixels = resize_size // pixel_size + 1

        image_batch = []
        masks = []

        image = self.pad_to_square(image, size=resize_size)
        gray = np.ones_like(image) * 128
        mask = np.ones_like(image)

        # Entire Image
        image_batch.append(image)
        masks.append(mask)

        # Vertical Pass
        for i in range(0, n_pixels):
            for j in range(i + 1, n_pixels):
                m = mask.copy()
                m[: min(i * pixel_size, resize_size) + 1, :] = 0
                m[min(j * pixel_size, resize_size) + 1 :, :] = 0
                neg_m = 1 - m

                # M refers to the viewing window, rest is filled with 128
                image_batch.append(image * m + gray * neg_m)
                masks.append(m)

        # Horizontal Pass
        for i in range(0, n_pixels + 1):
            for j in range(i + 1, n_pixels + 1):
                m = mask.copy()
                m[:, : min(i * pixel_size + 1, resize_size)] = 0
                m[:, min(j * pixel_size + 1, resize_size) :] = 0
                neg_m = 1 - m
                image_batch.append(image * m + gray * neg_m)
                masks.append(m)

        return image_batch, masks

    # Copied from the reference implementation
    def explain_instance(
        self,
        image: np.ndarray,
        text: str,
        image_encoder_fn: Callable,
        text_encoder_fn: Callable,
        image_preprocessor_fn: Callable = None,
        resize_size=224,
        pixel_size=10,
        iter=3,
    ):
        text_embedding = text_encoder_fn(text)
        images, masks = self.gen_image_batch(
            image, resize_size=resize_size, pixel_size=pixel_size
        )

        if image_preprocessor_fn is not None:
            images = torch.stack([image_preprocessor_fn(image) for image in images])

        image_embeddings = image_encoder_fn(images)

        sims = []
        scores = []
        mask_val = np.zeros_like(masks[0])

        for e, m in zip(image_embeddings, masks):
            print(e.shape)
            print(text_embedding.shape)
            sim = np.matmul(e, text_embedding.T)
            sims.append(sim)
            if len(sims) > 1:
                scores.append(sim * m)
                mask_val += 1 - m

        score = np.mean(np.clip(np.array(scores) - sims[0], 0, np.inf), axis=0)
        for i in range(iter):
            score = np.clip(score - np.mean(score), 0, np.inf)
        score = (score - np.min(score)) / (np.max(score) - np.min(score))
        return np.asarray(score)

    def plot_image_and_scores(image, text, scores):
        fig = plt.figure(figsize=(15, 30), facecolor="white")
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(133)

        ax1.imshow(image)
        ax2.imshow(np.asarray(image) / 255.0 * scores)

        ax1.axis("off")
        ax2.axis("off")

        ax2.title.set_text(text)

        plt.tight_layout()

        plt.show()
