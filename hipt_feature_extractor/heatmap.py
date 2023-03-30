import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from scipy.ndimage import zoom
from tqdm.notebook import tqdm
from torchvision import transforms


def get_unnormalized_last_selfattention(model, x):
    x = model.prepare_tokens(x)
    for i, block in enumerate(model.blocks):
        if i < len(model.blocks) - 1:
            x = block(x)
        else:
            attention_layer = block.attn

            B, N, C = x.shape
            qkv = (
                attention_layer.qkv(x)
                .reshape(
                    B, N, 3, attention_layer.num_heads, C // attention_layer.num_heads
                )
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * attention_layer.scale

            return attn


def extract_attention(feature_extractor, x):
    x = x.unsqueeze(0)
    batch_256, w_256, h_256 = feature_extractor.model.prepare_img_tensor(x)
    batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)
    batch_256 = rearrange(batch_256, "b c p1 p2 w h -> (b p1 p2) c w h")
    batch_256 = batch_256.to(feature_extractor.model.device256, non_blocking=True)
    features_cls256 = feature_extractor.model.model256(batch_256)

    attention_256 = feature_extractor.model.model256.get_last_selfattention(batch_256)
    nh = attention_256.shape[1]  # number of head
    attention_256 = attention_256[:, :, 0, 1:].reshape(256, nh, -1)
    attention_256 = attention_256.reshape(w_256 * h_256, nh, 16, 16)
    attention_256 = (
        nn.functional.interpolate(attention_256, scale_factor=16, mode="nearest")
        .cpu()
        .numpy()
    )
    attention_256 = rearrange(attention_256, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=16)

    features_grid256 = (
        features_cls256.reshape(w_256, h_256, 384)
        .transpose(0, 1)
        .transpose(0, 2)
        .unsqueeze(dim=0)
    )
    features_grid256 = features_grid256.to(
        feature_extractor.model.device4k, non_blocking=True
    )

    # attention_4k = fe.model.model4k.get_last_selfattention(features_grid256)
    attention_4k = get_unnormalized_last_selfattention(
        feature_extractor.model.model4k, features_grid256
    )
    nh = attention_4k.shape[1]  # number of head
    attention_4k = attention_4k[0, :, 0, 1:].reshape(nh, -1)
    attention_4k = attention_4k.reshape(nh, w_256, h_256)
    attention_4k = (
        nn.functional.interpolate(
            attention_4k.unsqueeze(0), scale_factor=256, mode="nearest"
        )[0]
        .cpu()
        .numpy()
    )

    return attention_256, attention_4k


def transform_image(x):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )

    return transform(x[:, :, :-1])


def extract_single_image_attention(feature_extractor, image_4k):
    transformed_image = transform_image(image_4k)
    return extract_attention(feature_extractor, transformed_image)
