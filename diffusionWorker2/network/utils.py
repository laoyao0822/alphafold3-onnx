# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


from collections import abc
import numbers

import torch

# VALID_DTYPES = [np.float32, np.float64, np.int8, np.int32, np.int64, bool]


# def remove_invalidly_typed_feats(
#     batch: features.BatchDict,
# ) -> features.BatchDict:
#   """Remove features of types we don't want to send to the TPU e.g. strings."""
#   return {
#       k: v
#       for k, v in batch.items()
#       if hasattr(v, 'dtype') and v.dtype in VALID_DTYPES
#   }


# def bfloat16_getter(next_getter, value, context):
#   """Ensures that a bfloat16 parameter is provided by casting if necessary."""
#   if context.original_dtype == jnp.bfloat16:
#     if value.dtype != jnp.bfloat16:
#       value = value.astype(jnp.bfloat16)
#   return next_getter(value)


# @contextlib.contextmanager
# def bfloat16_context():
#   with hk.custom_getter(bfloat16_getter):
#     yield

def mask_mean(mask, value, dim=None, keepdim=False, eps=1e-10):
    """Masked mean."""

    # mask_shape = mask.shape
    # value_shape = value.shape
    #
    # assert len(mask_shape) == len(
    #     value_shape
    # ), 'Shapes are not compatible, shapes: {}, {}'.format(mask_shape, value_shape)
    #
    # if isinstance(dim, numbers.Integral):
    #     # print("is Integral")
    #     dim = [dim]
    # elif dim is None:
    #     # print("is None")
    #     dim = list(range(len(mask_shape)))
    #
    # # print("dim:",dim)
    # assert isinstance(
    #     dim, abc.Iterable
    # ), 'axis needs to be either an iterable, integer or "None"'
    #
    # broadcast_factor = 1.0
    # for dim_ in dim:
    #     value_size = value_shape[dim_]
    #     mask_size = mask_shape[dim_]
    #     if mask_size == 1:
    #         broadcast_factor *= value_size
    #     else:
    #         error = f'Shapes are not compatible, shapes: {mask_shape}, {value_shape}'
    #         assert mask_size == value_size, error
    broadcast_factor = 1.0
    # print("broadcast_factor:",broadcast_factor)
    return torch.sum(mask * value, keepdim=keepdim, dim=dim) / (
        torch.clamp(torch.sum(mask, keepdim=keepdim, dim=dim) * broadcast_factor, min=eps)
    )
