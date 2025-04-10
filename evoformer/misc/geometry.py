# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import dataclasses
from typing import Self

import torch


def unstack(value: torch.Tensor, dim: int = -1) -> list[torch.Tensor]:
    return [
        torch.squeeze(v, dim=dim)
        for v in torch.chunk(value, value.shape[dim], dim=dim)
    ]


# ref:
# @struct_of_array.StructOfArray(same_dtype=True)
# class Vec3Array:
#   """Vec3Array in 3 dimensional Space implemented as struct of arrays.

#   This is done in order to improve performance and precision.
#   On TPU small matrix multiplications are very suboptimal and will waste large
#   compute ressources, furthermore any matrix multiplication on TPU happens in
#   mixed bfloat16/float32 precision, which is often undesirable when handling
#   physical coordinates.

#   In most cases this will also be faster on CPUs/GPUs since it allows for easier
#   use of vector instructions.
#   """

#   x: jnp.ndarray = dataclasses.field(metadata={'dtype': jnp.float32})
#   y: jnp.ndarray
#   z: jnp.ndarray

#   def __post_init__(self):
#     if hasattr(self.x, 'dtype'):
#       if not self.x.dtype == self.y.dtype == self.z.dtype:
#         raise ValueError(
#             f'Type mismatch: {self.x.dtype}, {self.y.dtype}, {self.z.dtype}'
#         )
#       if not self.x.shape == self.y.shape == self.z.shape:
#         raise ValueError(
#             f'Shape mismatch: {self.x.shape}, {self.y.shape}, {self.z.shape}'
#         )

#   def __add__(self, other: Self) -> Self:
#     return jax.tree.map(lambda x, y: x + y, self, other)

#   def __sub__(self, other: Self) -> Self:
#     return jax.tree.map(lambda x, y: x - y, self, other)

#   def __mul__(self, other: Float) -> Self:
#     return jax.tree.map(lambda x: x * other, self)

#   def __rmul__(self, other: Float) -> Self:
#     return self * other

#   def __truediv__(self, other: Float) -> Self:
#     return jax.tree.map(lambda x: x / other, self)

#   def __neg__(self) -> Self:
#     return jax.tree.map(lambda x: -x, self)

#   def __pos__(self) -> Self:
#     return jax.tree.map(lambda x: x, self)

#   def cross(self, other: Self) -> Self:
#     """Compute cross product between 'self' and 'other'."""
#     new_x = self.y * other.z - self.z * other.y
#     new_y = self.z * other.x - self.x * other.z
#     new_z = self.x * other.y - self.y * other.x
#     return Vec3Array(new_x, new_y, new_z)

#   def dot(self, other: Self) -> Float:
#     """Compute dot product between 'self' and 'other'."""
#     return self.x * other.x + self.y * other.y + self.z * other.z

#   def norm(self, epsilon: float = 1e-6) -> Float:
#     """Compute Norm of Vec3Array, clipped to epsilon."""
#     # To avoid NaN on the backward pass, we must use maximum before the sqrt
#     norm2 = self.dot(self)
#     if epsilon:
#       norm2 = jnp.maximum(norm2, epsilon**2)
#     return jnp.sqrt(norm2)

#   def norm2(self):
#     return self.dot(self)

#   def normalized(self, epsilon: float = 1e-6) -> Self:
#     """Return unit vector with optional clipping."""
#     return self / self.norm(epsilon)
