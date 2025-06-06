# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

#

"""Model param loading."""

import bisect
import collections
from collections.abc import Iterator
import contextlib
import io
import os
import pathlib
import re
import struct
import sys
from typing import IO

# import haiku as hk
# import jax.numpy as jnp
import numpy as np
import zstandard

from enum import Enum
from dataclasses import dataclass
from functools import partial
from typing import Union, List

import torch

_DTYPE_MAP = { 

    "float32": torch.float32, 

    "bfloat16": torch.bfloat16, 

    "float16": torch.float16, 

    "uint8": torch.uint8, 

} 


class RecordError(Exception):
  """Error reading a record."""


# def encode_record(scope: str, name: str, arr: np.ndarray) -> bytes:
#   """Encodes a single haiku param as bytes, preserving non-numpy dtypes."""
#   scope = scope.encode('utf-8')
#   name = name.encode('utf-8')
#   shape = arr.shape
#   dtype = str(arr.dtype).encode('utf-8')
#   arr = np.ascontiguousarray(arr)
#   if sys.byteorder == 'big':
#     arr = arr.byteswap()
#   arr_buffer = arr.tobytes('C')
#   header = struct.pack(
#       '<5i', len(scope), len(name), len(dtype), len(shape), len(arr_buffer)
#   )
#   return header + b''.join(
#       (scope, name, dtype, struct.pack(f'{len(shape)}i', *shape), arr_buffer)
#   )


def _read_record(stream: IO[bytes]) -> tuple[str, str, np.ndarray] | None:
  """Reads a record encoded by `_encode_record` from a byte stream."""
  header_size = struct.calcsize('<5i')
  header = stream.read(header_size)
  if not header:
    return None
  if len(header) < header_size:
    raise RecordError(f'Incomplete header: {header}')
  (scope_len, name_len, dtype_len, shape_len, arr_buffer_len) = struct.unpack(
      '<5i', header
  )
  fmt = f'<{scope_len}s{name_len}s{dtype_len}s{shape_len}i'
  payload_size = struct.calcsize(fmt) + arr_buffer_len
  payload = stream.read(payload_size)
  if len(payload) < payload_size:
    raise RecordError(f'Incomplete payload: {payload}')
  scope, name, dtype, *shape = struct.unpack_from(fmt, payload)
  scope = scope.decode('utf-8')
  name = name.decode('utf-8')
  dtype = dtype.decode('utf-8')
  arr = torch.frombuffer(
    bytearray(payload[-arr_buffer_len:]), dtype=_DTYPE_MAP[dtype])
  arr = torch.reshape(arr, shape)
  if sys.byteorder == 'big':
    arr = arr.byteswap()
  return scope, name, arr


def read_records(stream: IO[bytes]) -> Iterator[tuple[str, str, np.ndarray]]:
  """Fully reads the contents of a byte stream."""
  while record := _read_record(stream):
    yield record


class _MultiFileIO(io.RawIOBase):
  """A file-like object that presents a concatenated view of multiple files."""

  def __init__(self, files: list[pathlib.Path]):
    self._files = files
    self._stack = contextlib.ExitStack()
    self._handles = [
        self._stack.enter_context(file.open('rb')) for file in files
    ]
    self._sizes = []
    for handle in self._handles:
      handle.seek(0, os.SEEK_END)
      self._sizes.append(handle.tell())
    self._length = sum(self._sizes)
    self._offsets = [0]
    for s in self._sizes[:-1]:
      self._offsets.append(self._offsets[-1] + s)
    self._abspos = 0
    self._relpos = (0, 0)

  def _abs_to_rel(self, pos: int) -> tuple[int, int]:
    idx = bisect.bisect_right(self._offsets, pos) - 1
    return idx, pos - self._offsets[idx]

  def close(self):
    self._stack.close()

  def closed(self) -> bool:
    return all(handle.closed for handle in self._handles)

  def fileno(self) -> int:
    return -1

  def readable(self) -> bool:
    return True

  def tell(self) -> int:
    return self._abspos

  def seek(self, pos: int, whence: int = os.SEEK_SET, /):
    match whence:
      case os.SEEK_SET:
        pass
      case os.SEEK_CUR:
        pos += self._abspos
      case os.SEEK_END:
        pos = self._length - pos
      case _:
        raise ValueError(f'Invalid whence: {whence}')
    self._abspos = pos
    self._relpos = self._abs_to_rel(pos)

  def readinto(self, b: bytearray | memoryview) -> int:
    result = 0
    mem = memoryview(b)
    while mem:
      self._handles[self._relpos[0]].seek(self._relpos[1])
      count = self._handles[self._relpos[0]].readinto(mem)
      result += count
      self._abspos += count
      self._relpos = self._abs_to_rel(self._abspos)
      mem = mem[count:]
      if self._abspos == self._length:
        break
    return result


@contextlib.contextmanager
def open_for_reading(model_files: list[pathlib.Path], is_compressed: bool):
  with contextlib.closing(_MultiFileIO(model_files)) as f:
    if is_compressed:
      yield zstandard.ZstdDecompressor().stream_reader(f)
    else:
      yield f


def _match_model(
    paths: list[pathlib.Path], pattern: re.Pattern[str]
) -> dict[str, list[pathlib.Path]]:
  """Match files in a directory with a pattern, and group by model name."""
  models = collections.defaultdict(list)
  for path in paths:
    match = pattern.fullmatch(path.name)
    if match:
      models[match.group('model_name')].append(path)
  return {k: sorted(v) for k, v in models.items()}


def select_model_files(
    model_dir: pathlib.Path, model_name: str | None = None
) -> tuple[list[pathlib.Path], bool]:
  """Select the model files from a model directory."""
  files = [file for file in model_dir.iterdir() if file.is_file()]

  for pattern, is_compressed in (
      (r'(?P<model_name>.*)\.[0-9]+\.bin\.zst$', True),
      (r'(?P<model_name>.*)\.bin\.zst\.[0-9]+$', True),
      (r'(?P<model_name>.*)\.[0-9]+\.bin$', False),
      (r'(?P<model_name>.*)\.bin]\.[0-9]+$', False),
      (r'(?P<model_name>.*)\.bin\.zst$', True),
      (r'(?P<model_name>.*)\.bin$', False),
  ):
    models = _match_model(files, re.compile(pattern))
    if model_name is not None:
      if model_name in models:
        return models[model_name], is_compressed
    else:
      if models:
        if len(models) > 1:
          raise RuntimeError(f'Multiple models matched in {model_dir}')
        _, model_files = models.popitem()
        return model_files, is_compressed
  raise FileNotFoundError(f'No models matched in {model_dir}')


# def get_model_haiku_params(model_dir: pathlib.Path) -> hk.Params:
#   """Get the Haiku parameters from a model name."""
#   params: dict[str, dict[str, jnp.Array]] = {}
#   model_files, is_compressed = select_model_files(model_dir)
#   with open_for_reading(model_files, is_compressed) as stream:
#     for scope, name, arr in read_records(stream):
#       params.setdefault(scope, {})[name] = jnp.array(arr)
#   if not params:
#     raise FileNotFoundError(f'Model missing from "{model_dir}"')
#   return params

def get_model_haiku_params_to_torch(model_dir: pathlib.Path):
    if not os.path.exists(model_dir):
        raise Exception(
            f"Given checkpoint path not exist [{model_dir}]")
    # print(f"Loading from {model_dir}")
    is_compressed = False
    if model_dir.suffix == ".zst":
        is_compressed = True
    params = {}
    with open_for_reading([pathlib.Path(model_dir)], is_compressed) as stream:
        for scope, name, arr in read_records(stream):
            params[f"{scope}/{name}"] = arr
    return params

class ParamType(Enum):
    LinearWeight = partial( 
        lambda w: w.transpose(-1, -2)
    )
    LinearWeightMHA = partial(
        lambda w: w.reshape(*w.shape[:-2], -1).transpose(-1, -2)
    )
    LinearWeightNoTransposeMHA = partial(
        lambda w: w.reshape(-1, w.shape[-1])
    )
    LinearBiasMHA = partial(lambda w: w.reshape(*w.shape[:-2], -1))
    LinearFlat = partial(lambda w: w.unsqueeze(-1))
    Other = partial(lambda w: w)

    def __init__(self, fn):
        self.transformation = fn

@dataclass
class Param:
    param: Union[torch.Tensor, List[torch.Tensor]]
    param_type: ParamType = ParamType.Other
    stacked: bool = False

def stacked(param_dict_list, output=None):
    if output is None:
        output = {}
    template = param_dict_list[0]
    for k, _ in template.items():
        v = [d[k] for d in param_dict_list]
        # 嵌套的情况
        if type(v[0]) is dict:
            output[k] = {}
            stacked(v, output=output[k])
        # 参数的情况
        elif type(v[0]) is Param:
            stacked_param = Param(
                param=[param.param for param in v],
                param_type=v[0].param_type, # 使用第一个Param对象的param_type
                stacked=True,
            )

            output[k] = stacked_param

    return output

def assign(translation_dict, param_to_load):
    for k, param in translation_dict.items():
        with torch.no_grad(): # 没有必要计算梯度
            weights = torch.as_tensor(param_to_load[k]) # af3的k权重转换为tensor
            ref, param_type = param.param, param.param_type
            if param.stacked:
                # 如果长度与ref长度相同
                if len(ref) == weights.shape[0]:
                    weights = torch.unbind(weights, 0)
                # 两级嵌套
                elif len(ref) == weights.shape[0] * weights.shape[1]:
                    # 将weights展平
                    weights = torch.unbind(
                        weights.reshape(-1, *weights.shape[2:]), 0)
            else: # 如果不是stacked
                weights = [weights]
                ref = [ref]

            weights = list(map(param_type.transformation, weights))
            for p, w in zip(ref, weights):
                p.copy_(w)

def _process_translations_dict(d, _key_prefix, top_layer=True):
    flat = {}
    for k, v in d.items():
        if type(v) == dict:
            # 如果是顶层调用，则使用提供的_key_prefix作为前缀
            prefix = _key_prefix if top_layer else ""
            # 递归调用自身处理嵌套字典v
            sub_flat = {
                (prefix + "/".join([k, k_prime])): v_prime
                for k_prime, v_prime in _process_translations_dict(
                    v, _key_prefix, top_layer=False
                ).items()
            }
            flat.update(sub_flat)
        else:
            flat[k] = v

    return flat

def cat_params(params, prefix):
    return {
        f"{prefix}{k}": v
        for k, v in params.items()
    }
    
def LinearWeight(l, already_transpose_weights=False):
    if already_transpose_weights is True:
        return (Param(l))
    return (Param(l, param_type=ParamType.LinearWeight))


def LinearWeightMHA(l, already_transpose_weights=False):
    if already_transpose_weights is True:
        return (Param(l, param_type=ParamType.LinearWeightNoTransposeMHA))
    return (Param(l, param_type=ParamType.LinearWeightMHA))


def LinearBiasMHA(b): return (Param(b, param_type=ParamType.LinearBiasMHA))

def LinearParams(l, use_bias=False, already_transpose_weights=False):
    d = {"weights": LinearWeight(l.weight, already_transpose_weights)}

    if use_bias:
        d["bias"] = Param(l.bias)

    return d


def LinearfromFlatParams(l, use_bias=False):
    d = {"weights": Param(l.weight, param_type=ParamType.LinearFlat)}

    if use_bias:
        d["bias"] = Param(l.bias)

    return d

def LinearHMAParams(l, use_bias=False, already_transpose_weights=False):
    d = {"weights": LinearWeightMHA(l.weight, already_transpose_weights)}

    if use_bias:
        d["bias"] = LinearBiasMHA(l.bias)
    return d

# ref alphafold3/model/diffusion/diffusion/diffusion_transformer.py:adaptive_layernorm
def LayerNormParams(l, use_bias=True):
    d = {
        "scale": Param(l.weight),
    }
    if use_bias:
        d["offset"] = Param(l.bias)

    return d


def AdaptiveLayerNormParams(aln, use_single_cond=False):
    if use_single_cond is False:
        return {
            "layer_norm": LayerNormParams(aln.layer_norm),
        }
    else:
        return {
            "single_cond_layer_norm": LayerNormParams(aln.single_cond_layer_norm, use_bias=False),
            "single_cond_scale": LinearParams(aln.single_cond_scale, use_bias=True),
            "single_cond_bias": LinearParams(aln.single_cond_bias),
        }

# ref alphafold3/model/diffusion/diffusion/diffusion_transformer.py:adaptive_zero_init
def AdaLNZeroParams(ada_ln_zero, use_single_cond=False):
    d = {
        "transition2": LinearParams(ada_ln_zero.transition2),
    }

    if use_single_cond is True:
        d.update({
            "adaptive_zero_cond": LinearParams(ada_ln_zero.adaptive_zero_cond, use_bias=True),
        })

    return d

# ref alphafold3/model/diffusion/diffusion/model.py:class TriangleMultiplication
# class TriangleMultiplication(hk.Module):
#   """Triangle Multiplication."""

#   class Config(base_config.BaseConfig):
#     equation: Literal['ikc,jkc->ijc', 'kjc,kic->ijc']
#     use_glu_kernel: bool = True

#   def __init__(
#       self, config: Config, global_config: model_config.GlobalConfig, *, name
#   ):
#     super().__init__(name=name)
#     self.config = config
#     self.global_config = global_config

#   def __call__(self, act, mask):
#     """Applies Module.

#     Args:
#       act: The activation.
#       mask: The mask.

#     Returns:
#       Outputs, should have same shape/type as output_act
#     """
#     mask = mask[None, ...]
#     num_channels = act.shape[-1]
#     equation = {
#         'ikc,jkc->ijc': 'cik,cjk->cij',
#         'kjc,kic->ijc': 'ckj,cki->cij',
#     }[self.config.equation]

#     act = hm.LayerNorm(name='left_norm_input')(act)
#     input_act = act

#     if self.config.use_glu_kernel:
#       weights_projection, _ = hm.haiku_linear_get_params(
#           act, num_output=num_channels * 2, name='projection'
#       )
#       weights_gate, _ = hm.haiku_linear_get_params(
#           act,
#           num_output=num_channels * 2,
#           initializer=self.global_config.final_init,
#           name='gate',
#       )
#       weights_glu = jnp.stack([weights_gate, weights_projection], axis=1)

#       projection = gated_linear_unit.gated_linear_unit(
#           x=act,
#           weight=weights_glu,
#           activation=jax.nn.sigmoid,
#           implementation=None,
#       )
#       projection = jnp.transpose(projection, (2, 0, 1))
#       projection *= mask
#     else:
#       projection = hm.Linear(num_channels * 2, name='projection')(act)
#       projection = jnp.transpose(projection, (2, 0, 1))
#       projection *= mask

#       gate = hm.Linear(
#           num_channels * 2,
#           name='gate',
#           bias_init=1.0,
#           initializer=self.global_config.final_init,
#       )(act)
#       gate = jnp.transpose(gate, (2, 0, 1))
#       projection *= jax.nn.sigmoid(gate)

#     projection = projection.reshape(num_channels, 2, *projection.shape[1:])
#     a, b = jnp.split(projection, 2, axis=1)
#     a, b = jnp.squeeze(a, axis=1), jnp.squeeze(b, axis=1)
#     act = jnp.einsum(equation, a, b)
#     act = hm.LayerNorm(name='center_norm', axis=0, param_axis=0)(act)

#     act = jnp.transpose(act, (1, 2, 0))
#     act = hm.Linear(
#         num_channels,
#         initializer=self.global_config.final_init,
#         name='output_projection',
#     )(act)

#     gate_out = hm.Linear(
#         num_channels,
#         name='gating_linear',
#         bias_init=1.0,
#         initializer=self.global_config.final_init,
#     )(input_act)
#     act *= jax.nn.sigmoid(gate_out)

#     return act


def TriMulParams(tri_mul): return {
    "left_norm_input": LayerNormParams(tri_mul.left_norm_input),
    "projection": LinearParams(tri_mul.projection),
    "gate": LinearParams(tri_mul.gate),
    "center_norm": LayerNormParams(tri_mul.center_norm),
    "output_projection": LinearParams(tri_mul.output_projection),
    "gating_linear": LinearParams(tri_mul.gating_linear)
}

def OuterProductMeanParams(outer_product_mean): return {
    "layer_norm_input": LayerNormParams(outer_product_mean.layer_norm_input),
    "left_projection": LinearParams(outer_product_mean.left_projection),
    "right_projection": LinearParams(outer_product_mean.right_projection),
    "output_w": Param(outer_product_mean.output_w),
    "output_b": Param(outer_product_mean.output_b),
}


def TransitionParams(transition): return {
    "input_layer_norm": LayerNormParams(transition.input_layer_norm),
    "transition1": LinearParams(transition.transition1),
    "transition2": LinearParams(transition.transition2),
}


def GridSelfAttentionParams(pair_attention): return {
    "act_norm": LayerNormParams(pair_attention.act_norm),
    "pair_bias_projection": LinearParams(pair_attention.pair_bias_projection),
    "q_projection": LinearHMAParams(pair_attention.q_projection, already_transpose_weights=True),
    "k_projection": LinearHMAParams(pair_attention.k_projection, already_transpose_weights=True),
    "v_projection": LinearHMAParams(pair_attention.v_projection),
    "gating_query": LinearParams(pair_attention.gating_query, already_transpose_weights=True),
    "output_projection": LinearParams(pair_attention.output_projection),
}

def SelfAttentionParams(self_attention, use_single_cond=False):
    return {
        "q_projection": LinearHMAParams(self_attention.q_projection, use_bias=True),
        "k_projection": LinearHMAParams(self_attention.k_projection),
        "v_projection": LinearHMAParams(self_attention.v_projection),
        "gating_query": LinearParams(self_attention.gating_query),
        "transition2": LinearParams(self_attention.adaptive_zero_init.transition2),
        **AdaptiveLayerNormParams(self_attention.adaptive_layernorm, use_single_cond),
        **AdaLNZeroParams(self_attention.adaptive_zero_init, use_single_cond),
    }


def CrossAttentionParams(cross_attention): return {
    **cat_params(AdaptiveLayerNormParams(cross_attention.q_adaptive_layernorm, use_single_cond=True), "q"),
    **cat_params(AdaptiveLayerNormParams(cross_attention.k_adaptive_layernorm, use_single_cond=True), "k"),
    "q_projection": LinearHMAParams(cross_attention.q_projection, use_bias=True),
    "k_projection": LinearHMAParams(cross_attention.k_projection),
    "v_projection": LinearHMAParams(cross_attention.v_projection),
    "gating_query": LinearParams(cross_attention.gating_query),
    **AdaLNZeroParams(cross_attention.adaptive_zero_init, use_single_cond=True),
}


def MSAAttentionParams(msa_attention): return {
    "act_norm": LayerNormParams(msa_attention.act_norm),
    "pair_norm": LayerNormParams(msa_attention.pair_norm),
    "pair_logits": LinearParams(msa_attention.pair_logits),
    "v_projection": LinearHMAParams(msa_attention.v_projection),
    "gating_query": LinearParams(msa_attention.gating_query),
    "output_projection": LinearParams(msa_attention.output_projection),
}

def DiffusionTransitionParams(transition, use_single_cond=False):
    return {
        **AdaptiveLayerNormParams(transition.adaptive_layernorm, use_single_cond),
        "transition1": LinearParams(transition.transition1),
        **AdaLNZeroParams(transition.adaptive_zero_init, use_single_cond),
    }


def DiffusionTransformerParams(transformer):

    self_attention_params = stacked([SelfAttentionParams(
        l, use_single_cond=True) for l in transformer.self_attention])
    transistion_params = stacked([DiffusionTransitionParams(
        l, use_single_cond=True) for l in transformer.transition_block])

    return {
        "pair_input_layer_norm": LayerNormParams(transformer.pair_input_layer_norm, use_bias=False),
        "__layer_stack_with_per_layer/pair_logits_projection": stacked([LinearHMAParams(l) for l in transformer.pair_logits_projection]),
        **cat_params(self_attention_params, "__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformer"),
        **cat_params(transistion_params, "__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformerffw_"),
    }
    
def DiffusionCrossAttTransformerParams(transformer, prefix="diffusion_atom_transformer_encoder"):

    cross_attention_params = stacked([CrossAttentionParams(
        l) for l in transformer.cross_attention])
    transistion_params = stacked([DiffusionTransitionParams(
        l, use_single_cond=True) for l in transformer.transition_block])

    return {
        "pair_input_layer_norm": LayerNormParams(transformer.pair_input_layer_norm, use_bias=False),
        "pair_logits_projection": LinearHMAParams(transformer.pair_logits_projection),
        **cat_params(cross_attention_params, f"__layer_stack_with_per_layer/{prefix}"),
        **cat_params(transistion_params, f"__layer_stack_with_per_layer/{prefix}ffw_"),
    }

# ref:
# def _per_atom_conditioning(
#     config: AtomCrossAttEncoderConfig, batch: feat_batch.Batch, name: str
# ) -> tuple[jnp.ndarray, jnp.ndarray]:
#   """computes single and pair conditioning for all atoms in each token."""

#   c = config
#   # Compute per-atom single conditioning
#   # Shape (num_tokens, num_dense, channels)
#   act = hm.Linear(
#       c.per_atom_channels, precision='highest', name=f'{name}_embed_ref_pos'
#   )(batch.ref_structure.positions)
#   act += hm.Linear(c.per_atom_channels, name=f'{name}_embed_ref_mask')(
#       batch.ref_structure.mask.astype(jnp.float32)[:, :, None]
#   )
#   # Element is encoded as atomic number if the periodic table, so
#   # 128 should be fine.
#   act += hm.Linear(c.per_atom_channels, name=f'{name}_embed_ref_element')(
#       jax.nn.one_hot(batch.ref_structure.element, 128)
#   )
#   act += hm.Linear(c.per_atom_channels, name=f'{name}_embed_ref_charge')(
#       jnp.arcsinh(batch.ref_structure.charge)[:, :, None]
#   )
#   # Characters are encoded as ASCII code minus 32, so we need 64 classes,
#   # to encode all standard ASCII characters between 32 and 96.
#   atom_name_chars_1hot = jax.nn.one_hot(batch.ref_structure.atom_name_chars, 64)
#   num_token, num_dense, _ = act.shape
#   act += hm.Linear(c.per_atom_channels, name=f'{name}_embed_ref_atom_name')(
#       atom_name_chars_1hot.reshape(num_token, num_dense, -1)
#   )
#   act *= batch.ref_structure.mask.astype(jnp.float32)[:, :, None]

#   # Compute pair conditioning
#   # shape (num_tokens, num_dense, num_dense, channels)
#   # Embed single features
#   row_act = hm.Linear(
#       c.per_atom_pair_channels, name=f'{name}_single_to_pair_cond_row'
#   )(jax.nn.relu(act))
#   col_act = hm.Linear(
#       c.per_atom_pair_channels, name=f'{name}_single_to_pair_cond_col'
#   )(jax.nn.relu(act))
#   pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]
#   # Embed pairwise offsets
#   pair_act += hm.Linear(
#       c.per_atom_pair_channels,
#       precision='highest',
#       name=f'{name}_embed_pair_offsets',
#   )(
#       batch.ref_structure.positions[:, :, None, :]
#       - batch.ref_structure.positions[:, None, :, :]
#   )
#   # Embed pairwise inverse squared distances
#   sq_dists = jnp.sum(
#       jnp.square(
#           batch.ref_structure.positions[:, :, None, :]
#           - batch.ref_structure.positions[:, None, :, :]
#       ),
#       axis=-1,
#   )
#   pair_act += hm.Linear(
#       c.per_atom_pair_channels, name=f'{name}_embed_pair_distances'
#   )(1.0 / (1 + sq_dists[:, :, :, None]))

#   return act, pair_act

def AtomCrossAttEncoderParams(encoder,
                              with_token_atoms_act=False,
                              with_trunk_single_cond=False,
                              with_trunk_pair_cond=False,
                              prefix="evoformer_conditioning_atom_transformer_encoder"):
    d = {
        "embed_ref_pos": LinearParams(encoder.embed_ref_pos),
        "embed_ref_mask": LinearParams(encoder.embed_ref_mask),
        "embed_ref_element": LinearParams(encoder.embed_ref_element),
        "embed_ref_charge": LinearParams(encoder.embed_ref_charge),
        "embed_ref_atom_name": LinearParams(encoder.embed_ref_atom_name),
        "single_to_pair_cond_row": LinearParams(encoder.single_to_pair_cond_row),
        "single_to_pair_cond_col": LinearParams(encoder.single_to_pair_cond_col),
        "embed_pair_offsets": LinearParams(encoder.embed_pair_offsets),
        "embed_pair_distances": LinearParams(encoder.embed_pair_distances),
        "single_to_pair_cond_row_1": LinearParams(encoder.single_to_pair_cond_row_1),
        "single_to_pair_cond_col_1": LinearParams(encoder.single_to_pair_cond_col_1),
        "embed_pair_offsets_1": LinearParams(encoder.embed_pair_offsets_1),
        "embed_pair_distances_1": LinearParams(encoder.embed_pair_distances_1),
        "embed_pair_offsets_valid": LinearParams(encoder.embed_pair_offsets_valid),
        "pair_mlp_1": LinearParams(encoder.pair_mlp_1),
        "pair_mlp_2": LinearParams(encoder.pair_mlp_2),
        "pair_mlp_3": LinearParams(encoder.pair_mlp_3),
        "atom_transformer_encoder": DiffusionCrossAttTransformerParams(encoder.atom_transformer_encoder, prefix=prefix),
        "project_atom_features_for_aggr": LinearParams(encoder.project_atom_features_for_aggr),
    }

    if with_token_atoms_act is True:
        d.update({
            "atom_positions_to_features": LinearParams(encoder.atom_positions_to_features),
        })

    if with_trunk_single_cond is True:
        d.update({
            "lnorm_trunk_single_cond": LayerNormParams(encoder.lnorm_trunk_single_cond, use_bias=False),
            "embed_trunk_single_cond": LinearParams(encoder.embed_trunk_single_cond),
        })

    if with_trunk_pair_cond:
        d.update({
            "lnorm_trunk_pair_cond": LayerNormParams(encoder.lnorm_trunk_pair_cond, use_bias=False),
            "embed_trunk_pair_cond": LinearParams(encoder.embed_trunk_pair_cond),
        })

    return d


def AtomCrossAttDecoderParams(decoder): return {
    "project_token_features_for_broadcast": LinearParams(decoder.project_token_features_for_broadcast),
    "atom_transformer_decoder": DiffusionCrossAttTransformerParams(decoder.atom_transformer_decoder, prefix="diffusion_atom_transformer_decoder"),
    "atom_features_layer_norm": LayerNormParams(decoder.atom_features_layer_norm, use_bias=False),
    "atom_features_to_position_update": LinearParams(decoder.atom_features_to_position_update),
}


def TemplateEmbeddingParams(template_embedding):

    pairformer_params = stacked(
        [PairformerBlockParams(b, with_single=False) for b in template_embedding.single_template_embedding.template_embedding_iteration])

    return {
        "single_template_embedding/query_embedding_norm": LayerNormParams(template_embedding.single_template_embedding.query_embedding_norm),
        "single_template_embedding/template_pair_embedding_0": LinearParams(template_embedding.single_template_embedding.template_pair_embedding_0),
        "single_template_embedding/template_pair_embedding_1": LinearfromFlatParams(template_embedding.single_template_embedding.template_pair_embedding_1),
        "single_template_embedding/template_pair_embedding_2": LinearParams(template_embedding.single_template_embedding.template_pair_embedding_2),
        "single_template_embedding/template_pair_embedding_3": LinearParams(template_embedding.single_template_embedding.template_pair_embedding_3),
        "single_template_embedding/template_pair_embedding_4": LinearfromFlatParams(template_embedding.single_template_embedding.template_pair_embedding_4),
        "single_template_embedding/template_pair_embedding_5": LinearfromFlatParams(template_embedding.single_template_embedding.template_pair_embedding_5),
        "single_template_embedding/template_pair_embedding_6": LinearfromFlatParams(template_embedding.single_template_embedding.template_pair_embedding_6),
        "single_template_embedding/template_pair_embedding_7": LinearfromFlatParams(template_embedding.single_template_embedding.template_pair_embedding_7),
        "single_template_embedding/template_pair_embedding_8": LinearParams(template_embedding.single_template_embedding.template_pair_embedding_8),
        **cat_params(pairformer_params, f"single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/"),
        "single_template_embedding/output_layer_norm": LayerNormParams(template_embedding.single_template_embedding.output_layer_norm),
        "output_linear": LinearParams(template_embedding.output_linear),
    }


def PairformerBlockParams(b, with_single=False):
    d = {
        "triangle_multiplication_outgoing": TriMulParams(b.triangle_multiplication_outgoing),
        "triangle_multiplication_incoming": TriMulParams(b.triangle_multiplication_incoming),
        "pair_attention1": GridSelfAttentionParams(b.pair_attention1),
        "pair_attention2": GridSelfAttentionParams(b.pair_attention2),
        "pair_transition": TransitionParams(b.pair_transition),
    }

    if with_single is True:
        d.update({
            "single_pair_logits_norm": LayerNormParams(b.single_pair_logits_norm),
            "single_pair_logits_projection": LinearParams(b.single_pair_logits_projection),
            **cat_params(SelfAttentionParams(b.single_attention_), "single_attention_"),
            "single_transition": TransitionParams(b.single_transition),
        })

    return d


def EvoformerBlockParams(b): return {
    "outer_product_mean": OuterProductMeanParams(b.outer_product_mean),
    "msa_attention1": MSAAttentionParams(b.msa_attention1),
    "msa_transition": TransitionParams(b.msa_transition),
    "triangle_multiplication_outgoing": TriMulParams(b.triangle_multiplication_outgoing),
    "triangle_multiplication_incoming": TriMulParams(b.triangle_multiplication_incoming),
    "pair_attention1": GridSelfAttentionParams(b.pair_attention1),
    "pair_attention2": GridSelfAttentionParams(b.pair_attention2),
    "pair_transition": TransitionParams(b.pair_transition),
}


def DiffusionHeadParams(head):
    return {
        "pair_cond_initial_norm": LayerNormParams(head.pair_cond_initial_norm, use_bias=False),
        "pair_cond_initial_projection": LinearParams(head.pair_cond_initial_projection),
        **cat_params(DiffusionTransitionParams(head.pair_transition_0), "pair_transition_0ffw_"),
        **cat_params(DiffusionTransitionParams(head.pair_transition_1), "pair_transition_1ffw_"),
        "single_cond_initial_norm": LayerNormParams(head.single_cond_initial_norm, use_bias=False),
        "single_cond_initial_projection": LinearParams(head.single_cond_initial_projection),
        "noise_embedding_initial_norm": LayerNormParams(head.noise_embedding_initial_norm, use_bias=False),
        "noise_embedding_initial_projection": LinearParams(head.noise_embedding_initial_projection),
        **cat_params(DiffusionTransitionParams(head.single_transition_0), "single_transition_0ffw_"),
        **cat_params(DiffusionTransitionParams(head.single_transition_1), "single_transition_1ffw_"),
        **cat_params(AtomCrossAttEncoderParams(head.atom_cross_att_encoder,
                                               with_token_atoms_act=True,
                                               with_trunk_pair_cond=True,
                                               with_trunk_single_cond=True,
                                               prefix="diffusion_atom_transformer_encoder"), "diffusion_"),
        "single_cond_embedding_norm": LayerNormParams(head.single_cond_embedding_norm, use_bias=False),
        "single_cond_embedding_projection": LinearParams(head.single_cond_embedding_projection),
        "transformer": DiffusionTransformerParams(head.transformer),
        "output_norm": LayerNormParams(head.output_norm, use_bias=False),
        **cat_params(AtomCrossAttDecoderParams(head.atom_cross_att_decoder), "diffusion_")
    }


def ConfidenceHeadParams(head):

    pairformer_blocks_params = stacked(
        [PairformerBlockParams(b, with_single=True) for b in head.confidence_pairformer])

    return {
        "~_embed_features/left_target_feat_project": LinearParams(head.left_target_feat_project),
        "~_embed_features/right_target_feat_project": LinearParams(head.right_target_feat_project),
        "~_embed_features/distogram_feat_project": LinearParams(head.distogram_feat_project),
        "__layer_stack_no_per_layer/confidence_pairformer": pairformer_blocks_params,
        "logits_ln": LayerNormParams(head.logits_ln),
        "left_half_distance_logits": LinearParams(head.left_half_distance_logits),
        "pae_logits_ln": LayerNormParams(head.pae_logits_ln),
        "pae_logits": LinearParams(head.pae_logits),
        "plddt_logits_ln": LayerNormParams(head.plddt_logits_ln),
        "plddt_logits": LinearHMAParams(head.plddt_logits),
        "experimentally_resolved_ln": LayerNormParams(head.experimentally_resolved_ln),
        "experimentally_resolved_logits": LinearHMAParams(head.experimentally_resolved_logits),
    }

# ref:
# class Evoformer(hk.Module):
#   """Creates 'single' and 'pair' embeddings."""

#   class PairformerConfig(modules.PairFormerIteration.Config):  # pytype: disable=invalid-function-definition
#     block_remat: bool = False
#     remat_block_size: int = 8

#   class Config(base_config.BaseConfig):
#     """Configuration for Evoformer."""

#     max_relative_chain: int = 2
#     msa_channel: int = 64
#     seq_channel: int = 384
#     max_relative_idx: int = 32
#     num_msa: int = 1024
#     pair_channel: int = 128
#     pairformer: 'Evoformer.PairformerConfig' = base_config.autocreate(
#         single_transition=base_config.autocreate(),
#         single_attention=base_config.autocreate(),
#         num_layer=48,
#     )
#     per_atom_conditioning: atom_cross_attention.AtomCrossAttEncoderConfig = (
#         base_config.autocreate(
#             per_token_channels=384,
#             per_atom_channels=128,
#             atom_transformer=base_config.autocreate(
#                 num_intermediate_factor=2,
#                 num_blocks=3,
#             ),
#             per_atom_pair_channels=16,
#         )
#     )
#     template: template_modules.TemplateEmbedding.Config = (
#         base_config.autocreate()
#     )
#     msa_stack: modules.EvoformerIteration.Config = base_config.autocreate()

#   def __init__(
#       self,
#       config: Config,
#       global_config: model_config.GlobalConfig,
#       name='evoformer',
#   ):
#     super().__init__(name=name)
#     self.config = config
#     self.global_config = global_config

#   def _relative_encoding(
#       self, batch: feat_batch.Batch, pair_activations: jnp.ndarray
#   ) -> jnp.ndarray:
#     """Add relative position encodings."""
#     rel_feat = featurization.create_relative_encoding(
#         batch.token_features,
#         self.config.max_relative_idx,
#         self.config.max_relative_chain,
#     )
#     rel_feat = rel_feat.astype(pair_activations.dtype)

#     pair_activations += hm.Linear(
#         self.config.pair_channel, name='position_activations'
#     )(rel_feat)
#     return pair_activations

#   @hk.transparent
#   def _seq_pair_embedding(
#       self,
#       token_features: features.TokenFeatures,
#       target_feat: jnp.ndarray,
#   ) -> tuple[jnp.ndarray, jnp.ndarray]:
#     """Generated Pair embedding from sequence."""
#     left_single = hm.Linear(self.config.pair_channel, name='left_single')(
#         target_feat
#     )[:, None]
#     right_single = hm.Linear(self.config.pair_channel, name='right_single')(
#         target_feat
#     )[None]
#     dtype = left_single.dtype
#     pair_activations = left_single + right_single
#     num_residues = pair_activations.shape[0]
#     assert pair_activations.shape == (
#         num_residues,
#         num_residues,
#         self.config.pair_channel,
#     )
#     mask = token_features.mask
#     pair_mask = (mask[:, None] * mask[None, :]).astype(dtype)
#     assert pair_mask.shape == (num_residues, num_residues)
#     return pair_activations, pair_mask  # pytype: disable=bad-return-type  # jax-ndarray

#   @hk.transparent
#   def _embed_bonds(
#       self,
#       batch: feat_batch.Batch,
#       pair_activations: jnp.ndarray,
#   ) -> jnp.ndarray:
#     """Embeds bond features and merges into pair activations."""
#     # Construct contact matrix.
#     num_tokens = batch.token_features.token_index.shape[0]
#     contact_matrix = jnp.zeros((num_tokens, num_tokens))

#     tokens_to_polymer_ligand_bonds = (
#         batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
#     )
#     gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
#     gather_mask_polymer_ligand = (
#         tokens_to_polymer_ligand_bonds.gather_mask.prod(axis=1).astype(
#             gather_idxs_polymer_ligand.dtype
#         )[:, None]
#     )
#     # If valid mask then it will be all 1's, so idxs should be unchanged.
#     gather_idxs_polymer_ligand = (
#         gather_idxs_polymer_ligand * gather_mask_polymer_ligand
#     )

#     tokens_to_ligand_ligand_bonds = (
#         batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
#     )
#     gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
#     gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
#         axis=1
#     ).astype(gather_idxs_ligand_ligand.dtype)[:, None]
#     gather_idxs_ligand_ligand = (
#         gather_idxs_ligand_ligand * gather_mask_ligand_ligand
#     )

#     gather_idxs = jnp.concatenate(
#         [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
#     )
#     contact_matrix = contact_matrix.at[
#         gather_idxs[:, 0], gather_idxs[:, 1]
#     ].set(1.0)

#     # Because all the padded index's are 0's.
#     contact_matrix = contact_matrix.at[0, 0].set(0.0)

#     bonds_act = hm.Linear(self.config.pair_channel, name='bond_embedding')(
#         contact_matrix[:, :, None].astype(pair_activations.dtype)
#     )
#     return pair_activations + bonds_act

#   @hk.transparent
#   def _embed_template_pair(
#       self,
#       batch: feat_batch.Batch,
#       pair_activations: jnp.ndarray,
#       pair_mask: jnp.ndarray,
#       key: jnp.ndarray,
#   ) -> tuple[jnp.ndarray, jnp.ndarray]:
#     """Embeds Templates and merges into pair activations."""
#     dtype = pair_activations.dtype
#     key, subkey = jax.random.split(key)
#     template_module = template_modules.TemplateEmbedding(
#         self.config.template, self.global_config
#     )
#     templates = batch.templates
#     asym_id = batch.token_features.asym_id
#     # Construct a mask such that only intra-chain template features are
#     # computed, since all templates are for each chain individually.
#     multichain_mask = (asym_id[:, None] == asym_id[None, :]).astype(dtype)

#     template_fn = functools.partial(template_module, key=subkey)
#     template_act = template_fn(
#         query_embedding=pair_activations,
#         templates=templates,
#         multichain_mask_2d=multichain_mask,
#         padding_mask_2d=pair_mask,
#     )
#     return pair_activations + template_act, key

#   @hk.transparent
#   def _embed_process_msa(
#       self,
#       msa_batch: features.MSA,
#       pair_activations: jnp.ndarray,
#       pair_mask: jnp.ndarray,
#       key: jnp.ndarray,
#       target_feat: jnp.ndarray,
#   ) -> tuple[jnp.ndarray, jnp.ndarray]:
#     """Processes MSA and returns updated pair activations."""
#     dtype = pair_activations.dtype
#     msa_batch, key = featurization.shuffle_msa(key, msa_batch)
#     msa_batch = featurization.truncate_msa_batch(msa_batch, self.config.num_msa)
#     msa_feat = featurization.create_msa_feat(msa_batch).astype(dtype)

#     msa_activations = hm.Linear(
#         self.config.msa_channel, name='msa_activations'
#     )(msa_feat)

#     msa_activations += hm.Linear(
#         self.config.msa_channel, name='extra_msa_target_feat'
#     )(target_feat)[None]
#     msa_mask = msa_batch.mask.astype(dtype)

#     # Evoformer MSA stack.
#     evoformer_input = {'msa': msa_activations, 'pair': pair_activations}
#     masks = {'msa': msa_mask, 'pair': pair_mask}

#     def evoformer_fn(x):
#       return modules.EvoformerIteration(
#           self.config.msa_stack, self.global_config, name='msa_stack'
#       )(
#           activations=x,
#           masks=masks,
#       )

#     evoformer_stack = hk.experimental.layer_stack(
#         self.config.msa_stack.num_layer
#     )(evoformer_fn)

#     evoformer_output = evoformer_stack(evoformer_input)

#     return evoformer_output['pair'], key

#   def __call__(
#       self,
#       batch: feat_batch.Batch,
#       prev: dict[str, jnp.ndarray],
#       target_feat: jnp.ndarray,
#       key: jnp.ndarray,
#   ) -> dict[str, jnp.ndarray]:

#     assert self.global_config.bfloat16 in {'all', 'none'}

#     num_residues = target_feat.shape[0]
#     assert batch.token_features.aatype.shape == (num_residues,)

#     dtype = (
#         jnp.bfloat16 if self.global_config.bfloat16 == 'all' else jnp.float32
#     )

#     with utils.bfloat16_context():
#       pair_activations, pair_mask = self._seq_pair_embedding(
#           batch.token_features, target_feat
#       )

#       pair_activations += hm.Linear(
#           pair_activations.shape[-1],
#           name='prev_embedding',
#           initializer=self.global_config.final_init,
#       )(
#           hm.LayerNorm(name='prev_embedding_layer_norm')(
#               prev['pair'].astype(pair_activations.dtype)
#           )
#       )

#       pair_activations = self._relative_encoding(batch, pair_activations)

#       pair_activations = self._embed_bonds(
#           batch=batch, pair_activations=pair_activations
#       )

#       pair_activations, key = self._embed_template_pair(
#           batch=batch,
#           pair_activations=pair_activations,
#           pair_mask=pair_mask,
#           key=key,
#       )
#       pair_activations, key = self._embed_process_msa(
#           msa_batch=batch.msa,
#           pair_activations=pair_activations,
#           pair_mask=pair_mask,
#           key=key,
#           target_feat=target_feat,
#       )
#       del key  # Unused after this point.

#       single_activations = hm.Linear(
#           self.config.seq_channel, name='single_activations'
#       )(target_feat)

#       single_activations += hm.Linear(
#           single_activations.shape[-1],
#           name='prev_single_embedding',
#           initializer=self.global_config.final_init,
#       )(
#           hm.LayerNorm(name='prev_single_embedding_layer_norm')(
#               prev['single'].astype(single_activations.dtype)
#           )
#       )

#       def pairformer_fn(x):
#         pairformer_iteration = modules.PairFormerIteration(
#             self.config.pairformer,
#             self.global_config,
#             with_single=True,
#             name='trunk_pairformer',
#         )
#         pair_act, single_act = x
#         return pairformer_iteration(
#             act=pair_act,
#             single_act=single_act,
#             pair_mask=pair_mask,
#             seq_mask=batch.token_features.mask.astype(dtype),
#         )

#       pairformer_stack = hk.experimental.layer_stack(
#           self.config.pairformer.num_layer
#       )(pairformer_fn)

#       pair_activations, single_activations = pairformer_stack(
#           (pair_activations, single_activations)
#       )

#       assert pair_activations.shape == (
#           num_residues,
#           num_residues,
#           self.config.pair_channel,
#       )
#       assert single_activations.shape == (num_residues, self.config.seq_channel)
#       assert len(target_feat.shape) == 2
#       assert target_feat.shape[0] == num_residues
#       output = {
#           'single': single_activations,
#           'pair': pair_activations,
#           'target_feat': target_feat,
#       }

#     return output

def EvoformerParams(evoformer):

    msa_stack_params = stacked(
        [EvoformerBlockParams(b) for b in evoformer.msa_stack])

    trunk_pairformer_params = stacked(
        [PairformerBlockParams(b, with_single=True) for b in evoformer.trunk_pairformer])

    return {
        "left_single": LinearParams(evoformer.left_single),
        "right_single": LinearParams(evoformer.right_single),
        "prev_embedding_layer_norm": LayerNormParams(evoformer.prev_embedding_layer_norm),
        "prev_embedding": LinearParams(evoformer.prev_embedding),
        "~_relative_encoding/position_activations": LinearParams(evoformer.position_activations),
        "bond_embedding": LinearParams(evoformer.bond_embedding),
        "template_embedding": TemplateEmbeddingParams(evoformer.template_embedding),
        "msa_activations": LinearParams(evoformer.msa_activations),
        "extra_msa_target_feat": LinearParams(evoformer.extra_msa_target_feat),
        **cat_params(msa_stack_params, "__layer_stack_no_per_layer/msa_stack/"),
        "single_activations": LinearParams(evoformer.single_activations),
        "prev_single_embedding_layer_norm": LayerNormParams(evoformer.prev_single_embedding_layer_norm),
        "prev_single_embedding": LinearParams(evoformer.prev_single_embedding),
        **cat_params(trunk_pairformer_params, "__layer_stack_no_per_layer_1/trunk_pairformer/"),
    }


def get_translation_dict(model):
    translations = {
        **cat_params(AtomCrossAttEncoderParams(model.evoformer_conditioning), "evoformer_conditioning_"),
        "evoformer": EvoformerParams(model.evoformer),
        "~/diffusion_head": DiffusionHeadParams(model.diffusion_head),
        "distogram_head/half_logits": LinearParams(model.distogram_head.half_logits),
        "confidence_head": ConfidenceHeadParams(model.confidence_head),
    }

    return translations

# 最激动人心的一集
def import_jax_weights_(model, model_path: pathlib.Path):
    params = get_model_haiku_params_to_torch(model_path / "af3.bin")

    translations = get_translation_dict(model)

    flat = _process_translations_dict(translations, _key_prefix="diffuser/")

    assign(flat, params)

    model.__identifier__ = params['__meta__/__identifier__']

    fourier_embeddings = model.diffusion_head.fourier_embeddings
# FIXME： workaround: numpy生成的傅立叶嵌入和jnp实现行为不一致，最终导致非物理结构。于是先用原版jax生成并保存为npy供torchfold3使用
    fourier_embeddings_weight = np.load(
        open("fourier_embeddings/weight.npy", "rb"))
    fourier_embeddings.register_buffer(
        "weight", torch.from_numpy(fourier_embeddings_weight))
    fourier_embeddings_bias = np.load(
        open("fourier_embeddings/bias.npy", "rb"))
    fourier_embeddings.register_buffer(
        "bias", torch.from_numpy(fourier_embeddings_bias))
