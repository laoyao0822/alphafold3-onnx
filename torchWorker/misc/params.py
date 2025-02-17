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


def get_model_haiku_params_to_torch(model_dir: pathlib.Path):
    if not os.path.exists(model_dir):
        raise Exception(
            f"Given checkpoint path not exist [{model_dir}]")
    print(f"Loading from {model_dir}")
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




def GridSelfAttentionParams(pair_attention): return {
    "pair_bias_projection": LinearParams(pair_attention.pair_bias_projection),
    "q_projection": LinearHMAParams(pair_attention.q_projection, already_transpose_weights=True),
    "k_projection": LinearHMAParams(pair_attention.k_projection, already_transpose_weights=True),
    "v_projection": LinearHMAParams(pair_attention.v_projection),
    "gating_query": LinearParams(pair_attention.gating_query, already_transpose_weights=True),
    "output_projection": LinearParams(pair_attention.output_projection),
}


def TemplateEmbeddingParams(template_embedding):

    pairformer_params = stacked(
        [PairformerBlockParams(b, with_single=False) for b in template_embedding.single_template_embedding.template_embedding_iteration])

    return {
        **cat_params(pairformer_params, f"single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/"),
    }


def PairformerBlockParams(b, with_single=False):
    d = {

        "pair_attention1": GridSelfAttentionParams(b.pair_attention1),
        "pair_attention2": GridSelfAttentionParams(b.pair_attention2),
    }

    return d


def EvoformerBlockParams(b): return {

    "pair_attention1": GridSelfAttentionParams(b.pair_attention1),
    "pair_attention2": GridSelfAttentionParams(b.pair_attention2),
}


def ConfidenceHeadParams(head):

    pairformer_blocks_params = stacked(
        [PairformerBlockParams(b, with_single=True) for b in head.confidence_pairformer])
    return {

        "__layer_stack_no_per_layer/confidence_pairformer": pairformer_blocks_params,

    }



def EvoformerParams(evoformer):

    msa_stack_params = stacked(
        [EvoformerBlockParams(b) for b in evoformer.msa_stack])

    trunk_pairformer_params = stacked(
        [PairformerBlockParams(b, with_single=True) for b in evoformer.trunk_pairformer])

    return {

        "template_embedding": TemplateEmbeddingParams(evoformer.template_embedding),
        **cat_params(msa_stack_params, "__layer_stack_no_per_layer/msa_stack/"),
        **cat_params(trunk_pairformer_params, "__layer_stack_no_per_layer_1/trunk_pairformer/"),
    }



def get_translation_dict(model):
    translations = {

        "evoformer": EvoformerParams(model.evoformer),
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

