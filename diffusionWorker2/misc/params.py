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
        "__layer_stack_with_per_layer/pair_logits_projection": stacked(
            [LinearHMAParams(l) for l in transformer.pair_logits_projection]),
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
        **cat_params(pairformer_params, f"single_template_embedding/__layer_stack_no_per_layer/template_embedding_iteration/"),
    }


def PairformerBlockParams(b, with_single=False):
    d = {

        "pair_attention1": GridSelfAttentionParams(b.pair_attention1),
        "pair_attention2": GridSelfAttentionParams(b.pair_attention2),
    }

    return d





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
def get_translation_dict(model):
    translations = {
        "~/diffusion_head": DiffusionHeadParams(model.diffusion_head),
    }

    return translations
def get_diffusion_head_translation_dict(model):
    translations = {
        "~/diffusion_head": DiffusionHeadParams(model),
    }

    return translations

def import_diffusion_head_params(model, model_path: pathlib.Path):
    params = get_model_haiku_params_to_torch(model_path / "af3.bin")
    translations = get_diffusion_head_translation_dict(model)
    flat = _process_translations_dict(translations, _key_prefix="diffuser/")
    assign(flat, params)
    model.__identifier__ = params['__meta__/__identifier__']

    fourier_embeddings = model.fourier_embeddings
    # FIXME： workaround: numpy生成的傅立叶嵌入和jnp实现行为不一致，最终导致非物理结构。于是先用原版jax生成并保存为npy供torchfold3使用
    fourier_embeddings_weight = np.load(
        open("fourier_embeddings/weight.npy", "rb"))
    fourier_embeddings.weight.data = torch.from_numpy(fourier_embeddings_weight)
    fourier_embeddings_bias = np.load(
        open("fourier_embeddings/bias.npy", "rb"))
    fourier_embeddings.bias.data = torch.from_numpy(fourier_embeddings_bias)

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
    fourier_embeddings.weight.data = torch.from_numpy(fourier_embeddings_weight)
    fourier_embeddings_bias = np.load(
        open("fourier_embeddings/bias.npy", "rb"))
    fourier_embeddings.bias.data = torch.from_numpy(fourier_embeddings_bias)
    # fourier_embeddings_weight = np.load(
    #     open("fourier_embeddings/weight.npy", "rb"))
    # fourier_embeddings.register_buffer(
    #     "weight", torch.from_numpy(fourier_embeddings_weight))
    # fourier_embeddings_bias = np.load(
    #     open("fourier_embeddings/bias.npy", "rb"))
    # fourier_embeddings.register_buffer(
    #     "bias", torch.from_numpy(fourier_embeddings_bias))

