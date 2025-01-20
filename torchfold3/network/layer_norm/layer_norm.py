# ref:
# class LayerNorm(hk.LayerNorm):
#   """LayerNorm module.

#   Equivalent to hk.LayerNorm but with an extra 'upcast' option that casts
#   (b)float16 inputs to float32 before computing the layer norm, and then casts
#   the output back to the input type.

#   The learnable parameter shapes are also different from Haiku: they are always
#   vectors rather than possibly higher-rank tensors. This makes it easier
#   to change the layout whilst keep the model weight-compatible.
#   """

#   def __init__(
#       self,
#       *,
#       axis: int = -1,
#       create_scale: bool = True,
#       create_offset: bool = True,
#       eps: float = 1e-5,
#       scale_init: hk.initializers.Initializer | None = None,
#       offset_init: hk.initializers.Initializer | None = None,
#       use_fast_variance: bool = True,
#       name: str,
#       param_axis: int | None = None,
#       upcast: bool = True,
#   ):
#     super().__init__(
#         axis=axis,
#         create_scale=False,
#         create_offset=False,
#         eps=eps,
#         scale_init=None,
#         offset_init=None,
#         use_fast_variance=use_fast_variance,
#         name=name,
#         param_axis=param_axis,
#     )
#     self.upcast = upcast
#     self._temp_create_scale = create_scale
#     self._temp_create_offset = create_offset

#   def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#     dtype = x.dtype
#     is_16bit = x.dtype in [jnp.bfloat16, jnp.float16]
#     if self.upcast and is_16bit:
#       x = x.astype(jnp.float32)

#     param_axis = self.param_axis[0] if self.param_axis else -1
#     param_shape = (x.shape[param_axis],)

#     param_broadcast_shape = [1] * x.ndim
#     param_broadcast_shape[param_axis] = x.shape[param_axis]
#     scale = None
#     offset = None
#     if self._temp_create_scale:
#       scale = hk.get_parameter(
#           'scale', param_shape, x.dtype, init=self.scale_init
#       )
#       scale = scale.reshape(param_broadcast_shape)

#     if self._temp_create_offset:
#       offset = hk.get_parameter(
#           'offset', param_shape, x.dtype, init=self.offset_init
#       )
#       offset = offset.reshape(param_broadcast_shape)

#     out = super().__call__(x, scale=scale, offset=offset)

#     if self.upcast and is_16bit:
#       out = out.astype(dtype)

#     return out

import numbers
from typing import List, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Size
import os,sys

sys.path.append(os.path.dirname(__file__))

_shape_t = Union[int, List[int], Size]

from torchfold3.config import *

if _CUDA_LAYER_NORM_OPT:
    from torchfold3.network.layer_norm.torch_ext_compile import compile

    current_dir = os.path.dirname(__file__)
    fast_layer_norm_cuda = compile(
        name="fast_layer_norm_cuda",
        sources=[
            os.path.join(f"{current_dir}/kernel", file)
            for file in ["layer_norm_cuda.cpp", "layer_norm_cuda_kernel.cu"]
        ],
        extra_include_paths=[f"{current_dir}/kernel"],
        build_directory=current_dir,
    )

class FusedLayerNormAffineFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        d = input.dtype
        if d is torch.bfloat16:
            with torch.amp.autocast('cuda',enabled=False):
                ctx.normalized_shape = normalized_shape
                ctx.eps = eps
                input_ = input.contiguous()
                # weight_ = weight.contiguous().to(dtype=d)
                if weight is None:
                    weight_ = torch.ones(input.size()[-len(normalized_shape):], dtype=d, device=input.device)
                else:
                    weight_ = weight.contiguous().to(dtype=d)
                if bias is not None:
                    bias_ = bias.contiguous().to(dtype=d)
                else:
                    bias_ = torch.zeros(input.size()[-len(normalized_shape):], dtype=d, device=input.device)
                output, mean, invvar = fast_layer_norm_cuda.forward_affine(
                    input_, ctx.normalized_shape, weight_, bias_, ctx.eps
                )
                ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        else:
            ctx.normalized_shape = normalized_shape
            ctx.eps = eps
            input_ = input.contiguous()
            # weight_ = weight.contiguous()
            if weight is None:
                weight_ = torch.ones(input.size()[-len(normalized_shape):], dtype=d, device=input.device)
            else:
                weight_ = weight.contiguous().to(dtype=d)
            # bias_ = bias.contiguous()
            if bias is not None:
                bias_ = bias.contiguous().to(dtype=d)
            else:
                bias_ = torch.zeros(input.size()[-len(normalized_shape):], dtype=d, device=input.device)
            output, mean, invvar = fast_layer_norm_cuda.forward_affine(
                input_, ctx.normalized_shape, weight_, bias_, ctx.eps
            )
            ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        d = grad_output.dtype
        if d is torch.bfloat16:
            with torch.amp.autocast('cuda',enabled=False):
                input_, weight_, bias_, mean, invvar = ctx.saved_tensors
                grad_input = grad_weight = grad_bias = None
                grad_input, grad_weight, grad_bias = (
                    fast_layer_norm_cuda.backward_affine(
                        grad_output.contiguous(),
                        mean,
                        invvar,
                        input_,
                        ctx.normalized_shape,
                        weight_.to(dtype=d),
                        bias_.to(dtype=d),
                        ctx.eps,
                    )
                )
        else:
            input_, weight_, bias_, mean, invvar = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None
            grad_input, grad_weight, grad_bias = (
                fast_layer_norm_cuda.backward_affine(
                    grad_output.contiguous(),
                    mean,
                    invvar,
                    input_,
                    ctx.normalized_shape,
                    weight_,
                    bias_,
                    ctx.eps,
                )
            )

        return grad_input, grad_weight, grad_bias, None, None

class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(
            normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from torchfold3.config import _CUDA_LAYER_NORM_OPT
        if input.device.type == "cpu":
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:
            if _CUDA_LAYER_NORM_OPT:
                return self.kernel_forward(input)
            else:
                return F.layer_norm(
                    input, self.normalized_shape, self.weight, self.bias, self.eps
                )
    
    def kernel_forward(self, input):
        return FusedLayerNormAffineFunction.apply(input, self.weight, self.bias, self.normalized_shape, self.eps)

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )