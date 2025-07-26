# Copyright (c) Meta Platforms, Inc. and affiliates.

# This lingua codebase is adapted from the original implementation in:
# https://github.com/facebookresearch/lingua/
# Released under the BSD 3-Clause License by:
# Mathurin Videau*, Badr Youbi Idrissi*, Daniel Haziza, Luca Wehrstedt, Jade Copet, Olivier Teytaud, and David Lopez-Paz.
#
# This canon_helper.py file is newly introduced by Zeyuan Allen-Zhu, which is under the Apache 2.0 license, as stated in the root LICENSE file.
#

from typing import Any, Dict, List, Optional, Tuple
import torch

import warnings
from typing import Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformers.activations import ACT2FN

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None

import torch._dynamo

@torch._dynamo.disable
def causal_conv1d_fn_safe(*args, **kwargs):
    return causal_conv1d_fn(*args, **kwargs)

_CANON_LAYER_CACHE = {}

def make_canon_layer(hidden_size: int,
            kernel_size: int,
            bias: bool = False,
            activation: Optional[str] = 'silu',
            use_fast_conv1d: Optional[bool] = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
        ) -> nn.Module:
    
    key = (hidden_size, kernel_size, bias, activation, use_fast_conv1d)

    if key not in _CANON_LAYER_CACHE:
        ## This is an exact copy of `fla.modules.ShortConvolution` with no modification
        ## The purpose is to make sure you don't need to install fla-org, which is not a stable package yet.
        class ShortConvolution(nn.Conv1d):
            """
            Simple wrapper around `nn.Conv1d` that accepts dimension last.
            """

            def __init__(
                self
            ):
                super().__init__(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    groups=hidden_size,
                    bias=bias,
                    padding=kernel_size - 1,
                    device=device,
                    dtype=dtype,
                )

                self.hidden_size = hidden_size
                self.activation = None
                if activation is not None:
                    assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
                    self.activation = activation

                if causal_conv1d_fn is None:
                    if use_fast_conv1d:
                        raise RuntimeError(
                            "Please either install `causal-conv1d>=1.4.0` to enable fast causal short convolution CUDA kernel "
                            "or set `use_fast_conv1d` to False"
                        )
                    else:
                        warnings.warn(
                            "The naive Pytorch verison is very slow in practice, "
                            "please run `pip install causal-conv1d>=1.4.0` to install fast causal short convolution CUDA kernel",
                            category=ImportWarning
                        )
                self.use_fast_conv1d = use_fast_conv1d

            def __repr__(self):  # THIS helps TorchDynamo avoid collisions
                return f"CanonLayerCustom(hidden_size={self.hidden_size})"
                
            def extra_repr(self):
                s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
                    ', stride={stride}')
                if self.padding != (0,) * len(self.padding):
                    s += ', padding={padding}'
                if self.dilation != (1,) * len(self.dilation):
                    s += ', dilation={dilation}'
                if self.output_padding != (0,) * len(self.output_padding):
                    s += ', output_padding={output_padding}'
                if self.groups != 1:
                    s += ', groups={groups}'
                if self.bias is None:
                    s += ', bias=False'
                if self.padding_mode != 'zeros':
                    s += ', padding_mode={padding_mode}'
                if self.activation is not None:
                    s += ', activation={activation}'
                if not self.use_fast_conv1d:
                    s += ', use_fast_conv1d={use_fast_conv1d}'
                return s.format(**self.__dict__)

            def forward(
                self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cache: Optional[torch.Tensor] = None,
                output_final_state: bool = False,
                seq_idx: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                """
                Args:
                    x (`torch.Tensor`):
                        Tensor of shape `[batch_size, seq_len, hidden_size]`
                    mask (`Optional[torch.Tensor]`):
                        Attention mask dealing with padded positions.
                    cache (`Optional[torch.Tensor]`):
                        Previous cache tensor of shape `[batch_size, hidden_size, kernel_size]`.
                        If provided, the cache is updated **inplace**.
                    output_final_state (Optional[bool]):
                        Whether to output the final state of shape `[batch_size, hidden_size, kernel_size]`. Default: `False`.
                    seq_idx (Optional[torch.Tensor]):
                        Sequence index for each token. Used for varlen. Default: `None`.
                        Shape: [batch_size, seq_len]
                        Suppose a batch consists of two sequences with lengths 3 and 4, seq_idx=[0, 0, 0, 1, 1, 1, 1] for this batch.
                Returns:
                    Tensor of shape `[batch_size, seq_len, hidden_size]`.
                """

                batch_size, _, hidden_size = x.shape
                if mask is not None:
                    x = x.mul_(mask.unsqueeze(-1))
                if output_final_state and cache is None:
                    cache = x.new_zeros(batch_size, hidden_size, self.kernel_size[0])
                if cache is not None and x.shape[1] == 1:
                    return self.step(x, cache)
                x = rearrange(x, "b t d -> b d t")
                # Update state (B D W)
                if cache is not None:
                    cache.copy_(F.pad(x, (self.kernel_size[0] - x.shape[-1], 0)))
                if self.use_fast_conv1d:
                    x = causal_conv1d_fn_safe(
                        x=x,
                        weight=rearrange(self.weight, "d 1 w -> d w"),
                        bias=self.bias,
                        activation=self.activation,
                        seq_idx=seq_idx,
                    )
                else:
                    x = self._conv_forward(x, self.weight, self.bias)[..., :x.shape[-1]]
                    if self.activation is not None:
                        x = ACT2FN[self.activation](x)  # Note I'm using huggingface's ACT2FN here, not fla-org's original one, so that you don't need to install fla-org
                return rearrange(x, "b d t -> b t d"), cache

            def step(
                self,
                x: torch.Tensor,
                cache: torch.Tensor
            ):
                assert x.shape[1] == 1, "Only support decoding with 1 token at a time for now"

                x = x.squeeze(1)
                if self.use_fast_conv1d:
                    x = causal_conv1d_update(
                        x=x,
                        conv_state=cache,
                        weight=rearrange(self.weight, "d 1 w -> d w"),
                        bias=self.bias,
                        activation=self.activation,
                    )
                else:
                    dtype = x.dtype
                    cache.copy_(torch.roll(cache, shifts=-1, dims=-1))
                    cache[:, :, -1] = x
                    x = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
                    if self.bias is not None:
                        x = x + self.bias
                    if self.activation is not None:
                        x = ACT2FN[self.activation](x).to(dtype=dtype)
                return x.unsqueeze(1), cache

            @property
            def state_size(self) -> int:
                return self.hidden_size * self.kernel_size

        ShortConvolution.__name__ = f"CanonLayer_{hidden_size}_{kernel_size}_{activation}_{bias}_{use_fast_conv1d}"

        _CANON_LAYER_CACHE[key] = ShortConvolution

    return _CANON_LAYER_CACHE[key]()


def create_canon(dim, config):
    canon = make_canon_layer(dim, bias=config.canon_bias, kernel_size=config.canon_kernel, activation = 'silu' if config.canon_activation else None, use_fast_conv1d=config.canon_kernel in [2,3,4])
    #canon = make_canon_layer(dim, bias=config.canon_bias, kernel_size=config.canon_kernel, activation = 'silu' if config.canon_activation else None, use_fast_conv1d=False)
    if config.canon_bias: # and config.canon_bias_zero:
        canon.bias.data = torch.zeros_like(canon.bias)
        assert False, 'must change reset_parameters to reflect this initialization'
    # if config.canon_kernel_init==1:
    #     with torch.no_grad():
    #         canon.weight.data = torch.zeros_like(canon.weight)
    #         if not config.canon_kernel_residual:
    #             canon.weight.data[:, :, -1] = 1.0
    # if config.canon_kernel_constant:
    #     with torch.no_grad():
    #         canon.weight.data = torch.zeros_like(canon.weight)
    #         assert canon.weight.shape[2] == 4
    #         canon.weight.data[:, :, 0] = 0.1
    #         canon.weight.data[:, :, 1] = 0.2
    #         canon.weight.data[:, :, 2] = 0.4
    #         canon.weight.data[:, :, 3] = 0
    # if getattr(config, 'canon_nograd', False):
    #     canon.weight.requires_grad = False
    #     if config.canon_bias:
    #         canon.bias.requires_grad = False
    # if config.canon_kernel_std is not None:
    #     with torch.no_grad():
    #         torch.nn.init.normal_(canon.weight, mean=0.0, std=config.canon_kernel_std)
    #         # bias should already be zero (if exist)
    # canon._zeyuan_no_reinit = True
    canon._zeyuan_residual = config.canon_residual
    return canon


def apply_canon(store_name, canon, hidden_states, cache, attention_mask):
    conv_state = None
    if cache is not None:
        conv_state = getattr(cache, store_name)
        # if cache.print_option:
        #     print(f"Canon using cache for {store_name} on offset {cache.offset} with hidden {hidden_states.shape}, conv_state {conv_state.shape}, std {conv_state.std().item()}, mean {conv_state.mean().item()}")
    # if attention_mask is None or len(attention_mask.shape)==4:
    #     conv_mask = attention_mask.squeeze(1).squeeze(1)[:, -hidden_states.shape[1] :] if attention_mask is not None else None
    # else:
    #     assert len(attention_mask.shape)==2
    #     conv_mask = attention_mask[:, -hidden_states.shape[1] :] if attention_mask is not None else None
    conv_mask = None
    hidden_states2, conv_state_new = canon(x=hidden_states, mask=conv_mask, cache=conv_state, output_final_state=cache is not None)
    # if conv_state is not None:
    #     print(conv_state.shape)
    if cache is not None:
        if conv_state_new is not conv_state:
            conv_state.copy_(conv_state_new)
            assert False, f"Canon cache {store_name} not updated correctly, expected {conv_state.shape}, got {conv_state_new.shape}"
    if canon._zeyuan_residual: return hidden_states + hidden_states2
    else: return hidden_states2
