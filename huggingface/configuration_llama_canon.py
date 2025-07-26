# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This code is modified from the huggingface v4.47-release on the Llama model config
# Namely: https://github.com/huggingface/transformers/blob/v4.47-release/src/transformers/models/llama/configuration_llama.py
#
# Zeyuan's edit note: added support for canon layers, see "Part 4.1, Architecture Design and the Magic of Canon Layers" (https://ssrn.com/abstract=5240330)
#
# Zeyuan's edit note: added support for qk_norm, see for instance "Scaling Vision Transformers to 22 Billion Parameters" (arxiv.org/abs/2302.05442)
#
# Zeyuan's edit note: added support for rope_dim, which means only rope_dim of head_dim will be used for rotary position embeddings, if None, then all head_dim will be used
#                 PS: GPTNeoXModel on HF defaults this to 25% of the head_dim, while Llama model sets this to None
#
# Zeyuan's edit note: the lingua codebase has slightly different RoPE implementation (for which coordinates are real/imaginary), and it is not compatible with the huggingface implementation
#                     so we added a field to specify the version of RoPE, which is a string that can be either 'huggingface' or 'lingua'
#                     When loading a checkpoint trained using the lingua codebase, must set `rope_version='lingua'`
#
"""LLaMA Canon model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class LlamaCanonConfig(PretrainedConfig):

    model_type = "LlamaCanon"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `LlamaModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        rope_version='huggingface',
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # Zeyuan's edit note: added support for canon layers, see "Part 4.1, Architecture Design and the Magic of Canon Layers" (https://ssrn.com/abstract=5240330)
        self.canon_set = kwargs.pop("canon_set", "")
        self.canon_bias = kwargs.pop("canon_bias", False)
        self.canon_activation = kwargs.pop("canon_activation", False)
        self.canon_kernel = kwargs.pop("canon_kernel", 4)
        self.canon_residual = kwargs.pop("canon_residual", True)

        # Zeyuan's edit note: added support for qk_norm, see for instance "Scaling Vision Transformers to 22 Billion Parameters" (arxiv.org/abs/2302.05442)
        self.qk_norm = kwargs.pop("qk_norm", False)

        # Zeyuan's edit note: added support for rope_dim, which means only rope_dim of head_dim will be used for rotary position embeddings, if None, then all head_dim will be used
        # PS: GPTNeoXModel on HF defaults this to 25% of the head_dim, while Llama model sets this to None
        self.rope_dim = kwargs.pop("rope_dim", None)
        if self.rope_dim is not None:
            self.partial_rotary_factor = self.rope_dim / self.head_dim

        # Zeyuan's edit note: the lingua codebase has slightly different RoPE implementation (for which coordinates are real/imaginary), and it is not compatible with the huggingface implementation
        # so we added a field to specify the version of RoPE, which is a string that can be either 'huggingface' or 'lingua'
        self.rope_version = rope_version

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

