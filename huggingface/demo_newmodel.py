# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This code creates a new LlamaCanon model from scratch, so that you can train using it

from configuration_llama_canon import LlamaCanonConfig
from modeling_llama_canon import LlamaCanonForCausalLM


cfg = LlamaCanonConfig(vocab_size=32000,
                        hidden_size=2048,
                        intermediate_size=5632,
                        num_hidden_layers=24,
                        num_attention_heads=32,
                        num_key_value_heads=32,
                        qk_norm = False,
                        rope_dim = 16,
                        canon_set = "ABCD",
                        rope_theta = 10000.0,
                        rms_norm_eps = 1e-5,
                        max_position_embeddings=4096,
                        # Zeyuan's note: Lingua's transformer is slightly different from Huggingface Llama, in how the RoPE coordinates are permuted
                        # If you train a model using Lingua codebase, you must set `rope_version='lingua'` before loading the model;
                        #    check my code in huggingface/modeling_llama_canon.py
                        rope_version = 'huggingface',  
                    )
cfg._attn_implementation = 'sdpa'
# this must equal rope_dim / head_dim (or if rope_dim is not set, this can be 1.0)
# the reason we need to set this twice, is because my code will use rope_dim, but huggingface _compute_default_rope_parameters will use partial_rotary_factor
cfg.partial_rotary_factor = 0.25

model = LlamaCanonForCausalLM(cfg)

print(model)

for n,p in model.named_parameters():
    print(n, p.requires_grad, p.shape, p.std().item(), p.mean().item())
# One thing critical to check is that all canonA/B/C/D.weights have std around 0.288, which is the default Conv1D initialization
# Changing it to 0.02 is not wise for Transformer models, and please see our paper 
#    "Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers"
# especially v2.0 for the discussion on the Canon layer initialization.
# If you use our canon_helper.py on other models, please note some Huggingface models may overwrite all nn.Conv1D initialization to 0.02;
#    you must revise their reset_parameters() functions, to make sure they do not overwrite nn.Conv1D's weight to config.initializer_range for Canon layers
