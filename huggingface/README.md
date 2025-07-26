# Huggingface Model Code with Canon Layers and Enhanced Features

**Author**: Zeyuan Allen-Zhu  

---

## LlamaCanon Model

- **`demo_pretrained.py`**: Loads a pretrained LlamaCanon model from Huggingface.  
- **`demo_newmodel.py`**: Initializes a new LlamaCanon model from scratch for pretraining.  

### Key Features and Enhancements


### 1. **Canon Layers**  
Reference: [Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers](https://ssrn.com/abstract=5240330).

The LlamaCanon model supports **Canon layers** at various configurable points (A, B, C, D) in the architecture. You can control their behavior using the following settings:

- **`config.canon_set`**:  
  A string specifying the placement of Canon layers.  
  - Example: `"ABCD"` applies Canon layers at all supported locations, whereas `"AC"` applies them only at intermediate layers A and C.

- **`config.canon_residual`** *(default: `True`)*:  
  Enables residual connections for Canon layers.    
  **Recommendation**: Residual connections are *highly recommended* for use with Transformer models (see reference paper).

- **`config.canon_activation`** *(default: `False`)*:  
  Enables SiLU activation for Canon layers.  
  **Recommendation**: It is *NOT recommended* to enable this for Transformer models, according to research findings.

- **`config.canon_kernel`** *(default: `4`)*:  
  Determines the kernel size for the `causal_conv1d` operator, with possible values: `2`, `3`, or `4`.  
  **Note**: Values of `2`, `3`, or `4` allow efficient CUDA implementation. Avoid other values unless specific experimentation is required.

- **`config.canon_bias`** *(default: `False`)*:  
  Enables bias terms for Canon layers.  
  **Recommendation**: Avoid enabling Canon biases without modifying `reset_parameters` for appropriate bias initialization.

### 2. **QK-LayerNorm**  
References:  
- [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442)  
- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322).  

LlamaCanon integrates **QK-LayerNorm**, a configurable normalization mechanism for pretraining stability and scalability:

- **`config.qk_norm`** *(default: `False`)*:  
  Enables QK-LayerNorm with trainable parameters.  

> **Note**: For models with ≤8B parameters, experiments showed marginal benefits from QK-LayerNorm in terms of training stability/performance.

### 3. **Partial RoPE**  
Rotary Position Embedding (RoPE) is partially configurable in LlamaCanon:

- **`config.rope_dim`** *(default: `None`)*:  
  Specifies that only the first `rope_dim` dimensions (out of `head_dim`) will use RoPE.  
  - **Important**: If `config.rope_dim` is set, ensure you also set:
    `config.partial_rotary_factor = rope_dim / head_dim`.  
    This adjustment is necessary for Huggingface's RoPE initialization.

> **Insight**: Our research ([*Physics of Language Models: Part 4.1*](https://ssrn.com/abstract=5240330)) shows that when Canon layers are applied, reducing the extent of RoPE can improve model performance. Overuse of RoPE in such cases may degrade results.


### 4. **Alternative RoPE Coordination**  
Adjusting RoPE implementation at training time:

- **`config.rope_version`** *(default: `"huggingface"`)*:  
  Supports compatibility with Huggingface and Lingua RoPE coordinate systems.  
  - Use `rope_version = "lingua"` to load Lingua-pretrained models.  
  - For Huggingface-trained models, stick to `rope_version = "huggingface"`.

> **Note**: Once a model has been trained, *do not* change the RoPE version during inference or further training.


### 5. **Loading from Lingua**  
The code includes utilities for loading Lingua-pretrained Transformer models into the LlamaCanon architecture:

- **Method**: `model.load_from_lingua_state`  
- **Details**: Maps the Lingua-trained model's `state_dict` to the LlamaCanon architecture.  
  Implementation details are available in `model.from_pretrained()` (located in `modeling_llama_canon.py`).

> **Reference**: This functionality relies on the Lingua-modified codebase, available at [`lingua_modified`](../lingua_modified). It is compatible with our [model release](../lingua_results/), and supports for instance changing n_kv_heads and rope_theta.


## Future Releases: Models with Canon Layers

Additional models like GLA, GDN, and Mamba2 will be released soon. Experimental results are detailed in:  
[*Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers*](https://ssrn.com/abstract=5240330).

## 📖Citation

Please cite the following if you use our models or findings in your research:
```bibtex
@article{Allenzhu2025-canon,
  author = {{Allen-Zhu}, Zeyuan},
  title = {{Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers}},
  year = {2025},
  month = {May},
  journal = {SSRN Electronic Journal},
  note = {\url{https://ssrn.com/abstract=5240330}}
}
```