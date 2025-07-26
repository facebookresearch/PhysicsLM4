# Autoregressive U-Net (AU-Net)

**Mathurin Videau**\*, **Badr Youbi Idrissi**\*, Alessandro Leite, March Schoenauer, Olivier Teytaud, David Lopez-Paz. \***Equal and main contribution**

[[`arXiv`](https://arxiv.org/abs/2506.14761)]

This repository contains the official implementation of AU-Nets. For details, refer to our paper [**From Bytes to Ideas: Language Modeling with Autoregressive U-Nets**](https://arxiv.org/abs/2506.14761).



## Training

AU-Net is implemented as an app inside the Lingua framework. You can launch it the same way as other Lingua apps:

```bash
python -m lingua.stool script=apps.aunet.train config=apps/aunet/config/relevent_config.yaml nodes=<num_node> account=<slurm_account> qos=<slurm_qos>
```

Different configurations corresponding to different network sizes and architectures can be found in `apps/aunet/configs`.

### Config Description

AU-Net uses a configuration format similar to a classical Transformer, with slight modifications to accommodate its multi-stage structure.

#### Model Configuration
The model config contains list of arguments corresponding to argumetns specific to each stage. As AU-net is a symetric architecture, the first element of the list correspond to argument for the fisrt stage and last stage, then second and before last stage etc... The word 'level' when encountered in the code can be interpreted as 'stage' described in the paper.

```yaml
model:
    dimensions: [dim_1, ..., dim_n]              # Dimensions for each stage pair
    layers: [n_layer_1, ..., n_layer_n]          # Number of layers for each stage pair
    head_dims: [head_dim_1, ..., head_dim_n]     # Head dimension for attention at each stage pair
    residuals: [bool_1, ..., bool_n]             # Use residual connections (typically set to True)
    sliding_windows: [win_1, ..., win_n]         # Sliding window size per stage pair
    max_seqlens: [-1, max_2, ..., max_n]         # Max sequence length per stage pair (-1 = inherited from data)
    block:                                       # Shared parameters for all stages
        rope_theta: 500000.0
        multiple_of: 256
    lambda_level: 0.0                            # > 0. Adds a linear prediction of stage assignment
    pooling_type: simple_indexed_matmul          # Pooling strategy
```

#### Data Configuration

```yaml
data:
    root_dir: /path/to/data
    sources:
      dclm_baseline_1.0: 1.0
    tokenizer:
        name: bytes                              # AU-Net operates at byte level

    regex:
        strategy:
            strat_1: <strat_name>                # Cutting strategy per stage pair
            ...
            strat_n: <strat_name>
```

For more details about data loading, training, evaluation, and requirements, please refer to the general README of Lingua.

## Quick Code Overview

AU-Net-specific training relies mainly on two key files: `data/regex_cutting.py` and `hierarchical.py`.

### `regex_cutting.py`

Implements various strategies for segmenting byte sequences into larger units using the `RegexPool` class. Available options:

- `word`: splits text by whitespace
- `pretok`: follows pretokenization rules (more conservative)
- `punc`: pools on punctuation marks

You can add your own strategy by modifying `RegexPool.__init__()` and providing a tuple `(regex_start, regex_end)`, where each is a distinct regex. `regex_start` is used to identify the beginning of a segment using `match.start()`, and `regex_end` determines the end using `match.end()`. These two options helps to design regex that are invarient to right insertion.

This logic integrates with `data/data.py`, which outputs token sequences and their corresponding pooling masks (used as `level_mask` in `train.py`).

### `hierarchical.py`

Contains the core model implementation, focusing on two key areas:

1. **Stage Instantiation**: Symmetrical architecture with encoders, a central trunk, and decoders. All components are built using a shared `CausalTransformer` class.
2. **Pooling/Upsampling**: Managed by the `SimpleTransition` class. Pooling strategies include:
   - `repeat`
   - `indexed_matmul`
   - `non_param`

You can implement custom pooling classes, as long as they maintain the same interface (`up` and `down` methods) as `SimpleTransition`.

## Citation

```
@article{videau2025bytesideaslanguagemodeling,
  author = {Mathurin Videau and Badr Youbi Idrissi and Alessandro Leite and Marc Schoenauer and Olivier Teytaud and David Lopez-Paz},
  title = {From Bytes to Ideas: Language Modeling with Autoregressive U-Nets},
  journal = {arXiv preprint arXiv:2506.14761},
  year = {2025}
}
```
