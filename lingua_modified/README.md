This is a modification of the original [Meta Lingua](https://github.com/facebookresearch/lingua) pretraining codebase. The original README.md file is attached at the end.

# Meta Lingua (our modification)

**Author of this modification**: Zeyuan Allen-Zhu  

Our extension introduces several enhancement features, including support for Canon layers, QK-layernorm, Z-loss, and Partial RoPE. These features are explained in detail below.

## Key Features and Enhancements

### 1. **Canon Layers**  
Reference: [Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers](https://ssrn.com/abstract=5240330).

Canon layers are supported at optional points (A/B/C/D) in the architecture. Configuration options in the YAML file:  
- **`model.canon_set`**:  
  A string specifying where Canon layers are applied.  
  Example: `"ABCD"` indicates full Canon layers, while subsets (e.g., `"AC"`) reflect partial Canon configuration A and C only.  
- **`model.canon_residual`** *(default: `True`)*:  
  Enables residual connections on Canon layers. **Strongly recommended** for Transformer models (see the linked paper).  
- **`model.canon_activation`** *(default: `False`)*:  
  Enables SiLU activation on Canon layers. **Strongly recommended NOT to enable** for Transformer models.  
- **`model.canon_kernel`** *(default: `4`)*:  
  Specifies the kernel size for the `causal_conv1d` operator. Values of `2`, `3`, or `4` enable fast CUDA implementation. Avoid changing this unless necessary.  
- **`model.canon_bias`** *(default: `False`)*:  
  Adds bias terms to Canon layers. This is **NOT recommended**. If enabled, ensure to modify `reset_parameters` for appropriate initial bias values.  

### 2. **QK-Layernorm**  
References:  
- [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442)  
- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322).  

Configurations:  
- **`model.qk_norm`** *(default: `False`)*:  
  Enables QK-Layernorm with trainable parameters.  

> **Note**: For our released models (≤8B size), enabling QK-Layernorm did not show noticeable improvements in training performance/stability.

### 3. **Z-Loss**  
References:  
- [PaLM: Scaling Language Modeling with Pathways)](https://arxiv.org/abs/2204.02311)  
- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322).  

Configurations:  
- **`model.z_loss`** *(default: `False`)*:  
  Adds a Z-loss term `0.0001 * log^2` to the loss function.  

> **Note**: Similar to QK-Layernorm, enabling Z-loss showed negligible impact on training for our released models (≤8B size).

### 4. **Partial RoPE**  
- **`model.rope_dim`** *(int, default: `None`)*:  
  Specifies that only the first `rope_dim` dimensions (out of `head_dim`) will incorporate RoPE.  

> **Insight**: As shown in [*Physics of Language Models: Part 4.1*](https://ssrn.com/abstract=5240330), when Canon layers are enabled, RoPE usage can be greatly reduced, such as to 25% of `head_dim`. In fact, it is a folklore that RoPE "hurts" model performance, but without Canon layers, it can be hard to reduce or get rid of RoPE (e.g., Alibi is not as effective as Canon layers for the purpose of removing RoPE).

## Model Weights

Pretrained model weights are available on [Hugging Face](https://huggingface.co/facebook/PhysicsLM4.2__LlamaCanon-8B-Nemo-1T-lr0.003). The training recipes (yaml files) can be found in [`../lingua_recipes`](../lingua_recipes), although an example yaml file with Canon layers is in [`apps/main/configs/canon_1B.yaml`](apps/main/configs/canon_1B.yaml).


## Comparison and Conversion to Huggingface

The Transformer model implementation in the Meta Lingua repository is nearly identical to `LlamaModel` [on Huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py). However, there is one critical difference when applying RoPE: the Meta Lingua codebase uses a different coordinate permutation for RoPE application.

To bridge this gap, we provide a [**conversion script**](../huggingface/) that allows users to convert any models pretrained using the Meta Lingua codebase (including our added Canon layers, QK-norm, or Partial RoPE) into a Huggingface-compatible model. The converted model can then be seamlessly used for inference, including `model.generate`.

This conversion process ensures smooth interoperability between Meta Lingua's pretrained models and the Huggingface ecosystem.

## License

The original Lingua codebase adheres to its [BSD-3-Clause](LICENSE) license, but all of our modifications are under [Apache 2.0](../LICENSE).

---

---

---

# Meta Lingua (original README.md)

**Mathurin Videau***, **Badr Youbi Idrissi***, Daniel Haziza, Luca Wehrstedt, Jade Copet, Olivier Teytaud, David Lopez-Paz. ***Equal and main contribution**

Meta Lingua is a minimal and fast LLM training and inference library designed for research. Meta Lingua uses easy-to-modify PyTorch components in order to try new architectures, losses, data, etc. We aim for this code to enable end to end training, inference and evaluation as well as provide tools to better understand speed and stability. While Meta Lingua is currently under development, we provide you with multiple `apps` to showcase how to use this codebase.

<p align="center">  
 <img src="lingua_overview.svg" width="100%"/>
</p>

## Quick start

The following commands launch a SLURM job that creates an environment for Meta Lingua.
The env creation should take around 5 minutes without counting downloads. 

```bash
git clone https://github.com/facebookresearch/lingua
cd lingua

bash setup/create_env.sh
# or if you have access to a SLURM cluster
sbatch setup/create_env.sh
```
Once that is done your can activate the environment 
```bash
conda activate lingua_<date>
```
use the provided script to download and prepare data from huggingface (among `fineweb_edu`, `fineweb_edu_10bt`, or `dclm_baseline_1.0`).
This command will download the `fineweb_edu` and prepare it for training in the `./data` directory, specifying the amount of memory `terashuf` (the tool used to shuffle samples) will be allocated. By default, the number of chunks (`nchunks`) is 32. If you are running on fewer than 32 GPUs, it is recommended to set `nchunks` to 1 or to match `nchunks` with the number of GPUs (`nchunks` = NGPUs). See [here](https://github.com/facebookresearch/lingua/issues/55#issuecomment-2483643076) for more details.
```bash
python setup/download_prepare_hf_data.py fineweb_edu <MEMORY> --data_dir ./data --seed 42 --nchunks <NCHUNKS>
```
to download tokenizer (here llama3), use the folowing script:
```bash
python setup/download_tokenizer.py llama3 <SAVE_PATH> --api_key <HUGGINGFACE_TOKEN>
```
Now launch a debug job to check if everything works.  **The provided configurations are templates, you need to adapt them for them to work (change `dump_dir`, `data.root_dir`, `data.tokenizer.path`, etc ...)**

```bash
# stool stands for SLURM tool !
python -m lingua.stool script=apps.main.train config=apps/main/configs/debug.yaml nodes=1 partition=<partition>
# if you want to launch locally you can use torchrun
torchrun --nproc-per-node 8 -m apps.main.train config=apps/main/configs/debug.yaml
# or you can also launch on 1 GPU
python -m apps.main.train config=apps/main/configs/debug.yaml
```

When using `stool`, if a job crashes, it can be relaunched using sbatch:
```bash
sbatch path/to/dump_dir/submit.slurm
```
## Training Results 

We get very strong performance on many downstream tasks and match the performance of [DCLM baseline 1.0](https://arxiv.org/abs/2406.11794).

### 1B models on 60B DCLM tokens
| name           | arc_challenge | arc_easy | boolq |  copa | hellaswag |  obqa |  piqa |  siqa | winogrande |  nq  |  tqa  |
|----------------|:-------------:|:--------:|:-----:|:-----:|:---------:|:-----:|:-----:|:-----:|:----------:|:----:|:-----:|
| Transformer 1B |     36.48     |   62.83  | 62.57 | 79.00 |   63.62   | 37.40 | 75.14 | 45.19 |    61.64   | 8.75 | 26.31 |
| minGRU 1B      |     30.82     |   57.89  | 62.05 | 74.00 |   50.27   | 37.00 | 72.31 | 43.76 |    52.49   | 3.24 |  9.03 |
| minLSTM 1B     |     31.76     |   60.04  | 62.02 | 73.00 |   53.39   | 36.40 | 72.36 | 45.09 |    52.80   | 4.52 | 12.73 |
| Hawk 1B        |     34.94     |   63.68  | 62.42 | 76.00 |   63.10   | 38.20 | 73.23 | 46.01 |    55.33   | 8.42 | 23.58 |
| Mamba 1B       |     35.54     |   63.42  | 62.63 | 74.00 |   64.16   | 38.80 | 75.24 | 45.14 |    60.14   | 8.84 | 26.64 |

### 7B models

| name                             | arc_challenge | arc_easy | boolq | copa  | hellaswag | obqa  | piqa  | siqa  | winogrande | mmlu  | nq    | tqa   | bbh   |
|----------------------------------|---------------|----------|-------|-------|-----------|-------|-------|-------|------------|-------|-------|-------|-------|
| Mamba 7B 200B tokens             | 47.21         | 76.03    | 65.63 | 84.00 | 77.80     | 44.00 | 80.25 | 49.69 | 70.24      | 32.81 | 20.53 | 51.93 | 20.35 |
| Llama 7B 200B tokens             | 46.95         | 75.73    | 64.80 | 84.00 | 77.45     | 45.00 | 80.20 | 48.26 | 70.32      | 48.64 | 20.66 | 51.01 | 31.47 |
| Llama 7B squared relu 1T tokens  | 49.61         | 76.74    | 72.45 | 89.00 | 81.19     | 44.80 | 82.05 | 49.95 | 72.14      | 60.56 | 25.68 | 59.52 | 42.11 |

## Project overview

Meta Lingua is structured as follows:

```
📦meta-lingua
 ┣ 📂lingua # Core library
 ┃ ┣ 📜args.py
 ┃ ┣ 📜checkpoint.py
 ┃ ┣ 📜data.py
 ┃ ┣ 📜distributed.py
 ┃ ┣ 📜float8.py
 ┃ ┣ 📜logger.py
 ┃ ┣ 📜metrics.py
 ┃ ┣ 📜optim.py
 ┃ ┣ 📜probe.py
 ┃ ┣ 📜profiling.py
 ┃ ┣ 📜stool.py
 ┃ ┣ 📜tokenizer.py
 ┃ ┗ 📜transformer.py
 ┣ 📂setup
 ┃ ┣ 📜create_env.sh
 ┃ ┗ 📜download_prepare_hf_data.py
 ┗ 📂apps # Apps that put components together
   ┣ 📂main # Main language modeling app with llama
   ┃ ┣ 📂configs
   ┃ ┣ 📜eval.py
   ┃ ┣ 📜generate.py
   ┃ ┣ 📜train.py
   ┃ ┗ 📜transformer.py
   ┣ 📂fastRNN 
   ┃ ┣ 📂component
   ┃ ┣ 📂hawk
   ┃ ┣ 📂minGRU
   ┃ ┣ 📂minLSTM
   ┣ 📂mamba
   ┣ 📂mtp # Multi token prediction
   ┗ 📂plots
```

The `lingua` folder contains some essential and reusable components, while the `apps` folder contains scripts that put those components together. For instance the main training loop is in `apps/main`. We highly encourage you to use that as a template and modify it however you please to suit your experiments. 

Nothing is sacred in Meta Lingua. We've specifically tried to make it as easily modifiable as possible! So feel free to branch out and modify anything. 

Here's a quick description of the most important files and features:

- **`transformer.py`** : Defines model architecture. This is pure PyTorch `nn.Module` ! Nothing fancy here. 
- **`distributed.py`** : Handles distributing the model on multiple GPUs. This is done through `parallelize_module` function which wraps your vanilla `nn.Module` and applies nearly any combination of Data Parallel, Fully Sharded Data Parallel, Model Parallelism, `torch.compile`, activation checkpointing and `float8`. 
- **`data.py`** : Dataloader for LLM pretraining.

<p align="center">  
 <img src="dataloader.png" width="40%"/>
</p>

- **`profiling.py`** : Small wrapper around xformers' profiler which provides automatic MFU and HFU calculation and dumps profile traces in profiling folder in your dump directory. It also has memory profiling trace. 
- **`checkpoint.py`** : Manages model checkpoints. It saves model in checkpoints folder in your dump dir in .distcp format which is the new PyTorch distributed saving method. This format allows to reload the model with a different number of GPUs and with a different sharding. You can also convert those into normal PyTorch checkpoints with `torch.distributed.checkpoint.format_utils.dcp_to_torch_save` and the other way around `torch_save_to_dcp`.
- **`args.py`** : Utilities to work with configs. 

## Configuration

Most components need configuration and we chose to use data classes to represent these configuration objects. `args.py` helps with converting between `config.yaml` and config dictionaries into the respective data classes. 

So for examples the `TrainArgs` in `apps/main/train.py` has a `LMTransformerArgs`, `OptimArgs`, etc ... as children. 

Here is an example configuration file that will be converted to `TrainArgs`:

```yaml
# This is where Meta Lingua will store anything related to the experiment. 
dump_dir: /path/to/dumpdir
name: "debug"
steps: 1000

seed: 12

optim:
    lr: 3e-4
    warmup: 2000
    lr_min_ratio: 0.000001
    clip: 10.0

distributed:
    fsdp_type: full_shard
    compile: true
    selective_activation_checkpointing: false

model:
    dim: 1024
    n_layers: 8
    n_heads: 8

data:
    root_dir: data/shuffled
    sources:
      wikipedia: 80.0
      arxiv: 20.0
    batch_size: 32
    seq_len: 1024
    load_async: true
    tokenizer:
        name: sp
        path: tokenizers/llama2.model
```


## Launching jobs

### Command line arguments

The command line interface in all scripts (`train.py`, `eval.py`, `stool.py`) uses [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments)
This accepts arguments as a dot list
So if the dataclass looks like
```python
@dataclass
class DummyArgs:
    name: str = "blipbloup"
    mode: LMTransformerArgs = LMTransformerArgs()
    
@dataclass
class LMTransformerArgs:
    dim: int = 512
    n_layers: int = 12
```

Then you can pass `model.dim = 32` to change values in `LMTransformerArgs`
or just `name = tictac` for top level attributes.

**`train.py`** simply takes as argument the path to a config file and will load that config. The behavior here is as follows:
1. We instantiate `TrainArgs` with its default values
2. We override those default values with the ones in the provided config file
3. We override the result with the additional arguments provided through command line

If we take the `DummyArgs` example above, calling `train.py` with `train.py config=debug.yaml model.dim=64 name=tictac` 
where `debug.yaml` contains 
```yaml
model:
    n_layers: 24
```
will launch training with the config 
```python
DummyArgs(name="tictac", LMTransformerArgs(dim=64, n_layers=24))
```

### Launching with SLURM

Since we want to do distributed training, we need `train.py` to run N times (with N being the number of GPUs)

The easiest way to do this is through SLURM. And in order to make that simpler, we provide `lingua/stool.py` which is a simple python script that 
1. Saves the provided config to `dump_dir`
2. Copies your current code to `dump_dir` in order to back it up 
3. Creates an sbatch file `submit.slurm` which is then used to launch the job with the provided config. 

It can either be used through command line 

```bash
python -m lingua.stool config=apps/main/configs/debug.yaml nodes=1 account=fair_amaia_cw_codegen qos=lowest
```

Or the `launch_job` function directly. This allows you for example to create many arbitrary configs (to sweep parameters, do ablations) in a jupyter notebook and launch jobs directly from there. 

Since the configuration file is copied to `dump_dir`, an easy way to iterate is to simply change the config file and launch the same command above. 

## Debugging
In order to iterate quickly, it is preferable not to have to wait for a SLURM allocation every time. You can instead ask SLURM to allocate resources for you, then once they're allocated you can run multiple commands on that same allocation. 

For example you can do :

```bash
salloc --nodes 2 --cpus-per-gpu 16 --mem 1760GB --gres=gpu:8 --exclusive --time=72:00:00
```

Which will give you access to 2 nodes in your current terminal. Once the allocation is done, you will see some SLURM environement variables that were automatically added such as `$SLURM_JOB_ID` and others... This allows you for example to do in the same terminal

```bash
srun -n 16 python -m apps.main.train config=apps/main/configs/debug.yaml
```

Which will run the `python -m apps.main.train config=apps/main/configs/debug.yaml` command on each of the 16 GPUs. If this crashes or ends you can just relaunch `srun` again because the nodes are already allocated to you and you don't have to wait for SLURM to give you the resources again.

This will also show you the outputs of all those commands in the same terminal which might become cumbersome. 

Instead you can use `stool` directly to configure logs to be separated into different files per GPU.

```bash
python -m lingua.stool config=apps/main/configs/debug.yaml nodes=2 launcher=bash dirs_exists_ok=true
```

Notice that we added **`launcher=bash`** which basically means that the generated `submit.slurm` will simply be executed instead of submitting it through `sbatch`. The `submit.slurm` has an `srun` command also so this is very similar to the above `srun` command. We also add **`dirs_exists_ok=true`** to tell `stool` that it is okay to override things in an existing folder (code, config, etc)

If you want to use `pdb` to step through your code, you should use `-n 1` to run only on 1 GPU. 

## Evaluations

Evaluations can run either during training periodically or you directly launch evals on a given checkpoint as follows:

```bash
srun -n 8 python -u -m apps.main.eval config=apps/main/configs/eval.yaml
```

You need to specify the checkpoint and dump dir of the evaluation in that config

Or through `stool` with

```bash
python -m lingua.stool script=apps.main.eval config=apps/main/configs/eval.yaml nodes=1 account=fair_amaia_cw_codegen qos=lowest
```

## Dump dir structure

```
📂example_dump_dir
 ┣ 📂checkpoints
 ┃ ┣ 📂0000001000
 ┃ ┣ 📂0000002000
 ┃ ┣ 📂0000003000
 ┃ ┣ 📂0000004000
 ┃ ┣ 📂0000005000
 ┃ ┣ 📂0000006000
 ┃ ┣ 📂0000007000 # Checkpoint and train state saved every 1000 steps here
 ┃ ┃ ┣ 📜.metadata
 ┃ ┃ ┣ 📜__0_0.distcp
 ┃ ┃ ┣ 📜__1_0.distcp
 ┃ ┃ ┣ 📜params.json
 ┃ ┃ ┣ 📜train_state_00000.json
 ┃ ┃ ┗ 📜train_state_00001.json
 ┣ 📂code # Backup of the code at the moment the job was launched
 ┣ 📂logs
 ┃ ┗ 📂166172 # Logs for each GPU in this SLURM job.
 ┃ ┃ ┣ 📜166172.stderr
 ┃ ┃ ┣ 📜166172.stdout
 ┃ ┃ ┣ 📜166172_0.err
 ┃ ┃ ┣ 📜166172_0.out
 ┃ ┃ ┣ 📜166172_1.err
 ┃ ┃ ┗ 📜166172_1.out
 ┣ 📂profiling
 ┃ ┣ 📂memory_trace_plot # Trace of memory usage through time for all GPUs
 ┃ ┃ ┣ 📜000102_h100-192-145_451082.html
 ┃ ┃ ┣ 📜000102_h100-192-145_451083.html
 ┃ ┗ 📂profile_CPU_CUDA_000104 # Profiling traces for all GPUs
 ┃ ┃ ┣ 📜h100-192-145_451082.1720183858874741723.pt.trace.json.gz
 ┃ ┃ ┗ 📜h100-192-145_451083.1720183858865656716.pt.trace.json.gz
 ┣ 📜base_config.yaml
 ┣ 📜config.yaml
 ┣ 📜metrics.jsonl
 ┗ 📜submit.slurm
```

## Related repositories

Here we highlight some related work that is complementary to this one. Most important being [torchtitan](https://github.com/pytorch/torchtitan), [torchtune](https://github.com/pytorch/torchtune) and [fairseq2](https://github.com/facebookresearch/fairseq2). 

Lingua is designed for researchers who want to experiment with new ideas for LLM pretraining and get quick feedback on both training/inference speed and downstream benchmarks. Our goal is to lower the barrier to entry for LLM research by providing a lightweight and focused codebase.

We see torchtitan, torchtune, lingua and fairseq2 as complementary tools. Torchtitan is excellent for large-scale work because it features 3D parallelism and is likely to integrate the latest PyTorch distributed training features more quickly, thanks to its close ties to the PyTorch team. On the other hand, Torchtune excels at fine-tuning, especially when GPU resources are limited, by offering various fine-tuning strategies like LoRA, QLoRA, DPO, and PPO. Fairseq2 is a FAIR project for sequence modeling with multi-modal capabilities that provides various LLM training recipes, multi-GPU support with data and model parallelism, and efficient data processing for speech and multilingual content.

A typical workflow could look like this: you might first test a new idea in Lingua, then scale it up further with Torchtitan, and finally use Torchtune for instruction or preference fine-tuning.

Although there's definitely some overlap among these codebases, we think it's valuable to have focused tools for different aspects of LLM work. For example, Torchtitan aims to showcase the latest distributed training features of PyTorch in a clean, minimal codebase, but for most research, you really don't need every feature PyTorch has to offer or the capability to scale to 100B parameters on 4096 GPUs. For instance, we think that FSDP + torch compile will cover 90% of all needs of a researcher. With lingua, we tried to ask "What's the minimal set of features needed to draw solid conclusions on the scalability of idea X?"

We believe this targeted approach helps researchers make progress faster without the mental overhead of using many techniques that might not be needed.

## Citation

```
@misc{meta_lingua,
  author = {Mathurin Videau and Badr Youbi Idrissi and Daniel Haziza and Luca Wehrstedt and Jade Copet and Olivier Teytaud and David Lopez-Paz},
  title = {{Meta Lingua}: A minimal {PyTorch LLM} training library},
  url = {https://github.com/facebookresearch/lingua},
  year = {2024}
}
```
## License

Meta Lingua is licensed under BSD-3-Clause license. Refer to the LICENSE file in the top level directory.
