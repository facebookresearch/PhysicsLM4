# Copyright (c) Meta Platforms, Inc. and affiliates.

import glob
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import plotly.express as px
from omegaconf import OmegaConf


def parallel(func, files, num_workers=16):
    with Pool(num_workers) as p:
        results = p.map(func, files)
    results = list(results)
    # Flatten the list of results
    if len(results) > 0 and isinstance(results[0], list):
        results = [item for sublist in results for item in sublist]

    return results


def parallel_from_glob(func, glob_pattern, num_workers=16):
    files = glob.glob(glob_pattern, recursive=True)
    return parallel(partial(func), files, num_workers=num_workers)


def load_raw_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_raw_jsonl(jsonl_file):
    metrics = []

    with open(jsonl_file, "r") as f:
        for i, line in enumerate(f):
            try:
                json_obj = json.loads(line)
            except json.decoder.JSONDecodeError as e:
                print(f"Error decoding line {i+1} in file {jsonl_file}")

            metrics.append(json_obj)

    return metrics


def get_metrics(path):
    results_dir = Path(path)

    results = load_raw_jsonl(results_dir)
    params = OmegaConf.load(results_dir.parent / "config.yaml")
    params = OmegaConf.to_container(params, resolve=True)
    df = pd.json_normalize(
        [{"params": params, "metrics": res} for res in results], sep="/"
    )
    return df


def get_merged_df(path):
    dfs = parallel_from_glob(get_metrics, path, num_workers=80)
    return pd.concat(dfs)


# %% Example usage
df = get_merged_df("/path/to/metrics.jsonl")
fig = px.line(
    df,
    x="metrics/global_step",
    y="metrics/loss/out",
)
fig.update_yaxes(type="log")
fig.show()

# %%
