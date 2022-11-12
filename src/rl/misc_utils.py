import math
import os
import random
from typing import Dict, List

import torch
from omegaconf import DictConfig


def setup_rl_output_dir(conf: DictConfig):
    if conf.get("dry_run") == True:
        conf.output_dir = None
        return

    if conf.get("output_dir") is not None:
        os.makedirs(conf.output_dir, exist_ok=True)
        return

    dirs = [conf.basedir, conf.env]
    if "task" in conf.env_kwargs:
        dirs.append(conf.env_kwargs["task"])
    if "model" in conf.env_kwargs:
        model = (
            conf.env_kwargs["model-alias"]
            if "model-alias" in conf.env_kwargs
            else conf.env_kwargs.model
        )
        dirs.append(model)

    dirs.extend([conf.agent, conf.name])
    output_dir = os.path.join(*dirs)
    os.makedirs(output_dir, exist_ok=True)
    conf.output_dir = output_dir


def parse_step_from_checkpoint(path: str) -> int:
    try:
        filename = os.path.basename(path)
        assert filename.endswith(".ckpt")
        filename = filename[:-5]
        filename = filename.split("_")[1]
        return int(filename)
    except:
        raise Exception(f"Unable to get step from name {path}")


def tensor_stats(t: torch.Tensor) -> torch.Tensor:
    return torch.stack((t.min(), t.max(), t.std()))


def tailsum(t: torch.tensor) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(t, dims=(0,)), dim=0), dims=(0,))


def collate_summaries(summaries: List[Dict]) -> Dict:
    if len(summaries) == 1:
        return summaries[0]

    for summary in summaries:
        assert summary.keys() == summaries[0].keys()

    collated = {}
    for key, value in summaries[0].items():
        values = [summary[key] for summary in summaries]
        if isinstance(value, float):
            # take mean
            collated[key] = sum(values) / len(values)
            collated[key + "-raw"] = values
        elif isinstance(value, List):
            # keep lists
            collated[key + "-raw"] = values
        else:
            raise Exception("only know how to collate float and List")

    return collated


def normalized_entropy(probs: torch.FloatTensor) -> torch.FloatTensor:
    num_classes = probs.shape[0]
    return -(probs * torch.log2(probs)).nansum() / math.log2(num_classes)
