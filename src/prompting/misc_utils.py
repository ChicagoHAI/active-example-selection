import hashlib
import json
import logging
import os
import random
from collections import Counter
from collections.abc import Iterable
from functools import reduce
from itertools import permutations
from math import factorial
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

logger = logging.getLogger()


def seed_everything(seed: int):
    logger.info(f"re-seeding with seed {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_output_dir(conf: DictConfig):
    model = conf["model-alias"] if "model-alias" in conf else conf.model_name
    if not conf.get("output_dir"):
        output_dir = os.path.join(
            conf.basedir,
            conf.processor_kwargs.mode,
            "calibrate" if conf.model_kwargs.get("calibrate", False) else "vanilla",
            conf.task,
            model,
            conf.strategy,
            str(conf.seed),
        )
        conf.output_dir = output_dir
    os.makedirs(conf.output_dir, exist_ok=True)


def setup_logging(conf: DictConfig):
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%y-%m-%d %H:%M"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if conf.output_dir:
        log_path = os.path.join(conf.output_dir, "log.txt")
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log_path = os.path.join(conf.output_dir, "debug.txt")
        dfh = logging.FileHandler(log_path)
        dfh.setLevel(logging.DEBUG)
        dfh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(dfh)
        logger.info(f"logging to {log_path}.")


def entropy(probs: torch.FloatTensor) -> torch.FloatTensor:
    return -(probs * torch.log2(probs)).nansum()


def print_json(d: Dict, indent: Optional[int] = 2) -> str:
    return json.dumps(d, indent=indent) + "\n"


def read_json_file(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def deterministic_hash(s: str) -> int:
    return int(hashlib.sha512(s.encode("utf-8")).hexdigest(), 16)


def unique_permutations(indices: List, samples: int) -> List[List]:
    unique_counts = Counter(indices)
    total_combos = factorial(len(indices)) // reduce(
        lambda x, y: x * y, map(factorial, unique_counts.values()), 1
    )
    if total_combos <= samples:
        logger.info(
            f"total_combos ({total_combos}) <= ensemble_size ({samples})."
            f" Using all {total_combos} combinations."
        )
        if factorial(len(indices)) > 10**7:
            raise Exception(f"Too slow to permute, indices: {indices}")
        ensemble_indices = list(map(list, sorted(set(permutations(indices)))))

    else:
        ensemble_set = set()
        ensemble_indices = []
        while len(ensemble_indices) < samples:
            shuffled = random.sample(indices, k=len(indices))
            if tuple(shuffled) not in ensemble_set:
                ensemble_set.add(tuple(shuffled))
                ensemble_indices.append(shuffled)

    return ensemble_indices


def kl_div_disagreement(probs: torch.tensor):
    P = probs
    q = probs.mean(dim=0, keepdims=True)
    relative_probs = P / q
    return (P * torch.log2(relative_probs)).nansum(dim=1).max()


def deterministic_random(seed: int) -> random.Random:
    return random.Random(seed)


def flatten(x: Any) -> Generator:
    if isinstance(x, Iterable):
        for it in x:
            yield from flatten(it)
    else:
        yield x
