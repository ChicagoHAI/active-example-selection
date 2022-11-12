import logging
import os
from os.path import join
from typing import Any, Dict, List

import numpy as np
from omegaconf import DictConfig, OmegaConf

from prompting import BaseProcessor, GenerationOutput, GPT2Wrapper
from prompting.misc_utils import print_json, read_json_file

logger = logging.getLogger(__name__)


class BaseStrategy:
    def __init__(self, conf: DictConfig):
        self.output_dir = conf.output_dir
        self.write_config(conf)

    def __call__(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int, **kwargs):
        results = self.read_run(shot)
        if results is None:
            results = self.run_strategy(proc, model, shot, **kwargs)
            self.write_run(results, shot)
        else:
            logging.info(f"Found cache for shot={shot}, skipping run.")
        return results

    @property
    def class_name(self):
        return self.__class__.__name__

    def write_result(self, result: Dict[str, Any], shot: int):
        # Write intermediate results for some shot (repetitions)
        dirs = [f"{shot}-shot"]
        filename = f"result.json"
        self.write_file(print_json(result), filename, dirs=dirs)

    def write_run(self, results: Dict[str, Any], shot: int):
        dirs = [f"{shot}-shot"]
        # write evaluations
        self.write_file(print_json(results), "eval.json", dirs=dirs)
        self.write_file(print_json(results, indent=None), "eval.jsonl", append=True)

    def evaluate_results(self, all_results: List[GenerationOutput]):
        accs = np.array([r["acc"] for r in all_results])
        return {"acc-mean": accs.mean().item(), "acc-std": accs.std().item()}

    def write_file(
        self, obj: str, filename: str, dirs: List[str] = [], append: bool = False
    ):
        write_dir = join(self.output_dir, *dirs)
        os.makedirs(write_dir, exist_ok=True)
        write_file = join(write_dir, filename)
        write_mode = "appending" if append else "writing"
        logger.info(f"{write_mode} to {write_file}.")
        with open(write_file, "a" if append else "w") as f:
            f.write(obj)

    def read_run(self, shot: int):
        dirs = [f"{shot}-shot"]
        filename = f"eval.json"
        file_path = join(self.output_dir, *dirs, filename)
        if os.path.exists(file_path):
            return read_json_file(file_path)
        else:
            return None

    def write_config(self, conf):
        with open(join(self.output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config=conf, f=f)

    def write_train_examples(self, proc: BaseProcessor, indices: List[int], shot: int):
        for idx in indices:
            example = proc.convert_example_to_template_fields(proc.train_dataset[idx])
            self.write_file(
                print_json(example, indent=None),
                "train_examples.jsonl",
                dirs=[f"{shot}-shot"],
                append=True,
            )
