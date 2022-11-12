import logging
from typing import List

from omegaconf import DictConfig
from tqdm.auto import tqdm

from prompting import BaseProcessor, GPT2Wrapper
from prompting.misc_utils import entropy
from prompting.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class GreedyStrategy(BaseStrategy):
    def run_strategy(
        self, proc: BaseProcessor, model: GPT2Wrapper, shot: int, evaluate: bool = True
    ):
        logger.info(f"{self.class_name} - shot={shot}")

        if shot == 0:
            train_indices = []
        else:
            # retrieve previous indices and make a copy
            prev = self(proc, model, shot - 1, evaluate=False)
            train_indices = prev["train_indices"][:]

            assert len(train_indices) + 1 == shot
            new_idx = self.acquisition(model, proc, train_indices)
            train_indices.append(new_idx)

        simple_result = {
            "shot": shot,
            "train_indices": train_indices,
        }

        if evaluate:
            prompts, cali_prompts = proc.create_prompts(train_indices)
            outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
            eval_result = proc.extract_predictions(outputs)
            self.write_result(eval_result, shot)
            self.write_train_examples(proc, train_indices, shot)
            simple_result["acc"] = eval_result["acc"]

        return simple_result

    def acquisition(
        self, model: GPT2Wrapper, proc: BaseProcessor, train_indices: List[int]
    ) -> int:
        raise NotImplementedError


class MaxEntropyStrategy(GreedyStrategy):
    def acquisition(
        self, model: GPT2Wrapper, proc: BaseProcessor, train_indices: List[int]
    ) -> int:
        # acquire the training example with the highest pred. entropy
        train_prompts, cali_train_prompts = proc.create_prompts(
            train_indices, split="train"
        )
        outputs = model.complete_all(
            train_prompts, calibration_prompts=cali_train_prompts
        )

        new_idx = max(range(len(outputs)), key=lambda i: entropy(outputs[i].probs))
        return new_idx


class OracleStrategy(GreedyStrategy):
    def __init__(self, conf: DictConfig):
        super().__init__(conf)
        self.max_train_dataset_size = 100
        self.test_subset = None

    def acquisition(
        self, model: GPT2Wrapper, proc: BaseProcessor, train_indices: List[int]
    ) -> int:
        # acquire the training example that results in the highest dev acc.
        # intractable to search everything, use small train & dev subsets
        best_idx = -1
        best_acc = float("-inf")
        for idx, example in tqdm(enumerate(proc.train_dataset)):
            eval_prompts, cali_prompts = proc.create_prompts(
                train_indices + [idx], split="val"
            )
            outputs = model.complete_all(eval_prompts, calibration_prompts=cali_prompts)
            eval_result = proc.extract_predictions(outputs, split="val")
            if eval_result["acc"] > best_acc:
                best_idx = idx
                best_acc = eval_result["acc"]

        return best_idx
