import logging
import random
from collections import defaultdict
from itertools import permutations

from prompting import BaseProcessor, GPT2Wrapper
from prompting.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class RandomStrategy(BaseStrategy):
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"RandomStrategy - shot={shot}")
        train_indices = random.sample(range(len(proc.train_dataset)), k=shot)
        prompts, cali_prompts = proc.create_prompts(train_indices)
        outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)

        simple_result = {
            "shot": shot,
            "train_indices": train_indices,
            "acc": eval_result["acc"],
        }
        return simple_result


class RandomClassBalancedStrategy(BaseStrategy):
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"RandomClassBalancedStrategy - shot={shot}")
        train_indices_by_class = defaultdict(list)

        for i, example in enumerate(proc.train_dataset):
            label = proc.convert_example_to_template_fields(example)["label_text"]
            train_indices_by_class[label].append(i)

        for indices in train_indices_by_class.values():
            random.shuffle(indices)

        label_list = list(train_indices_by_class.keys())
        random.shuffle(label_list)

        train_indices = []
        for i in range(shot):
            label = label_list[i % len(label_list)]
            train_indices.append(train_indices_by_class[label].pop())

        random.shuffle(train_indices)

        prompts, cali_prompts = proc.create_prompts(train_indices)
        outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)

        simple_result = {
            "shot": shot,
            "train_indices": train_indices,
            "acc": eval_result["acc"],
        }
        return simple_result


class BestPermStrategy(BaseStrategy):
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"BestPermStrategy - shot={shot}")

        perm_accs = []
        train_original = random.sample(range(len(proc.train_dataset)), k=shot)

        for train_indices in permutations(train_original):
            prompts, cali_prompts = proc.create_prompts(train_indices, split="val")
            outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
            eval_result = proc.extract_predictions(outputs, split="val")
            perm_accs.append((train_indices, eval_result["acc"]))

        best_indices = max(perm_accs, key=lambda t: t[1])[0]
        prompts, cali_prompts = proc.create_prompts(best_indices)
        outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)

        simple_result = {
            "shot": shot,
            "train_indices": best_indices,
            "acc": eval_result["acc"],
            "perm_accs": perm_accs,
        }
        return simple_result


class BestOfKStrategy(BaseStrategy):
    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"BestOfKStrategy - shot={shot}")
        K = 10
        best_indices = None
        best_val_acc = -1.0

        for i in range(K):
            train_indices = random.sample(range(len(proc.train_dataset)), k=shot)
            prompts, cali_prompts = proc.create_prompts(train_indices, split="val")
            outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
            eval_result = proc.extract_predictions(outputs, split="val")
            if eval_result["acc"] > best_val_acc:
                best_val_acc = eval_result["acc"]
                best_indices = train_indices

        prompts, cali_prompts = proc.create_prompts(best_indices)
        outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)

        simple_result = {
            "shot": shot,
            "train_indices": best_indices,
            "acc": eval_result["acc"],
        }
        return simple_result
