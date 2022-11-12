import logging
import random
from itertools import permutations

import torch
from transformers import StoppingCriteria

from prompting import BaseProcessor, GPT2Wrapper
from prompting.misc_utils import entropy
from prompting.models import to_device
from prompting.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class stop(StoppingCriteria):
    def __call__(self, iids, _):
        assert iids.shape[0] == 1
        return iids[0][-2:].tolist() == [4906, 25]


def probe(model: GPT2Wrapper, prompt: str):

    batch = model.tokenizer(prompt, return_tensors="pt")
    input_length = batch["input_ids"].shape[1]
    batch = to_device(batch, model.device)
    output = model.model.generate(
        **batch,
        max_length=input_length + 128,
        output_hidden_states=True,
        do_sample=True,
        no_repeat_ngram_size=3,
        temperature=2.0,
        stopping_criteria=[stop()],
    )
    return model.tokenizer.decode(output[0, input_length:], skip_special_tokens=True)


class GlobalEntropyOrderingStrategy(BaseStrategy):
    """Lu et al. - https://arxiv.org/pdf/2104.08786.pdf"""

    def run_strategy(self, proc: BaseProcessor, model: GPT2Wrapper, shot: int):
        logger.info(f"GlobalEntropyStrategy - shot={shot}")
        train_indices = random.sample(range(len(proc.train_dataset)), k=shot)
        # get all permutations
        perms = permutations(train_indices)

        probe_raws = []
        probe_examples = []
        for perm in permutations(train_indices):
            probe_raw = probe(model, proc.get_probing_prompt(perm))
            probe_str = probe_raw.strip().split("type:")[0]
            probe_item = proc.parse_probe_example(probe_str)
            probe_raws.append(probe_raw)
            probe_examples.append(probe_item)

        perm_to_entropy = {}
        for perm in permutations(train_indices):
            prompts, cali_prompts = proc.create_prompts(
                perm, split="custom", custom_split=probe_examples
            )
            outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
            eval_result = proc.extract_predictions(
                outputs, split="custom", custom_split=probe_examples
            )
            label_counts = torch.tensor(eval_result["class-dist"])
            class_distribution = label_counts / label_counts.sum()
            global_entropy = entropy(class_distribution)
            perm_to_entropy[perm] = global_entropy

        best_perm = max(perm_to_entropy.keys(), key=lambda k: perm_to_entropy[k])

        prompts, cali_prompts = proc.create_prompts(best_perm)
        outputs = model.complete_all(prompts, calibration_prompts=cali_prompts)
        eval_result = proc.extract_predictions(outputs)
        self.write_result(eval_result, shot)

        simple_result = {
            "shot": shot,
            "train_indices": best_perm,
            "acc": eval_result["acc"],
        }
        return simple_result
