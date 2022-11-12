import logging
import math
import random
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from prompting import PROCESSORS, GPT2Wrapper, PromptTooLongError
from prompting.models import GPT3Wrapper
from rl.misc_utils import normalized_entropy, tensor_stats

logger = logging.getLogger(__name__)


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self):
        pass

    @property
    def state(self) -> torch.Tensor:
        return torch.empty(0)

    def set_mode(self, mode: str):
        if mode not in ("train", "val", "test"):
            raise Exception(f"unknown mode {mode}")
        self.mode = mode

    @abstractmethod
    def action_count(self) -> int:
        pass

    @abstractmethod
    def action_space(self):
        pass

    @abstractmethod
    def step(self, idx: int):
        pass

    @abstractmethod
    def summary(self):
        pass


class RandomProjection:
    def __init__(self, do_reduce: bool, in_features: int, out_features: int):
        self.do_reduce = do_reduce
        if do_reduce:
            logger.info("initializing random projection matrix")
            rng = torch.Generator().manual_seed(42)
            self.proj_mat = torch.normal(
                mean=0.0,
                std=1.0 / math.sqrt(out_features),
                size=(in_features, out_features),
                generator=rng,
            )

    def __call__(self, X: torch.Tensor):
        if self.do_reduce:
            return X @ self.proj_mat
        return X


class FewShotEnvironment(BaseEnvironment):
    def __init__(
        self,
        seed: int = 42,
        task: str = "sst-2",
        task_id: Optional[int] = None,
        model: str = "gpt2-xl",
        model_kwargs: Dict = {},
        cache: Optional[str] = None,
        redis_kwargs: Dict = {},
        max_feasible_steps: int = 99,
        max_steps: int = 4,
        step_mode: str = "fixed",
        state_repr: Union[str, List[str]] = [],
        action_repr: Union[str, List[str]] = ["logits"],
        cost_per_training_example: float = 0.0,
        reward_max_accuracy: float = 0.0,
        terminate_on_repeated_example: bool = False,
        skip_on_repeated_example: bool = False,
        train_subset_samples: Optional[int] = None,
        eval_subset_samples: Optional[int] = None,
        reduce_embedding_dim: bool = False,
        reduce_to: Optional[int] = None,
        named_features: str = None,
        dataset_mode: str = "labeled",
    ):

        cache_module = None

        self.task = task
        self.task_id = task_id
        processor = PROCESSORS[task](seed, dataset_mode)
        model = GPT2Wrapper(
            model, cache_module, **processor.model_kwargs, **model_kwargs
        )
        self.model = model
        self.proc = processor

        self.train_size = len(self.proc.train_dataset)
        self.val_size = len(self.proc.val_dataset)
        self.test_size = len(self.proc.test_dataset)
        logger.info(f"train_dataset size={self.train_size}")
        logger.info(f"val_dataset size={self.val_size}")
        logger.info(f"test_dataset size={self.test_size}")

        self.train_indices = []
        self.max_feasible_steps = max_feasible_steps
        self.max_steps = max_steps

        if step_mode not in ["fixed", "random"]:
            raise Exception(f"Unknown step_mode {step_mode}")
        self.step_mode = step_mode

        self.reduce_embedding_dim = reduce_embedding_dim
        if reduce_embedding_dim and reduce_to is None:
            raise Exception(
                "reduce_to cannot be None if " "reduce_embedding_dim is True"
            )

        if reduce_embedding_dim:
            self.rand_proj = RandomProjection(True, model.embedding_dim, reduce_to)
            embedding_dim = reduce_to
        else:
            self.rand_proj = RandomProjection(False, -1, -1)
            embedding_dim = model.embedding_dim

        self.named = named_features

        if isinstance(state_repr, str):
            self.state_repr = [state_repr]
        else:
            self.state_repr = list(state_repr)

        if task_id is not None:
            self.state_repr = ["task_id"] + self.state_repr

        self.state_dim = 0
        for _state_repr in self.state_repr:
            if _state_repr == "task_id":
                # task_id does not count towards feature dim
                pass
            elif _state_repr in ("hidden_states_mean", "hidden_states_last"):
                self.state_dim += embedding_dim * 2
            elif _state_repr in (
                "curr_step",
                "max_steps",
                "last_label",
                "perplexity",
                "val_dist_entropy",
            ):
                self.state_dim += 1
            elif _state_repr == "val_dist":
                self.state_dim += len(processor.labels)
            elif _state_repr == "val_dist_stats":
                self.state_dim += 3
            else:
                raise Exception(f"Unknown state representation {_state_repr}")

        if isinstance(action_repr, str):
            self.action_repr = [action_repr]
        else:
            self.action_repr = action_repr

        self.action_dim = 0

        for _action_repr in self.action_repr:
            if _action_repr in ("logits", "probs"):
                self.action_dim += len(processor.labels)
            elif _action_repr in ("logits_stats", "probs_stats"):
                self.action_dim += 3
            elif _action_repr in ("entropy", "perplexity"):
                self.action_dim += 1
            elif _action_repr in ("hidden_states_mean", "hidden_states_last"):
                self.action_dim += embedding_dim * 2
            else:
                raise Exception(f"Unknown action representation {_action_repr}")

        self.prev_acc = 0.0
        self.acc_history = []

        self.cost_per_training_example = cost_per_training_example
        self.reward_max_accuracy = reward_max_accuracy
        self.terminate_on_repeated_example = terminate_on_repeated_example
        self.skip_on_repeated_example = skip_on_repeated_example

        if train_subset_samples is None:
            train_subset_samples = len(self.proc.train_dataset)
        self.train_subset_samples = train_subset_samples
        logger.info(
            f"sampling {train_subset_samples} out of {self.train_size} for each"
            " training episode"
        )

        if eval_subset_samples is None:
            eval_subset_samples = len(self.proc.val_dataset)
        self.eval_subset_samples = eval_subset_samples
        logger.info(
            f"sampling {eval_subset_samples} out of {self.val_size} for each"
            " validation episode"
        )

    def __repr__(self):
        return self.task

    @property
    def zero_shot_acc(self):
        if self.mode == "train":
            return self.zero_shot_acc_train
        elif self.mode == "val":
            return self.zero_shot_acc_val
        elif self.mode == "test":
            return self.zero_shot_acc_test
        raise Exception

    @property
    def state(self):
        if not self.state_repr:
            return torch.empty(0)

        train_split = "train" if self.mode in ("train", "test") else "val"
        training_prompt = self.proc.get_training_prompt(
            self.train_indices, train_split=train_split
        )
        if not training_prompt:
            training_prompt = "\n"

        # no need to do calibration here, since not computing probs
        output = self.model.complete_all_with_hidden_states(
            [training_prompt], do_calibrate=False
        )[0]

        val_dist = torch.tensor(self.previous_eval_result["class-dist"])
        val_dist = val_dist / val_dist.sum()

        repr_tensors = []
        for state_repr in self.state_repr:
            if state_repr == "task_id":
                repr_tensors.append(torch.ones(1) * self.task_id)
            elif state_repr == "curr_step":
                repr_tensors.append(torch.ones(1) * len(self.train_indices))
            elif state_repr == "max_steps":
                repr_tensors.append(torch.ones(1) * self.max_steps_episode)

            elif state_repr == "hidden_states_mean":
                hsm = output.hidden_states.mean(dim=1)
                hsm_proj = self.rand_proj(hsm).reshape(-1)
                repr_tensors.append(hsm_proj)
            elif state_repr == "hidden_states_last":
                hsl = output.hidden_states[:, -1, :]
                hsl_proj = self.rand_proj(hsl).reshape(-1)
                repr_tensors.append(hsl_proj)
            elif state_repr == "perplexity":
                repr_tensors.append(output.perplexity.unsqueeze(0))
            elif state_repr == "last_label":
                if not self.train_indices:
                    repr_tensors.append(torch.ones(1) * -1)
                else:
                    last_example = self.proc.train_dataset[self.train_indices[-1]]
                    last_label = self.proc.get_label_idx(last_example)
                    repr_tensors.append(torch.ones(1) * last_label)
            elif state_repr == "val_dist":
                repr_tensors.append(val_dist)
            elif state_repr == "val_dist_entropy":
                repr_tensors.append(normalized_entropy(val_dist).unsqueeze(0))
            elif state_repr == "val_dist_stats":
                repr_tensors.append(tensor_stats(val_dist))
            else:
                raise Exception(f"unknown state repr {state_repr}")

        if self.named:
            return {r: t for r, t in zip(self.state_repr, repr_tensors)}
        return torch.cat(repr_tensors)

    @cached_property
    def zero_shot_acc_train(self):
        acc = self.evaluate_accuracy([], "train")
        self.zero_shot_train_eval_result = self.previous_eval_result
        return acc

    @cached_property
    def zero_shot_acc_val(self):
        acc = self.evaluate_accuracy([], "val")
        self.zero_shot_val_eval_result = self.previous_eval_result
        return acc

    @cached_property
    def zero_shot_acc_test(self):
        return self.evaluate_accuracy([], "test")

    def set_max_steps_episode(self) -> int:
        if self.mode in ["train", "val"]:
            if self.step_mode == "fixed":
                self.max_steps_episode = self.max_steps
            elif self.step_mode == "random":
                self.max_steps_episode = random.randint(1, self.max_feasible_steps)
        else:
            # test mode
            if self.step_mode == "fixed":
                self.max_steps_episode = self.max_steps
            elif self.step_mode == "random":
                self.max_steps_episode = self.max_feasible_steps

        logger.debug(f"setting max_steps_episode={self.max_steps_episode}")

    def reset(self) -> torch.Tensor:
        self.prev_acc = self.zero_shot_acc
        if self.mode == "train":
            selected_indices = random.sample(
                range(self.train_size),
                k=self.train_subset_samples,
            )
            self.train_subset_indices = selected_indices
            self.train_subset = [self.proc.train_dataset[i] for i in selected_indices]

        if self.mode == "val":
            selected_indices = random.sample(
                range(self.val_size),
                k=self.eval_subset_samples,
            )
            self.eval_subset_indices = selected_indices
            self.eval_subset = [self.proc.val_dataset[i] for i in selected_indices]
            self.previous_eval_result = self.zero_shot_val_eval_result
        else:
            _ = self.zero_shot_acc_train
            self.previous_eval_result = self.zero_shot_train_eval_result

        self.set_max_steps_episode()
        self.train_indices = []
        self.acc_history = []
        return self.state

    def get_curr_split(self):
        if self.mode == "train":
            assert (
                len(self.train_subset)
                == len(self.train_subset_indices)
                == self.train_subset_samples
            )
            split, split_indices = self.train_subset, self.train_subset_indices
        elif self.mode == "val":
            assert (
                len(self.eval_subset)
                == len(self.eval_subset_indices)
                == self.eval_subset_samples
            )
            split, split_indices = self.eval_subset, self.eval_subset_indices
        elif self.mode == "test":
            split, split_indices = self.proc.train_dataset, list(range(self.train_size))
        else:
            raise Exception

        if self.skip_on_repeated_example:
            split = [
                example
                for i, example in enumerate(split)
                if split_indices[i] not in self.train_indices
            ]
            split_indices = [i for i in split_indices if i not in self.train_indices]
        return split, split_indices

    def evaluate_accuracy(self, train_indices: List[int], mode: str):
        if mode in ("train", "val"):
            train_split = mode
            split = "val" if mode == "train" else "train"
            eval_prompts, eval_cali_prompts = self.proc.create_prompts(
                train_indices, train_split=train_split, split=split
            )
            outputs = self.model.complete_all_with_hidden_states(
                eval_prompts, calibration_prompts=eval_cali_prompts
            )
            eval_result = self.proc.extract_predictions(outputs, split=split)
            self.previous_eval_result = eval_result

        elif mode == "test":
            # get val-dist
            eval_prompts, eval_cali_prompts = self.proc.create_prompts(
                train_indices, split="val"
            )
            outputs = self.model.complete_all_with_hidden_states(
                eval_prompts, calibration_prompts=eval_cali_prompts
            )
            eval_result = self.proc.extract_predictions(outputs, split="val")
            self.previous_eval_result = eval_result
            eval_prompts, eval_cali_prompts = self.proc.create_prompts(
                train_indices, split="test"
            )
            outputs = self.model.complete_all_with_hidden_states(
                eval_prompts, calibration_prompts=eval_cali_prompts
            )
            eval_result = self.proc.extract_predictions(outputs, split="test")
        else:
            raise Exception

        return eval_result["acc"]

    def action_count(self):
        split, _ = self.get_curr_split()
        return len(split)

    def action_space(self):
        split, split_indices = self.get_curr_split()

        train_split = "train" if self.mode in ("train", "test") else "val"
        train_prompts, train_cali_prompts = self.proc.create_prompts(
            self.train_indices,
            train_split=train_split,
            split="custom",
            custom_split=split,
        )

        try:
            outputs = self.model.complete_all_with_hidden_states(
                train_prompts, calibration_prompts=train_cali_prompts
            )
        except PromptTooLongError:
            # when we fail here, we return a dummy action space and
            # set max_steps_episode to -1 to guarantee the error is handled later
            self.max_feasible_steps = max(len(self.train_indices), self.max_steps)
            self.max_steps_episode = -1
            logger.warning(f"setting max_feasible_steps to {self.max_feasible_steps}")
            logger.warning(f"setting max_steps_episode to -1")
            return torch.zeros(1, self.action_dim)

        actions = []
        for output in outputs:
            action = []
            for action_repr in self.action_repr:
                if action_repr == "logits":
                    action.append(output.logits)
                elif action_repr == "logits_stats":
                    action.append(tensor_stats(output.logits))
                elif action_repr == "probs":
                    action.append(output.probs)
                elif action_repr == "probs_stats":
                    action.append(tensor_stats(output.probs))
                elif action_repr == "entropy":
                    action.append(normalized_entropy(output.probs).unsqueeze(0))
                elif action_repr == "hidden_states_mean":
                    hsm = output.hidden_states.mean(dim=1)
                    hsm_proj = self.rand_proj(hsm).reshape(-1)
                    action.append(hsm_proj)
                elif action_repr == "hidden_states_last":
                    hsl = output.hidden_states[:, -1, :]
                    hsl_proj = self.rand_proj(hsl).reshape(-1)
                    action.append(hsl_proj)
                elif action_repr == "perplexity":
                    action.append(output.perplexity.unsqueeze(0))
                else:
                    raise Exception(f"{action_repr}")

            if self.named:
                actions.append({r: t for r, t in zip(self.action_repr, action)})
            else:
                actions.append(torch.cat(action))

        if self.named:
            # stack by individual representation
            stacked = {
                r: torch.stack([a[r] for a in actions]) for r in self.action_repr
            }
            return stacked
        return torch.stack(actions)

    @property
    def reward(self):
        acc = self.evaluate_accuracy(self.train_indices, self.mode)
        reward = acc - self.prev_acc - self.cost_per_training_example
        self.prev_acc = acc
        self.acc_history.append(acc)

        return reward

    def step(self, action_idx: int):
        # Check termination conditions
        _, split_indices = self.get_curr_split()

        if action_idx < 0 or action_idx > len(split_indices):
            raise Exception(f"invalid action index {action_idx}")

        if action_idx == len(split_indices):
            logger.debug(
                f"model early terminates, made "
                f"{len(self.train_indices)} / {self.max_steps_episode} steps."
            )
            reward = self.reward_max_accuracy * (
                max(self.acc_history, default=self.zero_shot_acc) - self.zero_shot_acc
            )
            return None, reward, True

        train_idx = split_indices[action_idx]
        if self.terminate_on_repeated_example and train_idx in self.train_indices:
            logger.debug(
                f"terminating on repeated example {train_idx} in {self.train_indices}"
            )
            reward = self.reward_max_accuracy * (
                max(self.acc_history) - self.zero_shot_acc
            )
            return None, reward, True
        self.train_indices.append(train_idx)

        is_terminal = len(self.train_indices) >= self.max_steps_episode
        try:
            if self.max_steps_episode < 0:
                raise PromptTooLongError("rethrow error caught in action_space()")
            reward = self.reward
        except PromptTooLongError:
            # have to terminate due to long prompt
            logger.warning(
                f"These indices caused PromptTooLongError: {self.train_indices}"
            )
            self.train_indices.pop()
            self.max_feasible_steps = max(len(self.train_indices), self.max_steps)
            logger.warning(f"setting max_feasible_steps to {self.max_feasible_steps}")

            reward = 0.0
            is_terminal = True

        if is_terminal:
            reward += self.reward_max_accuracy * (
                max(self.acc_history) - self.zero_shot_acc
            )

        return self.state, reward, is_terminal

    def summary(self):
        acc_history = [self.zero_shot_acc] + self.acc_history
        return {
            f"{self.mode}-indices": self.train_indices,
            f"{self.mode}-final-acc": self.prev_acc,
            f"{self.mode}-max-acc": max(self.acc_history)
            if self.acc_history
            else self.zero_shot_acc,
            f"{self.mode}-acc-history": acc_history,
        }


class GPT3Environment(BaseEnvironment):
    def __init__(
        self,
        seed: int = 42,
        task: str = "sst-2",
        task_id: Optional[int] = None,
        model: str = "ada",
        api_key_file: str = "",
        model_kwargs: Dict = {},
        calibrate: bool = False,
        max_feasible_steps: int = 99,
        max_steps: int = 4,
        step_mode: str = "fixed",
        state_repr: Union[str, List[str]] = [],
        action_repr: Union[str, List[str]] = ["logits"],
        cost_per_training_example: float = 0.0,
        reward_max_accuracy: float = 0.0,
        terminate_on_repeated_example: bool = False,
        train_subset_samples: Optional[int] = None,
        eval_subset_samples: Optional[int] = None,
        named_features: str = None,
    ):
        dataset_mode = "labeled-gpt3"
        self.task = task
        self.task_id = task_id
        processor = PROCESSORS[task](seed, dataset_mode)
        model = GPT3Wrapper(
            model,
            api_key_file,
            **processor.model_kwargs,
            calibrate=calibrate,
            **model_kwargs,
        )
        self.model = model
        self.proc = processor

        self.train_size = len(self.proc.train_dataset)
        self.val_size = len(self.proc.val_dataset)
        self.test_size = len(self.proc.test_dataset)
        logger.info(f"train_dataset size={self.train_size}")
        logger.info(f"val_dataset size={self.val_size}")
        logger.info(f"test_dataset size={self.test_size}")

        self.train_indices = []
        self.max_feasible_steps = max_feasible_steps
        self.max_steps = max_steps

        if step_mode not in ["fixed", "random"]:
            raise Exception(f"Unknown step_mode {step_mode}")
        self.step_mode = step_mode

        self.named = named_features

        if isinstance(state_repr, str):
            self.state_repr = [state_repr]
        else:
            self.state_repr = list(state_repr)

        if task_id is not None:
            self.state_repr = ["task_id"] + self.state_repr

        self.state_dim = 0
        for _state_repr in self.state_repr:
            if _state_repr == "task_id":
                # task_id does not count towards feature dim
                pass
            elif _state_repr in (
                "curr_step",
                "max_steps",
                "last_label",
            ):
                self.state_dim += 1
            else:
                raise Exception(f"Unknown state representation {_state_repr}")

        if isinstance(action_repr, str):
            self.action_repr = [action_repr]
        else:
            self.action_repr = action_repr

        self.action_dim = 0

        for _action_repr in self.action_repr:
            if _action_repr == "probs":
                self.action_dim += len(processor.labels)
            elif _action_repr == "probs_stats":
                self.action_dim += 3
            elif _action_repr == "entropy":
                self.action_dim += 1
            else:
                raise Exception(f"Unknown action representation {_action_repr}")

        self.prev_acc = 0.0
        self.acc_history = []

        self.cost_per_training_example = cost_per_training_example
        self.reward_max_accuracy = reward_max_accuracy
        self.terminate_on_repeated_example = terminate_on_repeated_example

        if train_subset_samples is None:
            train_subset_samples = len(self.proc.train_dataset)
        self.train_subset_samples = train_subset_samples
        logger.info(
            f"sampling {train_subset_samples} out of {self.train_size} for each"
            " training episode"
        )

        if eval_subset_samples is None:
            eval_subset_samples = len(self.proc.val_dataset)
        self.eval_subset_samples = eval_subset_samples
        logger.info(
            f"sampling {eval_subset_samples} out of {self.val_size} for each"
            " validation episode"
        )

    def __repr__(self):
        return self.task

    @property
    def state(self):
        if not self.state_repr:
            return torch.empty(0)

        train_split = "train" if self.mode in ("train", "test") else "val"

        repr_tensors = []
        for state_repr in self.state_repr:
            if state_repr == "task_id":
                repr_tensors.append(torch.ones(1) * self.task_id)
            elif state_repr == "curr_step":
                repr_tensors.append(torch.ones(1) * len(self.train_indices))
            elif state_repr == "max_steps":
                repr_tensors.append(torch.ones(1) * self.max_steps_episode)
            elif state_repr == "last_label":
                if not self.train_indices:
                    repr_tensors.append(torch.ones(1) * -1)
                else:
                    last_example = self.proc.train_dataset[self.train_indices[-1]]
                    last_label = self.proc.get_label_idx(last_example)
                    repr_tensors.append(torch.ones(1) * last_label)
            else:
                raise Exception(f"unknown state repr {state_repr}")

        if self.named:
            return {r: t for r, t in zip(self.state_repr, repr_tensors)}
        return torch.cat(repr_tensors)

    def set_max_steps_episode(self) -> int:
        if self.mode in ["train", "val"]:
            if self.step_mode == "fixed":
                self.max_steps_episode = self.max_steps
            elif self.step_mode == "random":
                self.max_steps_episode = random.randint(1, self.max_feasible_steps)
        else:
            # test mode
            if self.step_mode == "fixed":
                self.max_steps_episode = self.max_steps
            elif self.step_mode == "random":
                self.max_steps_episode = self.max_feasible_steps

        logger.debug(f"setting max_steps_episode={self.max_steps_episode}")

    def reset(self) -> torch.Tensor:
        self.prev_acc = 0.0
        if self.mode == "train":
            selected_indices = random.sample(
                range(self.train_size),
                k=self.train_subset_samples,
            )
            self.train_subset_indices = selected_indices
            self.train_subset = [self.proc.train_dataset[i] for i in selected_indices]

        if self.mode == "val":
            selected_indices = random.sample(
                range(self.val_size),
                k=self.eval_subset_samples,
            )
            self.eval_subset_indices = selected_indices
            self.eval_subset = [self.proc.val_dataset[i] for i in selected_indices]

        self.set_max_steps_episode()
        self.train_indices = []
        self.acc_history = []
        return self.state

    def get_curr_split(self):
        if self.mode == "train":
            assert (
                len(self.train_subset)
                == len(self.train_subset_indices)
                == self.train_subset_samples
            )
            return self.train_subset, self.train_subset_indices
        elif self.mode == "val":
            assert (
                len(self.eval_subset)
                == len(self.eval_subset_indices)
                == self.eval_subset_samples
            )
            return self.eval_subset, self.eval_subset_indices
        elif self.mode == "test":
            return self.proc.train_dataset, list(range(self.train_size))
        else:
            raise Exception

    def evaluate_accuracy(self, train_indices: List[int], mode: str):
        if mode in ("train", "val"):
            train_split = mode
            split = "val" if mode == "train" else "train"
            eval_prompts, eval_cali_prompts = self.proc.create_prompts(
                train_indices, train_split=train_split, split=split
            )
            outputs = self.model.complete_all(
                eval_prompts, calibration_prompts=eval_cali_prompts
            )
            eval_result = self.proc.extract_predictions(outputs, split=split)

        elif mode == "test":
            eval_prompts, eval_cali_prompts = self.proc.create_prompts(
                train_indices, split="test"
            )
            outputs = self.model.complete_all(
                eval_prompts, calibration_prompts=eval_cali_prompts
            )
            eval_result = self.proc.extract_predictions(outputs, split="test")
        else:
            raise Exception

        return eval_result["acc"]

    def action_count(self):
        split, _ = self.get_curr_split()
        return len(split)

    def action_space(self):
        split, _ = self.get_curr_split()
        train_split = "train" if self.mode in ("train", "test") else "val"
        train_prompts, train_cali_prompts = self.proc.create_prompts(
            self.train_indices,
            train_split=train_split,
            split="custom",
            custom_split=split,
        )

        outputs = self.model.complete_all(
            train_prompts, calibration_prompts=train_cali_prompts
        )

        actions = []
        for output in outputs:
            action = []
            for action_repr in self.action_repr:
                if action_repr == "probs":
                    action.append(output.probs)
                elif action_repr == "probs_stats":
                    action.append(tensor_stats(output.probs))
                elif action_repr == "entropy":
                    action.append(normalized_entropy(output.probs).unsqueeze(0))
                else:
                    raise Exception(f"{action_repr}")

            if self.named:
                actions.append({r: t for r, t in zip(self.action_repr, action)})
            else:
                actions.append(torch.cat(action))

        if self.named:
            # stack by individual representation
            stacked = {
                r: torch.stack([a[r] for a in actions]) for r in self.action_repr
            }
            return stacked
        return torch.stack(actions)

    def reward(self, is_terminal):
        if is_terminal:
            acc = self.evaluate_accuracy(self.train_indices, self.mode)
        else:
            acc = 0.0
        reward = acc - self.prev_acc - self.cost_per_training_example
        self.prev_acc = acc
        self.acc_history.append(acc)

        return reward

    def step(self, action_idx: int):
        # Check termination conditions
        _, split_indices = self.get_curr_split()

        if action_idx < 0 or action_idx > len(split_indices):
            raise Exception(f"invalid action index {action_idx}")

        if action_idx == len(split_indices):
            logger.debug(
                f"model early terminates, made "
                f"{len(self.train_indices)} / {self.max_steps_episode} steps."
            )
            return None, self.reward(True), True

        train_idx = split_indices[action_idx]
        self.train_indices.append(train_idx)

        is_terminal = len(self.train_indices) >= self.max_steps_episode
        return self.state, self.reward(is_terminal), is_terminal

    def summary(self):
        acc_history = self.acc_history
        return {
            f"{self.mode}-indices": self.train_indices,
            f"{self.mode}-final-acc": self.prev_acc,
            f"{self.mode}-max-acc": max(self.acc_history)
            if self.acc_history
            else self.zero_shot_acc,
            f"{self.mode}-acc-history": acc_history,
        }


class MultiDatasetEnvironment(BaseEnvironment):
    def __init__(
        self,
        train_env_configs: List[DictConfig],
        test_env_config: DictConfig,
        **kwargs: Dict,
    ):
        common_config = DictConfig(kwargs)
        train_env_configs = [
            OmegaConf.merge(common_config, c) for c in train_env_configs
        ]
        test_env_config = OmegaConf.merge(common_config, test_env_config)

        self.train_envs = [
            FewShotEnvironment(task_id=task_id, **task_kwargs)
            for task_id, task_kwargs in enumerate(train_env_configs)
        ]

        self.test_task = FewShotEnvironment(task_id=0, **test_env_config)

        train_task_strs = [t.task for t in self.train_envs]
        test_task_str = self.test_task.task
        logger.info(
            f"initialized sub-environments, training on {train_task_strs}, "
            f"testing on {test_task_str}"
        )

        action_dims = set([self.test_task.action_dim])
        state_dims = set([self.test_task.state_dim])
        for env in self.train_envs:
            action_dims.add(env.action_dim)
            state_dims.add(env.state_dim)

        if not len(action_dims) == len(state_dims) == 1:
            raise Exception("check if all tasks have the same state & action reprs.")

        self.action_dim = action_dims.pop()
        self.state_dim = state_dims.pop()

        self.curr_task = None
        self.val_task_idx = 0

    def reset(self):
        if self.mode in "train":
            self.curr_task = random.choice(self.train_envs)
            self.val_task_idx = 0
        elif self.mode == "test":
            self.curr_task = self.test_task
            self.val_task_idx = 0
        elif self.mode == "val":
            task_idx = self.val_task_idx % len(self.train_envs)
            self.curr_task = self.train_envs[task_idx]
            self.val_task_idx += 1
        else:
            raise Exception

        logger.debug(f"set_mode to {self.mode} for task = {self.curr_task}")
        self.curr_task.set_mode(self.mode)
        return self.curr_task.reset()

    @property
    def state(self) -> torch.Tensor:
        return self.curr_task.state

    def action_count(self):
        return self.curr_task.action_count()

    def action_space(self):
        return self.curr_task.action_space()

    def step(self, idx: int):
        return self.curr_task.step(idx)

    def summary(self):
        return self.curr_task.summary()


class ToyEnvironment(BaseEnvironment):
    def __init__(self, *args, **kwargs):
        self.state_dim = 0
        self.action_dim = 100
        self._action_space = F.one_hot(torch.arange(0, 100)).float()
        self.max_steps = 4
        self.indices = []

    def reset(self):
        self.indices = []
        return self.state

    def action_count(self):
        return self._action_space.shape[0]

    def action_space(self):
        return self._action_space

    def step(self, idx: int):
        if idx == 3:
            reward = 0.1
        else:
            reward = -0.1
        self.indices.append(idx)
        is_terminal = len(self.indices) >= self.max_steps
        return self.state, reward, is_terminal

    def summary(self):
        return {f"{self.mode}-indices": self.indices}


class ToyRecurrentEnvironment(BaseEnvironment):
    def __init__(self, *args, **kwargs):
        self.state_dim = 10
        self.action_dim = 10
        self._action_space = torch.randn((100, 10))
        self.max_steps = 4
        self.indices = []
        self.wins = [[1, 3, 5], [2, 4, 6], [33, 44, 55], [88, 77, 66]]
        self._state = torch.rand(10)

    @property
    def state(self):
        return self._state

    def reset(self):
        self.indices = []
        return self.state

    def action_count(self):
        return self._action_space.shape[0]

    def action_space(self):
        return self._action_space

    def step(self, idx: int):
        if idx in self.wins[len(self.indices)]:
            reward = 0.1
        else:
            reward = -0.1
        self.indices.append(idx)
        is_terminal = len(self.indices) >= self.max_steps
        return self.state, reward, is_terminal

    def summary(self):
        return {f"{self.mode}-indices": self.indices}
