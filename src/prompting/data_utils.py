from functools import cached_property
from logging import getLogger
from typing import Any, Dict, List

from datasets import load_dataset

from prompting.misc_utils import deterministic_hash, deterministic_random
from prompting.models import GenerationOutput, Prompt

logger = getLogger(__name__)


class BaseProcessor:
    @cached_property
    def dataset(self):
        return load_dataset(self.dataset_name)

    @cached_property
    def train_split(self):
        return self.dataset["train"]

    @cached_property
    def val_split(self):
        return self.dataset["validation"]

    @cached_property
    def test_split(self):
        return self.dataset["test"]

    def generate_datasets(self, seed: int, mode: str):
        # train -> train_train + train_test + test_train
        # test -> test
        logger.info(f"generating datasets using seed {seed}")

        if len(self.train_split) < 1200:
            raise Exception("dataset is too small")

        shuffled = deterministic_random(seed).sample(
            range(len(self.train_split)),
            k=len(self.train_split),
        )

        labeled_train_indices = shuffled[0:100]
        labeled_val_indices = shuffled[100:200]

        if mode.startswith("unlabeled-custom-"):
            unlabeled_size = int(mode.split("-")[-1])
            assert len(shuffled) >= 200 + unlabeled_size

            unlabeled_train_indices = shuffled[200 : 200 + unlabeled_size]
            logger.info(f"set unlabeled size to {unlabeled_size}")
        else:
            unlabeled_train_indices = shuffled[200:1200]

        if mode == "labeled" or mode == "labeled-gpt3":
            self.train_dataset = [self.train_split[i] for i in labeled_train_indices]
            self.val_dataset = [self.train_split[i] for i in labeled_val_indices]
        else:
            self.train_dataset = [self.train_split[i] for i in unlabeled_train_indices]
            self.val_dataset = [self.train_split[i] for i in labeled_val_indices]

        test_size = len(self.test_split)

        if mode == "labeled-gpt3":
            test_sample_size = 1000
        else:
            test_sample_size = 10000

        if test_size > test_sample_size:
            logger.info(f"subsampling test set to {test_sample_size} examples")
            test_size = test_sample_size

        # always use the same test set
        test_indices = deterministic_random(42).sample(
            range(len(self.test_split)), k=test_size
        )
        self.test_dataset = [self.test_split[i] for i in test_indices]

        dataset_hash = deterministic_hash(
            f"{labeled_train_indices}{labeled_val_indices}{unlabeled_train_indices}"
        )
        logger.info(f"dataset hash={dataset_hash}")

    def get_label_idx(self, example: Dict):
        return example["label"]

    @cached_property
    def prompt_start(self):
        return ""

    def convert_example_to_template_fields(self, example: Dict):
        return example

    def fill_train_template(self, example: Dict):
        fields = self.convert_example_to_template_fields(example)
        return self.train_template.format(**fields)

    def fill_eval_template(self, example: Dict):
        fields = self.convert_example_to_template_fields(example)
        return self.eval_template.format(**fields)

    def fill_calibration_template(self, calibration_example: Dict):
        return self.eval_template.format(**calibration_example)

    def get_training_prompt(self, train_indices: List[int], train_split: str = "train"):
        if train_split not in ("train", "val"):
            raise Exception

        data = self.train_dataset if train_split == "train" else self.val_dataset
        training_prompt = self.prompt_start + "".join(
            [self.fill_train_template(data[idx]) for idx in train_indices]
        )
        return training_prompt

    def get_probing_prompt(self, train_indices: List[int]):
        data = self.train_dataset
        template = "input: {text}\ntype: {label_text}"
        prompt = "\n\n".join(
            [
                template.format(**self.get_probing_fields(data[idx]))
                for idx in train_indices
            ]
            + ["input:"]
        )
        return prompt

    def get_probing_fields(self, example: Dict):
        return self.convert_example_to_template_fields(example)

    def create_prompts(
        self,
        train_indices: List[int],
        train_split: str = "train",
        split: str = "test",
        custom_split: List[Dict] = None,
    ):
        prompts = []
        cali_prompts = []

        training_prompt = self.get_training_prompt(
            train_indices, train_split=train_split
        )

        if split != "custom" and custom_split is not None:
            raise Exception("custom_split should be None unless split=custom")

        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        elif split == "custom":
            if custom_split is None:
                raise Exception("split=custom but custom_split is None")
            dataset = custom_split
        else:
            raise Exception(f"Unknown split {split}")

        for eval_example in dataset:
            prompt = Prompt(training_prompt + self.fill_eval_template(eval_example))
            prompts.append(prompt)

        for calibrate_example in self.calibration_examples:
            calibrate_prompt = Prompt(
                training_prompt + self.fill_calibration_template(calibrate_example),
                is_calibration_prompt=True,
            )
            cali_prompts.append(calibrate_prompt)

        return prompts, cali_prompts

    def extract_predictions(
        self,
        outputs: List[GenerationOutput],
        split: str = "test",
        custom_split: List[Dict] = None,
    ) -> Dict[str, Any]:

        if split != "custom" and custom_split is not None:
            raise Exception("custom_split should be None unless split=custom")

        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        elif split == "custom":
            if custom_split is None:
                raise Exception("split=custom but custom_split is None")
            dataset = custom_split
        else:
            raise Exception(f"Unknown split {split}")

        if len(outputs) != len(dataset):
            raise Exception(
                f"number of predictions ({len(outputs)}) != "
                f"number of examples ({len(dataset)})"
            )
        preds = []
        labels = []
        class_dist = [0] * len(self.labels)

        for output, test_example in zip(
            outputs, map(self.convert_example_to_template_fields, dataset)
        ):
            for i, label in enumerate(self.labels):
                if output.completion.startswith(label):
                    pred = label
                    class_dist[i] += 1
                    break
            else:
                raise Exception(
                    f"Completion {output.completion} does not match any of the labels {self.labels}"
                )
            preds.append(pred)
            labels.append(test_example["label_text"])
        acc = sum(
            1 if pred == label else 0 for pred, label in zip(preds, labels)
        ) / len(labels)

        return_dict = {
            "acc": acc,
            "preds": preds,
            "labels": labels,
            "class-dist": class_dist,
        }
        return return_dict

    def subsample_train_dataset(self, samples: int, seed: int = 42):
        logger.info(f"Subsampling training dataset to {samples} rows.")
        shuffled = self.dataset["train"].shuffle(seed=seed)
        self.train_dataset = [shuffled[i] for i in range(samples)]

    @property
    def calibration_examples(self):
        return [
            {"text": "N/A"},
            {"text": "[MASK]"},
            {"text": ""},
        ]


class SST2Processor(BaseProcessor):
    def __init__(self, seed: int, mode: str):
        self.dataset_name = "SetFit/sst2"
        self.train_template = "Review: {text}\n" "Sentiment: {label_text}\n\n"
        self.eval_template = "Review: {text}\n" "Sentiment:"

        self.labels = ["positive", "negative"]
        self.model_kwargs = {"labels": self.labels}
        self.generate_datasets(seed, mode)

    def parse_probe_example(self, s: str):
        return {"text": s, "label_text": "positive"}


class AGNewsProcessor(BaseProcessor):
    def __init__(self, seed: int, mode: str):
        self.dataset_name = "ag_news"
        self.val_split = None
        self.train_template = "Article: {text}\n" "Answer: {label_text}\n\n"
        self.eval_template = "Article: {text}\n" "Answer:"

        self.labels = ["World", "Sports", "Business", "Technology"]
        self.model_kwargs = {"labels": self.labels}
        self.generate_datasets(seed, mode)

    def convert_example_to_template_fields(self, example: Dict):
        label_text = self.labels[example["label"]]
        return {"text": example["text"], "label_text": label_text}

    def parse_probe_example(self, s: str):
        return {"text": s, "label": 0}


class TRECProcessor(BaseProcessor):
    def __init__(self, seed: int, mode: str):
        self.dataset_name = "trec"
        self.val_split = None
        self.prompt_start = (
            "Classify the questions based on whether their answ"
            "er type is a Number, Location, Person, Description"
            ", Entity, or Abbreviation.\n\n"
        )

        self.train_template = "Question: {text}\n" "Answer Type: {label_text}\n\n"
        self.eval_template = "Question: {text}\n" "Answer Type:"

        self.labels = [
            "Description",
            "Entity",
            "Abbreviation",
            "Person",
            "Number",
            "Location",
        ]
        self.model_kwargs = {"labels": self.labels}
        self.generate_datasets(seed, mode)

    def get_label_idx(self, example: Dict):
        return example["label-coarse"]

    def convert_example_to_template_fields(self, example):
        label_text = self.labels[example["label-coarse"]]
        return {"text": example["text"], "label_text": label_text}

    def parse_probe_example(self, s: str):
        return {"text": s, "label-coarse": 0}


class AmazonProcessor(BaseProcessor):
    def __init__(self, seed: int, mode: str):
        self.dataset_name = "amazon_polarity"
        self.val_split = None
        self.train_template = (
            "Title: {title}\n" "Review: {review}\n" "Sentiment: {label_text}\n\n"
        )
        self.eval_template = "Title: {title}\n" "Review: {review}\n" "Sentiment:"

        self.labels = ["negative", "positive"]
        self.model_kwargs = {"labels": self.labels}
        self.generate_datasets(seed, mode)

    def convert_example_to_template_fields(self, example):
        label_text = self.labels[example["label"]]
        return {
            "title": example["title"],
            "review": example["content"],
            "label_text": label_text,
        }

    @property
    def calibration_examples(self):
        return [
            {"title": "N/A", "review": "N/A"},
            {"title": "[MASK]", "review": "[MASK]"},
            {"title": "", "review": ""},
        ]

    def get_probing_fields(self, example: Dict):
        label_text = self.labels[example["label"]]
        return {
            "text": example["title"] + "\n" + example["content"],
            "label_text": label_text,
        }

    def parse_probe_example(self, s: str):
        if "\n" in s:
            title, review = s.split("\n", maxsplit=1)
        else:
            title, review = "", s
        return {"title": title, "content": review, "label": 0}
