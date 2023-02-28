# Active Example Selection

This repository contains code for the paper
> **[Active Example Selection for In-Context Learning.](https://arxiv.org/abs/2211.04486)**  
> Yiming Zhang, Shi Feng and Chenhao Tan  
> Empirical Methods in Natural Language Processing (EMNLP), 2022.  

## Setup

1. Clone the repository.
2. Create env with conda: `conda create -n active-example-selection python=3.9`.
3. Install pytorch with version >= 1.10.2.
4. Run `pip install .` (add the `-e` flag if you plan to make changes).

## Baselines

We include code for the following 5 methods for example selection that we
  compare with in the paper:

- **random**: random demonstration examples,
- **max-entropy**: max-entropy active learning baseline,
- **best-of-k**: best demonstration out of k random sets using a dev set,
- **oracle**: iteratively picking the best demonstration example using a dev set,
- **reordering**: reordering demonstration examples using the *Global Entropy*
  method by [Lu et al., 2022](https://arxiv.org/abs/2104.08786).

This command runs the 4-shot **random** baseline on GPT-2 medium with
[calibration](https://arxiv.org/abs/2102.09690):
> `python src/prompting/main.py prompting_configs/baseline-gpt2.yaml`

The following command runs the 4-shot **random** baseline on GPT-3 ada, and make
  sure the field `api_key_file` points to a file with your
  OpenAI API key:
> `python src/prompting/main.py prompting_configs/baseline-gpt3.yaml`

## Training Example Selection Policies

Example selection policies are trained on random trajectories sampled offline.
To generate these random trajectories (e.g., on *AGNews*), run
> `python src/rl/main.py rl_configs/random-agent.yaml`

Then, to train an *AGNews* example selection policy, run
> `python src/rl/main.py rl_configs/agnews-same-task.yaml`

After generating random trajectories similarly for *Amazon*, *SST-2* and *TREC*,
run the following to train an example selection policy on these three datasets
and transferred to *AGNews*:
> `python src/rl/main.py rl_configs/agnews-transfer.yaml`

## Citation

```bibtex
@inproceedings{zhang-etal-2022-active,
    title = "Active Example Selection for In-Context Learning",
    author = "Zhang, Yiming and Feng, Shi and Tan, Chenhao",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.622",
    pages = "9134--9148",
    abstract = "With a handful of demonstration examples, large-scale language models demonstrate strong capability to perform various tasks by in-context learning from these examples, without any fine-tuning. We demonstrate that in-context learning performance can be highly unstable across samples of examples, indicating the idiosyncrasies of how language models acquire information. We formulate example selection for in-context learning as a sequential decision problem, and propose a reinforcement learning algorithm for identifying generalizable policies to select demonstration examples. For GPT-2, our learned policies demonstrate strong abilities of generalizing to unseen tasks in training, with a 5.8{\%} improvement on average. Examples selected from our learned policies can even achieve a small improvement on GPT-3 Ada. However, the improvement diminishes on larger GPT-3 models, suggesting emerging capabilities of large language models.",
}
```
