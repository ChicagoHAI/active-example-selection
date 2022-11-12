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
`python src/prompting/main.py prompting_configs/baseline-gpt2.yaml`.

The following command runs the 4-shot **random** baseline on GPT-3 ada, and make
  sure the field `api_key_file: ./openai_api_key` points to a file with your
  OpenAI API key:
`python src/prompting/main.py prompting_configs/baseline-gpt3.yaml`.

## Training Example Selection Policies

Example selection policies are trained on random trajectories sampled offline.
To generate these random trajectories (e.g., on *AGNews*), run
`python src/rl/main.py rl_configs/random-agent.yaml`.

Then, to train an *AGNews* example selection policy, run
`python src/rl/main.py rl_configs/agnews-same-task.yaml`.

After generating random trajectories similarly for *Amazon*, *SST-2* and *TREC*,
run the following to train an example selection policy on these three datasets
and transferred to *AGNews*:
`python src/rl/main.py rl_configs/agnews-transfer.yaml`.
