from rl.agents import DQNAgent, RandomAgent
from rl.environment import (
    BaseEnvironment,
    FewShotEnvironment,
    GPT3Environment,
    MultiDatasetEnvironment,
    ToyEnvironment,
    ToyRecurrentEnvironment,
)

ENVIRONMENTS = {
    "few-shot": FewShotEnvironment,
    "gpt3": GPT3Environment,
    "multi-dataset": MultiDatasetEnvironment,
}

AGENTS = {"dqn": DQNAgent, "random": RandomAgent}
