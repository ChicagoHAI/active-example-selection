import logging
import random
from collections import deque
from typing import Union

import torch

from rl.environment import FewShotEnvironment, MultiDatasetEnvironment

logger = logging.getLogger(__name__)


class Transition:
    def __init__(
        self, states, action_idx, action_space, next_states, next_action_space, reward
    ):
        self.states = states
        self.action_idx = action_idx
        self.action_space = action_space
        self.next_states = next_states
        self.next_action_space = next_action_space
        self.reward = reward


class NamedTransition(Transition):
    def __init__(self, *args):
        super(NamedTransition, self).__init__(*args)


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, t: Transition):
        if t.action_space.shape[0] == 1 or (
            t.next_action_space is not None and t.next_action_space.shape[0] == 1
        ):
            logger.info("skip push")
            return
        self.memory.append(t)

    def sample(self, k=1):
        return random.sample(self.memory, k=k)

    def __len__(self):
        return len(self.memory)

    def load(
        self,
        path: Union[str, list[str]],
        env: Union[FewShotEnvironment, MultiDatasetEnvironment],
        add_exit_action: bool,
        exit_action_prob: float = 0.05,
    ):
        if isinstance(env, MultiDatasetEnvironment):
            envs = env.train_envs
            assert len(path) == len(envs)
            for p, e in zip(path, envs):
                self.load(p, e, add_exit_action, exit_action_prob)
            return

        assert isinstance(env, FewShotEnvironment)
        logger.info(f"loading transitions from path {path} with env {env}")
        transitions = torch.load(path)
        state_repr = env.state_repr
        action_repr = env.action_repr
        embedding_dim = env.model.embedding_dim

        def handle_state(r, t):
            if r in ("hidden_states_mean", "hidden_states_last"):
                t = t.view(2, embedding_dim)
                t = env.rand_proj(t)
                t = t.view(-1)
            elif r == "task_id":
                return torch.ones(1) * env.task_id
            return t

        def handle_action(r, t):
            A = t.shape[0]
            if r in ("hidden_states_mean", "hidden_states_last"):
                t = t.view(A, 2, embedding_dim)
                t = env.rand_proj(t)
                t = t.view(A, -1)
            return t

        for t in transitions:
            states = torch.stack(
                [
                    torch.cat([handle_state(r, state.get(r)) for r in state_repr])
                    for state in t.states
                ]
            )
            next_states = (
                torch.stack(
                    [
                        torch.cat([handle_state(r, state.get(r)) for r in state_repr])
                        for state in t.next_states
                    ]
                )
                if t.next_states
                else None
            )
            action_space = torch.cat(
                [handle_action(r, t.action_space[r]) for r in action_repr], dim=1
            )
            next_action_space = (
                torch.cat(
                    [handle_action(r, t.next_action_space[r]) for r in action_repr],
                    dim=1,
                )
                if t.next_action_space
                else None
            )
            self.push(
                Transition(
                    states,
                    t.action_idx,
                    action_space,
                    next_states,
                    next_action_space,
                    t.reward,
                )
            )
            if add_exit_action and random.random() < exit_action_prob:
                self.push(
                    Transition(
                        states,
                        action_space.shape[0],
                        action_space,
                        None,
                        None,
                        torch.tensor(0.0),
                    )
                )

        del transitions
