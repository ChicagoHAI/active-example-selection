import glob
import logging
import os
import random
from os.path import join
from typing import Optional

import torch
from tqdm.auto import tqdm

from rl import FewShotEnvironment
from rl.agents.replay import NamedTransition
from rl.misc_utils import parse_step_from_checkpoint

logger = logging.getLogger(__name__)


class RandomAgent:
    def __init__(
        self,
        env: FewShotEnvironment,
        output_dir: Optional[str] = None,
        train_steps: int = 1000,
        save_every: int = 200,
    ):
        self.env = env
        self.train_steps = train_steps
        self.save_every = save_every
        self.transitions = []
        self.ckpt_dir = join(output_dir, "ckpts") if output_dir else None
        self.curr_step = 0
        self.load_checkpoints()

    def choose_action(self, states, action_space):
        return random.randrange(self.env.action_count())

    def load_checkpoints(self):
        if self.ckpt_dir is None:
            return

        replay_ckpts = {
            parse_step_from_checkpoint(f): f
            for f in glob.glob(join(self.ckpt_dir, "transitions_*.ckpt"))
        }

        ckpts_found = set(replay_ckpts.keys())

        if not ckpts_found:
            logger.info("no existing checkpoints, train from scratch...")
            if 0 in replay_ckpts:
                logger.info("loading initial replay memory...")
                self.transitions = torch.load(replay_ckpts[0])
            return

        step = max(ckpts_found)
        logger.info(f"setting step={step}")
        self.curr_step = step
        logger.info(f"loading transitions from step={step}")
        self.transitions = torch.load(replay_ckpts[step])

    def save_checkpoints(self):
        if self.ckpt_dir is None:
            return
        step = self.curr_step
        os.makedirs(self.ckpt_dir, exist_ok=True)

        logger.info(f"saving transitions from for step={step}")
        t_ckpt_path = join(self.ckpt_dir, f"transitions_{step}.ckpt")
        torch.save(self.transitions, t_ckpt_path)

    def rollout(self):
        env = self.env
        state = env.reset()
        terminal = False
        rewards = []
        past_states = [state]
        action_indices = []
        action_spaces = []

        while not terminal:
            states = past_states
            action_space = env.action_space()
            action_idx = self.choose_action(states, action_space)
            next_state, reward, terminal = env.step(action_idx)

            rewards.append(reward)
            if not terminal:
                past_states.append(next_state)
            action_indices.append(action_idx)
            action_spaces.append(action_space)

        for i in range(len(rewards) - 1):
            states = past_states[: i + 1]
            action_idx = action_indices[i]
            action_space = action_spaces[i]
            next_states = past_states[: i + 2]
            next_action_space = action_spaces[i + 1]
            reward = torch.tensor(rewards[i])

            t = NamedTransition(
                states,
                action_idx,
                action_space,
                next_states,
                next_action_space,
                reward,
            )
            self.transitions.append(t)

        # push terminal transition
        states = past_states
        action_idx = action_indices[-1]
        action_space = action_spaces[-1]
        reward = torch.tensor(rewards[-1])
        t = NamedTransition(states, action_idx, action_space, None, None, reward)
        self.transitions.append(t)

    def train(self):
        self.env.set_mode("train")
        assert self.env.named

        with tqdm(total=self.train_steps - self.curr_step) as pbar:
            while self.curr_step < self.train_steps:
                self.rollout()

                self.curr_step += 1
                pbar.update(1)

                if self.save_every > 0 and self.curr_step % self.save_every == 0:
                    self.save_checkpoints()
