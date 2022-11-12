import glob
import logging
import os
import random
from collections import deque
from os.path import join
from typing import Dict, List, Optional, Union

import jsonlines
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import wandb
from rl import BaseEnvironment
from rl.agents.network import initialize_network
from rl.agents.replay import ReplayMemory, Transition
from rl.misc_utils import collate_summaries, parse_step_from_checkpoint, tailsum

logger = logging.getLogger(__name__)


class DQNAgent:
    def __init__(
        self,
        env: BaseEnvironment,
        output_dir: Optional[str] = None,
        overwrite_existing: bool = False,
        train_steps: int = 400,
        save_every: int = 400,
        eval_every: int = 400,
        val_rounds: int = 1,
        target_update_every: int = 10,
        optimization_steps_per_train_step=1,
        batch_size: int = 4,
        replay_memory_size: int = 1000,
        eps_start: float = 0.99,
        eps_end: float = 0.2,
        decay_steps: Optional[int] = None,
        max_grad_norm: float = 10.0,
        lr: Optional[float] = 1e-2,
        base_lr: Optional[float] = None,
        max_lr: Optional[float] = None,
        step_size_up: int = 100,
        step_size_down: Optional[int] = None,
        weight_decay: float = 1e-3,
        add_exit_action: bool = True,
        network_params: dict = {},
        cql_loss_weight: float = 0.0,
        load_transitions: Optional[Union[str, list[str]]] = None,
        offline_steps: int = 0,
    ):
        self.env = env
        self.output_dir = output_dir
        self.ckpt_dir = join(output_dir, "ckpts") if self.output_dir else None
        self.overwrite_existing = overwrite_existing
        self.curr_step = 0
        self.train_steps = train_steps
        self.save_every = save_every
        self.eval_every = eval_every
        self.val_rounds = val_rounds
        self.target_update_every = target_update_every
        self.cql_loss_weight = cql_loss_weight
        self.offline_steps = offline_steps

        if save_every % target_update_every != 0:
            raise Exception(
                "target_update_every should divide save_every"
                " to simplify model checkpointing logic"
            )

        if eval_every % save_every != 0:
            raise Exception(
                "save_every should divide eval_every"
                " to ensure best model can be loaded later"
            )

        self.optimization_steps_per_train_step = optimization_steps_per_train_step
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.replay_memory = ReplayMemory(replay_memory_size)

        self.eps_start = eps_start
        self.eps_end = eps_end
        if decay_steps is None:
            decay_steps = self.train_steps
        self.eps_decay = (eps_end / eps_start) ** (1 / (decay_steps - 1))

        self.mode = "train"
        self.add_exit_action = add_exit_action
        network_params["add_exit_action"] = add_exit_action

        self.policy_net = initialize_network(
            env.state_dim, env.action_dim, **network_params
        )
        self.target_net = initialize_network(
            env.state_dim, env.action_dim, requires_grad=False, **network_params
        )

        self.target_update()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr, weight_decay=weight_decay
        )

        if base_lr is None:
            assert lr is not None
            base_lr = max_lr = lr
            step_size_up = 100000

        if max_lr is None:
            raise Exception("max_lr cannot be None if base_lr is given.")

        logger.info(
            f"setting up scheduler with base_lr={base_lr}, "
            f"max_lr={max_lr}, step_size_up={step_size_up}, "
            f"step_size_down={step_size_down}"
        )

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr,
            max_lr,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            cycle_momentum=False,
        )
        if load_transitions is not None:
            self.replay_memory.load(load_transitions, env, add_exit_action)
        self.load_checkpoints()

    def load_checkpoints(self):
        if self.ckpt_dir is None:
            return

        if self.overwrite_existing:
            logger.info("train from scratch and overwrite existing checkpoints...")
            return

        replay_ckpts = {
            parse_step_from_checkpoint(f): f
            for f in glob.glob(join(self.ckpt_dir, "replay_*.ckpt"))
        }

        model_ckpts = {
            parse_step_from_checkpoint(f): f
            for f in glob.glob(join(self.ckpt_dir, "model_*.ckpt"))
        }
        optim_ckpts = {
            parse_step_from_checkpoint(f): f
            for f in glob.glob(join(self.ckpt_dir, "optim_*.ckpt"))
        }

        scheduler_ckpts = {
            parse_step_from_checkpoint(f): f
            for f in glob.glob(join(self.ckpt_dir, "sched_*.ckpt"))
        }

        ckpts_found = (
            set(replay_ckpts.keys())
            & set(model_ckpts.keys())
            & set(optim_ckpts.keys())
            & set(scheduler_ckpts.keys())
        )

        if not ckpts_found:
            logger.info("no existing checkpoints, train from scratch...")
            if 0 in replay_ckpts:
                logger.info("loading initial replay memory...")
                self.replay_memory = torch.load(replay_ckpts[0])
            return

        step = max(ckpts_found)
        logger.info(f"setting step={step}")
        self.curr_step = step
        logger.info(
            "loading replay memory, policy network and optimizer " f"from step={step}"
        )
        self.replay_memory = torch.load(replay_ckpts[step])
        self.policy_net.load_state_dict(torch.load(model_ckpts[step]))
        self.target_update()
        self.optimizer.load_state_dict(torch.load(optim_ckpts[step]))
        self.scheduler.load_state_dict(torch.load(scheduler_ckpts[step]))

    def target_update(self):
        logger.debug("updating target net with policy net parameters")
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoints(self):
        if self.ckpt_dir is None:
            return
        step = self.curr_step
        os.makedirs(self.ckpt_dir, exist_ok=True)

        logger.info(
            f"saving replay memory, policy network and optimizer for step={step}"
        )
        replay_ckpt_path = join(self.ckpt_dir, f"replay_{step}.ckpt")
        model_ckpt_path = join(self.ckpt_dir, f"model_{step}.ckpt")
        optim_ckpt_path = join(self.ckpt_dir, f"optim_{step}.ckpt")
        sched_ckpt_path = join(self.ckpt_dir, f"sched_{step}.ckpt")
        torch.save(self.replay_memory, replay_ckpt_path)
        torch.save(self.policy_net.state_dict(), model_ckpt_path)
        torch.save(self.optimizer.state_dict(), optim_ckpt_path)
        torch.save(self.scheduler.state_dict(), sched_ckpt_path)

    def choose_action(self, states, action_space):
        if self.mode == "train":
            # epsilon-greedy exploration with exp. decay
            eps = self.eps_start * self.eps_decay**self.curr_step
            eps = max(eps, self.eps_end)
            wandb.log(data={"epsilon": eps}, step=self.curr_step)
            rand_val = random.random()
            if rand_val < eps:
                # explore
                action_idx = random.choice(range(action_space.shape[0]))
                return action_idx

        # pick best action (argmax of Q values)
        Q_values = self.policy_net(states, action_space)
        wandb.log(
            data={
                "Q-mean": Q_values.mean().item(),
                "Q-std": Q_values.std().item(),
                "Q-max": Q_values.max().item(),
            },
            step=self.curr_step,
        )

        action_idx = Q_values.argmax().item()
        return action_idx

    def evaluate_action(self, states, action_space, action_idx):
        Q_values = self.policy_net(states, action_space)
        Q_value = Q_values[action_idx]

        # stop action (-1) is implicit
        # it happens when Q values for all other actions are negative
        if Q_value < 0.0:
            return Q_values, torch.tensor(0.0), True

        return Q_values, Q_value, False

    def rollout(self):
        env = self.env
        state = env.reset()
        terminal = False
        rewards = []
        past_states = [state]
        action_indices = []
        action_spaces = []
        device = self.policy_net.device

        while not terminal:
            states = torch.stack(past_states)
            action_space = env.action_space()
            if self.add_exit_action:
                action_space_ = torch.cat(
                    (action_space.to(device), self.policy_net.exit_action)
                )
            action_idx = self.choose_action(states, action_space_)
            next_state, reward, terminal = env.step(action_idx)

            rewards.append(reward)
            if not terminal:
                past_states.append(next_state)
            action_indices.append(action_idx)
            action_spaces.append(action_space)

        if (
            not len(rewards)
            == len(past_states)
            == len(action_indices)
            == len(action_spaces)
        ):
            raise Exception(
                f"should have len(rewards) ({len(rewards)}) == "
                f"len(past_states) ({len(past_states)}) == "
                f"len(action_indices) ({len(action_indices)}) =="
                f"len(action_spaces) ({len(action_spaces)})"
            )

        if self.mode == "train":
            # push non-terminal transitions to replay memory
            for i in range(len(rewards) - 1):
                states = torch.stack(past_states[: i + 1])
                action_idx = action_indices[i]
                action_space = action_spaces[i]
                next_states = torch.stack(past_states[: i + 2])
                next_action_space = action_spaces[i + 1]
                reward = torch.tensor(rewards[i])

                t = Transition(
                    states,
                    action_idx,
                    action_space,
                    next_states,
                    next_action_space,
                    reward,
                )
                self.replay_memory.push(t)

            # push terminal transition
            states = torch.stack(past_states)
            action_idx = action_indices[-1]
            action_space = action_spaces[-1]
            reward = torch.tensor(rewards[-1])
            t = Transition(states, action_idx, action_space, None, None, reward)
            self.replay_memory.push(t)

        rewards = torch.tensor(rewards)
        return rewards

    def evaluate_trajectory(self, trajectory: List[int]):
        # given a trajectory, evaluate how good it is
        env = self.env
        state = env.reset()
        terminal = False
        rewards = []
        past_states = [state]
        Q_action_space_hist = []
        Q_value_hist = []
        rewards = []

        for action_idx in trajectory:
            states = torch.stack(past_states)
            action_space = env.action_space()
            Q_values, Q_value, early_stopping = self.evaluate_action(
                states, action_space, action_idx
            )

            if early_stopping:
                next_state, reward, terminal = env.step(-1)
            else:
                next_state, reward, terminal = env.step(action_idx)

            Q_action_space_hist.append(Q_values.tolist())
            Q_value_hist.append(Q_value.item())
            rewards.append(reward)

            if not terminal:
                past_states.append(next_state)
            else:
                break

        return Q_action_space_hist, Q_value_hist, rewards

    def compute_conservative_loss(self, Q_pred_all, Q_pred):
        logsumexp = torch.logsumexp(Q_pred_all, dim=1)
        return (logsumexp - Q_pred).mean()

    def optimize(self):
        if len(self.replay_memory) < self.batch_size:
            logger.warning("replay memory empty, skip optimization step.")
            return

        self.optimizer.zero_grad()

        device = self.policy_net.device

        transitions = self.replay_memory.sample(k=self.batch_size)
        states = [t.states.to(device) for t in transitions]
        action_space = torch.stack([t.action_space for t in transitions]).to(device)

        action_idx = torch.tensor([t.action_idx for t in transitions], device=device)
        rewards = torch.tensor([t.reward for t in transitions], device=device)

        non_terminal_mask = torch.tensor(
            [t.next_states is not None for t in transitions],
            dtype=torch.bool,
            device=device,
        )

        if non_terminal_mask.sum() < 2:
            logger.info(
                "skipping optimization step since not enough "
                " non-terminal transitions are sampled."
            )
            return

        next_states = [
            t.next_states.to(device) for t in transitions if t.next_states is not None
        ]

        next_action_space = torch.stack(
            [
                t.next_action_space
                for t in transitions
                if t.next_action_space is not None
            ]
        ).to(device)

        if self.add_exit_action:
            p_exit_action = self.policy_net.exit_action
            p_exit_action_rep = p_exit_action.unsqueeze(0).repeat(
                action_space.shape[0], 1, 1
            )
            action_space = torch.cat((action_space, p_exit_action_rep), dim=1)

            t_exit_action = self.target_net.exit_action
            t_exit_action_rep = t_exit_action.unsqueeze(0).repeat(
                next_action_space.shape[0], 1, 1
            )
            next_action_space = torch.cat((next_action_space, t_exit_action_rep), dim=1)

        Q_pred_all = self.policy_net(states, action_space)
        Q_pred = Q_pred_all.gather(1, action_idx.unsqueeze(1)).squeeze()

        Q_target = torch.zeros(self.batch_size, device=device)
        Q_future_all = self.target_net(next_states, next_action_space)
        Q_future = Q_future_all.max(dim=1).values
        Q_target[non_terminal_mask] = Q_future
        Q_target = Q_target + rewards

        loss = F.l1_loss(Q_pred, Q_target, reduction="mean")

        cql_loss = self.compute_conservative_loss(Q_pred_all, Q_pred)
        loss = loss + self.cql_loss_weight * cql_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        wandb.log(
            data={"lr": self.scheduler.get_last_lr()[0], "loss": loss.item()},
            step=self.curr_step,
        )
        self.scheduler.step()

    def train(self):
        self.mode = "train"
        self.env.set_mode("train")
        self.policy_net.train()
        self.target_net.train()

        reward_hist = deque()
        with tqdm(total=self.train_steps - self.curr_step) as pbar:
            while self.curr_step < self.train_steps:
                # interact with environment
                if self.curr_step >= self.offline_steps:
                    rewards = self.rollout()
                    summary = self.env.summary()
                    summary["train-reward"] = rewards.sum().item()
                    reward_hist.append(summary["train-reward"])
                    if len(reward_hist) > 100:
                        reward_hist.popleft()
                    summary["avg-reward-last-100-episodes"] = sum(reward_hist) / len(
                        reward_hist
                    )
                    logger.debug(f"step {self.curr_step}, {summary}")
                    wandb.log(data=summary, step=self.curr_step)

                # optimize
                for _ in range(self.optimization_steps_per_train_step):
                    self.optimize()

                self.curr_step += 1
                pbar.update(1)

                if self.curr_step % self.target_update_every == 0:
                    self.target_update()

                if self.save_every > 0 and self.curr_step % self.save_every == 0:
                    self.save_checkpoints()

                if self.eval_every > 0 and self.curr_step % self.eval_every == 0:
                    self.eval(eval_mode="val")

        self.save_checkpoints()

    def eval(
        self,
        eval_mode: str = "val",
        eval_prefix: Optional[str] = None,
        load: str = "last",
    ):
        logger.info(
            "evaluating, "
            f"eval_mode = {eval_mode}, "
            f"eval_prefix = {eval_prefix}, "
            f"load = {load}."
        )

        self.mode = "eval"
        self.env.set_mode(eval_mode)

        if eval_mode == "test":
            if load == "best":
                self.load_best_model()
            elif load == "last":
                self.load_checkpoints()
            else:
                raise Exception(f"unknown load option {load}")

        self.policy_net.eval()
        self.target_net.eval()

        rounds = self.val_rounds if eval_mode == "val" else 1
        if eval_prefix is None:
            eval_prefix = eval_mode

        with torch.no_grad():
            summaries = []
            for round in range(rounds):
                rewards = self.rollout()
                future_rewards = tailsum(rewards)
                summary = self.env.summary()
                summary[f"{eval_mode}-reward"] = future_rewards[0].item()
                summaries.append(summary)

            summary = collate_summaries(summaries)
            summary["step"] = self.curr_step

            self.write_eval_results(eval_prefix, summary)
            logger.info(f"step {self.curr_step}, {summary}")
            if eval_mode == "val":
                wandb.log(data=summary, step=self.curr_step)
            else:
                wandb.log(data={f"{eval_prefix}-{k}": v for k, v in summary.items()})

        self.mode = "train"
        self.env.set_mode("train")

        self.policy_net.train()
        self.target_net.train()

    def write_eval_results(self, eval_mode: str, data: Dict):
        res_path = os.path.join(self.output_dir, f"{eval_mode}-results.jsonl")
        with jsonlines.open(res_path, mode="a") as f:
            f.write(data)

    def load_model_at_step(self, step):
        logger.info(f"loading policy net from step {step}")
        ckpt = join(self.ckpt_dir, f"model_{step}.ckpt")
        self.policy_net.load_state_dict(torch.load(ckpt))

    def load_best_model(self):
        val_res_path = join(self.output_dir, "val-results.jsonl")

        best_step = None
        best_perf = float("-inf")
        if os.path.exists(val_res_path):
            with jsonlines.open(val_res_path) as r:
                for data in r:
                    if "val-final-acc" not in data:
                        break
                    if data["val-final-acc"] > best_perf:
                        best_step = data["step"]
                        best_perf = data["val-final-acc"]

        if best_step is None:
            logger.warning(
                "cannot find dev history, not loading best checkpoint for testing"
            )
            return

        logger.info(f"loading best checkpoint from step {best_step} for testing")
        wandb.log(data={"best-val-final-acc": best_perf}, step=self.curr_step)
        self.load_model_at_step(best_step)
