import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_sequence

logger = logging.getLogger(__name__)


def initialize_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.01)


def initialize_network(
    state_dim: int,
    action_dim: int,
    task_count: Optional[int] = None,
    linear: bool = False,
    hidden_dim: int = 16,
    recurrent: bool = False,
    dropout: float = 0.0,
    normalize: bool = True,
    tanh: bool = False,
    requires_grad: bool = True,
    add_exit_action: bool = True,
) -> nn.Module:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if linear:
        net = LinearNetwork(
            state_dim, action_dim, task_count, normalize, add_exit_action, device
        )
    elif recurrent:
        net = LSTMNetwork(
            state_dim,
            action_dim,
            hidden_dim,
            dropout,
            normalize,
            tanh,
            add_exit_action,
            device,
        )
    else:
        net = MLPNetwork(
            state_dim,
            action_dim,
            hidden_dim,
            task_count,
            dropout,
            normalize,
            tanh,
            add_exit_action,
            device,
        )

    initialize_weights(net)

    for p in net.parameters():
        p.requires_grad = requires_grad

    return net.to(device)


# Welford's online variance algorithm
# https://math.stackexchange.com/questions/198336/how-to-calculate-standard-deviation-with-streaming-inputs


class RunningNorm(nn.Module):
    def __init__(self, num_features: int):
        super(RunningNorm, self).__init__()
        self.register_buffer("count", torch.tensor(0))
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("M2", torch.zeros(num_features))
        self.register_buffer("eps", torch.tensor(1e-5))

    def track(self, x):
        x = x.detach().reshape(-1, x.shape[-1])

        self.count = self.count + x.shape[0]
        delta = x - self.mean
        self.mean.add_(delta.sum(dim=0) / self.count)
        self.M2.add_((delta * (x - self.mean)).sum(dim=0))

    def forward(self, x):
        # track stats only when training
        if self.training:
            self.track(x)

        # use stats to normalize current batch
        if self.count < 2:
            return x

        # biased var estimator
        var = self.M2 / self.count + self.eps
        x_normed = (x - self.mean) / torch.sqrt(var)
        return x_normed


class LinearNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        task_count: Optional[int],
        normalize: bool,
        add_exit_action: bool,
        device: str,
    ):
        super(LinearNetwork, self).__init__()
        input_dim = state_dim + action_dim
        self.net = nn.Linear(input_dim, 1)
        self.add_exit_action = add_exit_action
        self.normalize = normalize
        self.norm = RunningNorm(state_dim + action_dim)
        if add_exit_action:
            self.exit_action = nn.parameter.Parameter(
                torch.zeros(1, action_dim), requires_grad=True
            )
        if task_count is not None:
            self.add_task_embedding = True
            self.task_embedding = nn.Embedding(task_count, 1)
            torch.nn.init.zeros_(self.task_embedding.weight)
        else:
            self.add_task_embedding = False
        self.device = device

    def forward(self, states, action_space):
        # states: B x L* x D / L x D
        # action_space: B x A x D / A x D

        if isinstance(states, list):
            state = torch.stack([s[-1] for s in states])

        elif states.dim() == 2:
            assert action_space.dim() == 2
            state = states[-1].unsqueeze(0)
            action_space = action_space.unsqueeze(0)

        state = state.to(self.device)
        action_space = action_space.to(self.device)

        if action_space.shape[1] == 1:
            return torch.zeros(1, 1)

        if self.add_task_embedding:
            task_ids = state[:, 0].long()
            state = state[:, 1:]
            task_offset = self.task_embedding(task_ids).unsqueeze(1)
        else:
            task_offset = torch.zeros(1).to(state.device)

        action_count = action_space.shape[1]
        state_aligned = torch.stack([state] * action_count, dim=1)
        state_action_space = torch.cat((state_aligned, action_space), dim=2)

        if self.normalize:
            state_action_space = self.norm(state_action_space)

        return (self.net(state_action_space) + task_offset).squeeze()


class MLPNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        task_count: Optional[int],
        dropout: float,
        normalize: bool,
        tanh: bool,
        add_exit_action: bool,
        device: str,
    ):
        super(MLPNetwork, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # optionally normalize across actions (dimension 0)
        self.normalize = normalize
        self.norm = RunningNorm(state_dim + action_dim)
        self.tanh = tanh
        self.add_exit_action = add_exit_action
        if add_exit_action:
            self.exit_action = nn.parameter.Parameter(
                torch.zeros(1, action_dim), requires_grad=True
            )
        if task_count is not None:
            self.add_task_embedding = True
            self.task_embedding = nn.Embedding(task_count, 1)
            torch.nn.init.zeros_(self.task_embedding.weight)
        else:
            self.add_task_embedding = False

        input_dim = state_dim + action_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.device = device

    def forward(self, states, action_space):
        # states: B x L* x D / L x D
        # action_space: B x A x D / A x D

        if isinstance(states, list):
            state = torch.stack([s[-1] for s in states])
        elif states.dim() == 3:
            state = states[:, 0, :]
        elif states.dim() == 2:
            assert action_space.dim() == 2
            state = states[-1].unsqueeze(0)
            action_space = action_space.unsqueeze(0)

        state = state.to(self.device)
        action_space = action_space.to(self.device)

        if action_space.shape[1] == 1:
            return torch.zeros(1, 1)

        if self.add_task_embedding:
            task_ids = state[:, 0].long()
            state = state[:, 1:]
            task_offset = self.task_embedding(task_ids).unsqueeze(1)
        else:
            task_offset = torch.zeros(1).to(state.device)

        action_count = action_space.shape[1]
        state_aligned = torch.stack([state] * action_count, dim=1)
        state_action_space = torch.cat((state_aligned, action_space), dim=2)

        if self.normalize:
            state_action_space = self.norm(state_action_space)

        state_action_space = self.dropout(state_action_space)
        x = self.input_layer(state_action_space)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        logits = self.output_layer(x)
        if self.tanh:
            logits = torch.tanh(logits)
        return (logits + task_offset).squeeze()


class LSTMNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        dropout: float,
        normalize: bool,
        tanh: bool,
        add_exit_action: bool,
        device: str,
    ):
        super(LSTMNetwork, self).__init__()
        if state_dim == 0:
            logger.warning(
                "got state_dim=0, LSTM not processing any state information..."
            )

        self.normalize = normalize
        self.norm = RunningNorm(state_dim)
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.mlp = MLPNetwork(
            hidden_dim,
            action_dim,
            hidden_dim,
            dropout,
            normalize,
            tanh,
            add_exit_action,
            device,
        )
        if add_exit_action:
            self.exit_action = self.mlp.exit_action
        self.device = device

    def forward(self, states, action_space):
        if isinstance(states, list):
            states = [s.to(self.device) for s in states]
            if self.normalize:
                states = [self.norm(s) for s in states]
            states = pack_sequence(states, enforce_sorted=False)

        elif states.dim() == 2:
            assert action_space.dim() == 2
            states = states.to(self.device)
            states = states.unsqueeze(0)
            action_space = action_space.unsqueeze(0)
            if self.normalize:
                states = self.norm(states)

        _, (state, _) = self.lstm(states)

        return self.mlp(state.permute(1, 0, 2), action_space)
