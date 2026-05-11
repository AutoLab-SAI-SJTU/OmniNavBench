import torch
import torch.nn as nn
from torch.distributions import Normal


def _get_activation(name: str):
    """Map string to torch activation."""
    name = (name or "").lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class ActorCritic(nn.Module):
    """Minimal Actor-Critic used for H1 locomotion inference.

    Matches the older rsl_rl-style interface (obs -> MLP mean output).
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        # Keep kwargs handling explicit to surface unexpected parameters.
        if kwargs:
            # Mirror aliengo behavior: log and continue to keep compatibility minimal.
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        super().__init__()
        actor_hidden_dims = actor_hidden_dims or [1024, 512, 256]
        critic_hidden_dims = critic_hidden_dims or [1024, 512, 256]
        act = _get_activation(activation)

        def _build_mlp(in_dim: int, hidden_dims, out_dim: int, add_tanh: bool = False):
            layers = []
            last = in_dim
            for i, h in enumerate(hidden_dims):
                layers.append(nn.Linear(last, h))
                layers.append(act)
                last = h
            layers.append(nn.Linear(last, out_dim))
            if add_tanh:
                layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        self.actor = _build_mlp(num_actor_obs, actor_hidden_dims, num_actions, add_tanh=False)
        self.critic = _build_mlp(num_critic_obs, critic_hidden_dims, 1, add_tanh=False)

        # Action noise parameter, aligns with legacy checkpoints.
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

    def act_inference(self, obs: torch.Tensor, **kwargs):
        """Deterministic action (mean) for inference."""
        return self.actor(obs)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        return self.critic(critic_observations)

    @property
    def action_mean(self):
        return self.distribution.mean if self.distribution is not None else None

    @property
    def action_std(self):
        return self.distribution.stddev if self.distribution is not None else None

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1) if self.distribution is not None else None

