# source: https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
# https://arxiv.org/pdf/2208.06193
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from dataclasses import asdict, dataclass, field
import os, sys
from pathlib import Path
import random
import uuid

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from diffusion_ql.agents.diffusion import Diffusion
from diffusion_ql.agents.model import MLP
from diffusion_ql.agents.helpers import EMA
from diffusion_ql.utils.data_sampler import Data_Sampler

APP_DIR= os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(APP_DIR)
sys.path.append(APP_DIR+'/rlhf')

# reward model
from rlhf.utils import replace_dataset_reward, load_reward_models

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

# like in the original repository
ENV_PARAMS = {
    'kitchen-complete-v0':{'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 250 , 'gn': 9.0,  'top_k': 2},
    'kitchen-partial-v0': {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'kitchen-mixed-v0':   {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0}
}


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cpu"
    env: str = "kitchen-mixed-v0"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(10)  # How often (time steps) we evaluate - default int(5e3)
    n_episodes: int = 1  # How many episodes run during evaluation - default 10
    max_timesteps: int = int(100)  # Max time steps to run environment - default int(1e6)
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = None  # Model load file name, None doesn't load anything
    # Diffusion-QL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    max_q_backup=False
    eta=0.005
    beta_schedule='linear'
    n_timesteps=100
    ema_decay=0.995
    step_start_ema=1000
    update_ema_every=5
    lr=3e-4
    lr_decay=False
    lr_maxt=1000
    grad_norm=10
    # Wandb logging
    project: str = "Uni-RLHF"
    group: str = "DiffusionQL"
    name: str = "exp"
    reward_model_paths: list = field(default_factory=lambda: [
        "../../rlhf/model_logs/kitchen-mixed-v0/mlp/epoch_100_query_25_len_200_seed_888/models/scripted_comparative_reward_mlp.pt",
        "../../rlhf/model_logs/kitchen-mixed-v0/mlp/epoch_100_query_20_len_100_seed_888/models/scripted_evaluative_reward_mlp.pt",
    ])
    keypoint_predictor_path: str = "../../rlhf/model_logs/kitchen-mixed-v0/mlp/epoch_100_query_2_len_50_seed_888/models/human_keypoint_predictor_mlp.pt"
    def __post_init__(self):
        # self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        self.name = f"{self.name}-{self.env}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# class ReplayBuffer:
#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         buffer_size: int,
#         device: str = "cpu",
#     ):
#         self._buffer_size = buffer_size
#         self._pointer = 0
#         self._size = 0

#         self._states = torch.zeros(
#             (buffer_size, state_dim), dtype=torch.float32, device=device
#         )
#         self._actions = torch.zeros(
#             (buffer_size, action_dim), dtype=torch.float32, device=device
#         )
#         self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
#         self._next_states = torch.zeros(
#             (buffer_size, state_dim), dtype=torch.float32, device=device
#         )
#         self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
#         self._device = device

#     def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
#         return torch.tensor(data, dtype=torch.float32, device=self._device)

#     # Loads data in d4rl format, i.e. from Dict[str, np.array].
#     def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
#         if self._size != 0:
#             raise ValueError("Trying to load data into non-empty replay buffer")
#         n_transitions = data["observations"].shape[0]
#         if n_transitions > self._buffer_size:
#             raise ValueError(
#                 "Replay buffer is smaller than the dataset you are trying to load!"
#             )
#         self._states[:n_transitions] = self._to_tensor(data["observations"])
#         self._actions[:n_transitions] = self._to_tensor(data["actions"])
#         self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
#         self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
#         self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
#         self._size += n_transitions
#         self._pointer = min(self._size, n_transitions)

#         print(f"Dataset size: {n_transitions}")

#     def sample(self, batch_size: int) -> TensorBatch:
#         indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
#         states = self._states[indices]
#         actions = self._actions[indices]
#         rewards = self._rewards[indices]
#         next_states = self._next_states[indices]
#         dones = self._dones[indices]
#         return [states, actions, rewards, next_states, dones]

#     def add_transition(self):
#         # Use this method to add new data into the replay buffer during fine-tuning.
#         # I left it unimplemented since now we do not do fine-tuning.
#         raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        # id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)
    
    def train(self, batch: TensorBatch) -> Dict[str, float]:
        batch_size = len(batch)
        state, action, next_state, reward, not_done = batch

        """ Q Training """
        current_q1, current_q2 = self.critic(state, action)

        if self.max_q_backup:
            next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
            next_action_rpt = self.ema_model(next_state_rpt)
            target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
            target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            target_q = torch.min(target_q1, target_q2)
        else:
            next_action = self.ema_model(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

        target_q = (reward + not_done * self.discount * target_q).detach()

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        """ Policy Training """
        bc_loss = self.actor.loss(action, state)
        new_action = self.actor(state)

        q1_new_action, q2_new_action = self.critic(state, new_action)
        if np.random.uniform() > 0.5:
            q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
        else:
            q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        actor_loss = bc_loss + self.eta * q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0: 
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()


        """ Step Target network """
        if self.step % self.update_ema_every == 0:
            self.step_ema()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.step += 1

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        """ Log """
        logs = {
            'bc_loss': actor_loss.item(), 
            'ql_loss': bc_loss.item(), 
            'actor_loss': q_loss.item(), 
            'critic_loss': critic_loss.item(),
            'target_q_mean': target_q.mean().item(),
        }
        if self.grad_norm > 0:
            logs['actor_grad_norm'] = actor_grad_norms.max().item()
            logs['critic_grad_norm'] = critic_grad_norms.max().item()

        return logs

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)
    config.group = config.env
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    reward_models, reward_model_types = load_reward_models(config.env, state_dim, action_dim, config.reward_model_paths, config.device)
    dataset = replace_dataset_reward(dataset, reward_models, reward_model_types, device=config.device)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = Data_Sampler(dataset, config.device)
    # replay_buffer = ReplayBuffer(
    #     state_dim,
    #     action_dim,
    #     config.buffer_size,
    #     config.device,
    # )
    # replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "device": config.device,
        "discount": config.discount,
        "tau": config.tau,
        "max_q_backup": config.max_q_backup,
        "eta": config.eta,
        "beta_schedule": config.beta_schedule,
        "n_timesteps": config.n_timesteps,
        "ema_decay": config.ema_decay,
        "step_start_ema": config.step_start_ema,
        "update_ema_every": config.update_ema_every,
        "lr": config.lr,
        "lr_decay": config.lr_decay,
        "lr_maxt": config.lr_maxt,
        "grad_norm": ENV_PARAMS[config.env]["gn"],
    }

    print("---------------------------------------")
    print(f"Training DiffusionQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = Diffusion_QL(**kwargs)

    if config.load_model:
        policy_file = Path(config.load_model)
        trainer.load_model(torch.load(policy_file))
    actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.step)
        if t % 5000 == 0:
            print(log_dict)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                trainer.save_model(config.checkpoints_path, "checkpoint_{t}")
            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score}, step=trainer.step
            )


if __name__ == "__main__":
    train()
