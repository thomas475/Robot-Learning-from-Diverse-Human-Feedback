import collections
import numpy as np
import gym
from tqdm import trange
import torch
import torch.nn as nn
import math
from typing import Optional
from pathlib import Path
import os


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

def to_torch(x, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)

class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-5 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


@torch.no_grad()
def reward_from_preference(
    dataset: D4RLDataset,
    reward_model,
    batch_size: int = 256,
    reward_model_type: str = "transformer",
    device="cuda"
):
    data_size = dataset["rewards"].shape[0]
    interval = int(data_size / batch_size) + 1
    new_r = np.zeros_like(dataset["rewards"])
    
    if reward_model_type == "transformer":
        max_seq_len = reward_model.max_seq_len
        for each in reward_model.ensemble:
            each.eval()
 
        obs, act = [], []
        ptr = 0
        for i in trange(data_size):
            
            if len(obs) < max_seq_len:
                obs.append(dataset["observations"][i])
                act.append(dataset["actions"][i])
            
            if dataset["terminals"][i] > 0 or i == data_size - 1 or len(obs) == max_seq_len:
                tensor_obs = to_torch(np.array(obs)[None,], dtype=torch.float32).to(device)
                tensor_act = to_torch(np.array(act)[None,], dtype=torch.float32).to(device)
                
                new_reward = 0
                for each in reward_model.ensemble:
                    new_reward += each(tensor_obs, tensor_act).detach().cpu().numpy()
                new_reward /= len(reward_model.ensemble)
                if tensor_obs.shape[1] <= -1:
                    new_r[ptr:ptr+tensor_obs.shape[1]] = dataset["rewards"][ptr:ptr+tensor_obs.shape[1]]
                else:
                    new_r[ptr:ptr+tensor_obs.shape[1]] = new_reward
                ptr += tensor_obs.shape[1]
                obs, act = [], []
    else:
        for i in trange(interval):
            start_pt = i * batch_size
            end_pt = (i + 1) * batch_size

            observations = dataset["observations"][start_pt:end_pt]
            actions = dataset["actions"][start_pt:end_pt]
            obs_act = np.concatenate([observations, actions], axis=-1)

            new_reward = reward_model.get_reward_batch(obs_act).reshape(-1)
            new_r[start_pt:end_pt] = new_reward
    
    dataset["rewards"] = new_r.copy()
    
    # rr = dataset["rewards"].copy()
    # fr = new_r.copy()
    
    # rr = (rr-rr.min())/(rr.max()-rr.min())
    # fr = (fr-fr.min())/(fr.max()-fr.min())
    
    # rr_n_bins, _ = np.histogram(rr, 10, (0, 1))
    # fr_n_bins, _ = np.histogram(fr, 10, (0, 1))
    
    # print(rr_n_bins)
    # print(fr_n_bins)
    
    return dataset

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PrefTransformer1(nn.Module):
    ''' Transformer Structure used in Preference Transformer.
    
    Description:
        This structure holds a causal transformer, which takes in a sequence of observations and actions, 
        and outputs a sequence of latent vectors. Then, pass the latent vectors through self-attention to
        get a weight vector, which is used to weight the latent vectors to get the final preference score.
    
    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - max_seq_len: maximum length of sequence
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''
    def __init__(self,
        observation_dim: int, action_dim: int, 
        max_seq_len: int = 100,
        d_model: int = 256, nhead: int = 4, num_layers: int = 1, 
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)
        
        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.causual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 
            num_layers
        )
        self.mask = nn.Transformer.generate_square_subsequent_mask(2*self.max_seq_len)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.r_proj = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        if self.mask.device != obs.device: self.mask = self.mask.to(obs.device)
        batch_size, traj_len = obs.shape[:2]
        
        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos
        
        x = torch.empty((batch_size, 2*traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act

        x = self.causual_transformer(x, self.mask[:2*traj_len,:2*traj_len])[:, 1::2]
        # x: (batch_size, traj_len, d_model)

        q = self.q_proj(x) # (batch_size, traj_len, d_model)
        k = self.k_proj(x) # (batch_size, traj_len, d_model)
        r = self.r_proj(x) # (batch_size, traj_len, 1)
        
        w = torch.softmax(q@k.permute(0, 2, 1)/np.sqrt(self.d_model), -1).mean(-2)
        # w: (batch_size, traj_len)
        
        z = (w * r.squeeze(-1)) # (batch_size, traj_len)
        
        return torch.tanh(z)


class PrefTransformer2(nn.Module):
    ''' Preference Transformer with no causal mask and no self-attention but one transformer layer to get the weight vector.
    
    Description:
        This structure has no causal mask and no self-attention.
        Instead, it uses one transformer layer to get the weight vector.
        
    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''
    def __init__(self,
        observation_dim: int, action_dim: int, 
        d_model: int, nhead: int, num_layers: int, 
    ):
        super().__init__()
        while num_layers < 2: num_layers += 1
        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 
            num_layers - 1
        )
        self.value_layer = nn.Sequential(nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 1
        ), nn.Linear(d_model, 1))
        self.weight_layer = nn.Sequential(nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 1
        ), nn.Linear(d_model, 1))

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        batch_size, traj_len = obs.shape[:2]
        
        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos
        
        x = torch.empty((batch_size, 2*traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act
        
        x = self.transformer(x)[:, 1::2]
        v = self.value_layer(x)
        w = torch.softmax(self.weight_layer(x), 1)
        return (w*v).squeeze(-1)
    

class PrefTransformer3(nn.Module):
    ''' Preference Transformer with no causal mask and no weight vector.
    
    Description:
        This structure has no causal mask and even no weight vector.
        Instead, it directly outputs the preference score.
        
    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''
    def __init__(self,
        observation_dim: int, action_dim: int, 
        d_model: int, nhead: int, num_layers: int, 
    ):
        super().__init__()

        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 
            num_layers
        )
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        batch_size, traj_len = obs.shape[:2]
        
        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos
        
        x = torch.empty((batch_size, 2*traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act
        
        x = self.transformer(x)[:, 1::2]
        return self.output_layer(x).squeeze(-1)
    
class StateOnlyPrefTransformer(nn.Module):
    def __init__(self,
        o_dim: int, attr_dim: int, 
        d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        self.attr_dim = attr_dim
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.obs_emb = nn.Sequential(
            nn.Linear(o_dim, d_model), nn.LayerNorm(d_model))
        self.attr_emb = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), num_layers)
        self.out_layer = nn.Linear(d_model, attr_dim)
        
    def forward(self, traj: torch.Tensor):
        batch_size, traj_len = traj.shape[:2]
        pos = self.pos_emb(torch.arange(traj_len, device=traj.device))[None,]
        obs = self.obs_emb(traj)
        x = self.transformer(torch.cat([obs+pos, self.attr_emb.repeat(batch_size,1,1)], 1))
        return self.out_layer(x[:, -1])
    
class AttrFunc(nn.Module):
    def __init__(self, 
        o_dim: int, attr_dim: int,
        attr_clip: float = 20., ensemble_size: int = 3,
        lr: float = 1e-4, weight_decay: float = 1e-4,
        d_model: int = 128, nhead: int = 4, num_layers: int = 2, normalizer=None):
        super().__init__()
        self.o_dim = o_dim
        self.attr_dim = attr_dim
        self.attr_clip = attr_clip
        self.ensemble_size = ensemble_size
        self.normalizer = None
        
        self.attr_func_ensemble = nn.ModuleList([
            StateOnlyPrefTransformer(o_dim, attr_dim, d_model, nhead, num_layers)
            for _ in range(self.ensemble_size)])
        self.optim = [torch.optim.AdamW(ensemble.parameters(), lr=lr, weight_decay=weight_decay)
            for ensemble in self.attr_func_ensemble]

    def _predict_attr_ensemble(self, traj: torch.Tensor, ensemble_idx: int = 0):
        '''
        Input:
            - traj: (batch_size, traj_len, o_dim)
            - ensemble_idx: int
            
        Output:
            - attr_strength: (batch_size, attr_dim) 
        '''
        traj_attr = self.attr_func_ensemble[ensemble_idx](traj)
        attr_strength = self.attr_clip * torch.tanh(traj_attr / self.attr_clip)
        if self.normalizer:
            attr_strength = self.normalizer.transform(attr_strength)
        return attr_strength
    
    def predict_attr(self, traj: torch.Tensor, ensemble_idx: Optional[int] = None):
        '''
        Input:
            - traj: (batch_size, traj_len, o_dim)
            - ensemble_idx: int
            
        Output:
            - attr_strength: (batch_size, attr_dim) 
        '''
        if ensemble_idx is not None:
            return self._predict_attr_ensemble(traj, ensemble_idx)
        else:
            sum_ensemble = [self._predict_attr_ensemble(traj, i) for i in range(self.ensemble_size)]
            return sum(sum_ensemble) / self.ensemble_size
        
    def predict_pref_prob(self, 
            traj0: torch.Tensor, traj1: torch.Tensor, 
            ensemble_idx: Optional[int] = None):
        """
        Compute P[t_0 > t_1] = exp[sum(r(t_0))]/{exp[sum(r(t_0))]+exp[sum(r(t_1))]}= 1 /{1+exp[sum(r(t_1) - r(t_0))]}
        ----
        Input:
            - traj0: (batch_size, traj_len, o_dim)
            - traj1: (batch_size, traj_len, o_dim)
        
        Output:
            - prob: (batch_size, attr_dim)
        """
        traj_attr_strength_0 = self.predict_attr(traj0, ensemble_idx) # (batch_size, attr_dim)
        traj_attr_strength_1 = self.predict_attr(traj1, ensemble_idx) # (batch_size, attr_dim)
        a1_minus_a0 = traj_attr_strength_1 - traj_attr_strength_0
        prob = 1.0 / (1.0 + torch.exp(a1_minus_a0))
        return prob
    
    def update(self, 
            traj0: torch.Tensor, traj1: torch.Tensor, 
            pref: torch.Tensor,
            ensemble_idx: Optional[int] = None
        ):
        """
        Update the parameters of the attribute function by minimizing the negative log-likelihood loss
        ----
        Input:
            - traj0: (batch_size, traj_len, o_dim)
            - traj1: (batch_size, traj_len, o_dim)
            - pref: (batch_size, attr_dim) # 0 means traj0 is preferred and 1 means traj1 is preferred
            - ensemble_idx: int # which ensemble to update
        
        Output:
            - loss: float
        """
        prob = self.predict_pref_prob(traj0, traj1, ensemble_idx)
        loss = - ((1-pref)*torch.log(prob+1e-8) + pref*torch.log(1-prob+1e-8)).mean()
        self.optim[ensemble_idx].zero_grad()
        loss.backward()
        self.optim[ensemble_idx].step()
        return loss.item()
    
    def save(self, file_directory, file_name):
        file_path = os.path.join(file_directory, file_name)
        if not os.path.exists(file_directory): os.makedirs(file_directory)
        torch.save(self.state_dict(), file_path + ".pt")
        
    def load(self, file_directory, file_name, map_location = None):
        file_path = os.path.join(file_directory, file_name)
        self.load_state_dict(torch.load(file_path + ".pt", map_location=map_location))

    def add_output_normalization(self, normalizer):
        self.normalizer = normalizer

class GaussianNormalizer():
    def __init__(self, x: torch.Tensor):
        self.mean, self.std = x.mean(0), x.std(0)
        self.std[torch.where(self.std==0.)] = 1.
    def normalize(self, x: torch.Tensor):
        return (x - self.mean[None,]) / self.std[None,]
    def unnormalize(self, x: torch.Tensor):
        return x * self.std[None,] + self.mean[None,]
    
def get_episode_boundaries(dones):
    episode_boundaries = []
    start = 0
    for i in range(len(dones)):
        if dones[i] == 1 or i == (len(dones) - 1):
            episode_boundaries.append((start, i))
            start = i + 1
    return episode_boundaries

class MinMaxScaler():
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
    def transform(self, x):
        return (x - self.minimum) / (self.maximum - self.minimum)
    
def generate_trajectory_boundaries(n_trajectories, trajectory_length, episode_boundaries, device):
    trajectory_boundaries = np.zeros((n_trajectories, 2), dtype=int)
    episode_indeces = torch.randint(len(episode_boundaries), (n_trajectories,), device=device)
    for i, episode_idx in enumerate(episode_indeces):
        episode_start_idx, episode_end_idx = episode_boundaries[episode_idx]
        trajectory_boundaries[i][0] = np.random.randint(episode_start_idx, episode_end_idx - trajectory_length)
        trajectory_boundaries[i][1] = trajectory_boundaries[i][0] + trajectory_length
    return trajectory_boundaries
    
# def generate_trajectory_pairs(n_trajectories, trajectory_length, episode_boundaries, device):
#     # generate n_trajectories * 2
#     trajectory_boundaries = []
#     episode_indeces = torch.randint(len(episode_boundaries), (n_trajectories * 2,), device=device)
#     for episode_idx in episode_indeces:
#         episode_start_idx, episode_end_idx = episode_boundaries[episode_idx]
#         trajectory_start_idx = torch.randint(episode_start_idx, episode_end_idx - trajectory_length, (1,), device=device)
#         trajectory_end_idx = episode_start_idx + trajectory_length
#         trajectory_boundaries.append((trajectory_start_idx, trajectory_end_idx))
    
#     # pair up trajectores
#     trajectory_pairs = np.array((2, n_trajectories))
#     for i in range(n_trajectories):
#         trajectory_pairs[0][i] = trajectory_boundaries[2 * i][0]
#         trajectory_pairs[1][i] = trajectory_boundaries[2 * i + 1][0]

#     return trajectory_pairs
    