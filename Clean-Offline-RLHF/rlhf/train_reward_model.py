import os
import time
import warnings
import numpy as np
import gym

from reward_model import RewardModel, CNNRewardModel, TransformerRewardModel
# from utils import AttrFunc, GaussianNormalizer, get_episode_boundaries, MinMaxScaler, generate_trajectory_pairs
from utils import *
from tqdm import tqdm
import logger
import random
import torch
import pickle
from pathlib import Path
from cfg import parse_cfg
from sklearn.model_selection import train_test_split


warnings.filterwarnings('ignore')

__CONFIG__, __LOGS__ = 'cfgs', 'reward_model_logs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_d4rl_dataset(env):
    import d4rl
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset['observations'],
        actions=dataset['actions'],
        next_observations=dataset['next_observations'],
        rewards=dataset['rewards'],
        dones=dataset['terminals'].astype(np.float32),
    )


def get_atari_dataset(env):
    datasets = env.get_dataset()
    
    print("Finished, loaded {} timesteps.".format(int(datasets["rewards"].shape[0])))
    print(datasets.keys())
    
    return datasets


def load_queries_with_indices(dataset, num_query, len_query, saved_indices, saved_labels=None, label_type=1,
                              scripted_teacher=False, equivalence_threshold=0, modality="state", partition_idx=None, feedback_type="comparative"):    
    if modality == "state":
        observation_dim = (dataset["observations"].shape[-1], )
    elif modality == "pixel":
        observation_dim = dataset["observations"].shape[-3:]
    else:
        raise ValueError("Modality error")

    action_dim = dataset["actions"].shape[-1]
    
    if saved_labels is None:
        query_range = np.arange(num_query)
    else:
        # do not query all label
        if partition_idx is None:
            query_range = np.arange(len(saved_labels) - num_query, len(saved_labels))
        else:
            # If dataset is large, you should load the dataset in slices.
            query_range = np.arange(partition_idx * num_query, (partition_idx + 1) * num_query)

    total_reward_seq = np.zeros((2, num_query, len_query))
    total_obs_seq = np.zeros((2, num_query, len_query) + observation_dim)
    total_act_seq = np.zeros((2, num_query, len_query, action_dim))
    total_timestep = np.zeros((2, num_query, len_query), dtype=np.int32)

    for query_count, i in enumerate(query_range):
        for j in range(len(saved_indices)):
            start_idx = int(saved_indices[j][i])
            end_idx = start_idx + len_query
            total_reward_seq[j][query_count] = dataset['rewards'][start_idx:end_idx]
            total_obs_seq[j][query_count] = dataset['observations'][start_idx:end_idx]
            total_act_seq[j][query_count] = dataset['actions'][start_idx:end_idx]
            total_timestep[j][query_count] = np.arange(1, len_query + 1)

    batch = {}
    batch['rewards'] = total_reward_seq[0].copy()
    batch['observations'] = total_obs_seq[0].copy()
    batch['actions'] = total_act_seq[0].copy()
    batch['timestep'] = total_timestep[0].copy()
    batch['start_indices'] = saved_indices[0]
    if feedback_type in ['comparative', 'attribute']:
        batch['rewards_2'] = total_reward_seq[1].copy()
        batch['observations_2'] = total_obs_seq[1].copy()
        batch['actions_2'] = total_act_seq[1].copy()
        batch['timestep_2'] = total_timestep[1].copy()
        batch['start_indices_2'] = saved_indices[1]

    if feedback_type in ['comparative', 'attribute']:    
        # script_labels - label_type = 0 perfectly rational / label_type = 1 equivalence_threshold
        sum_r_t_1 = np.sum(batch['rewards'], axis=1)
        sum_r_t_2 = np.sum(batch['rewards_2'], axis=1)
        binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
        rational_labels = np.zeros((len(binary_label), 2))
        rational_labels[np.arange(binary_label.size), binary_label] = 1.0
        if label_type == 1: # equivalence_threshold
            margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= equivalence_threshold).reshape(-1)
            rational_labels[margin_index] = 0.5
        batch['script_labels'] = rational_labels

        if scripted_teacher:
            # fake label
            batch['labels'] = saved_labels
        else:
            # human label 
            if feedback_type in ['comparative']:    
                label_shape = (len(saved_labels), 2)
                human_labels = np.zeros(label_shape)
                human_labels[np.array(saved_labels) == 0, 0] = 1.
                human_labels[np.array(saved_labels) == 1, 1] = 1.
                human_labels[np.array(saved_labels) == -1] = 0.5
            elif feedback_type in ['attribute']:
                human_labels = np.array(saved_labels)
                human_labels[np.array(saved_labels) == -1] = 0.5
            human_labels = human_labels[query_range]
            batch['labels'] = human_labels
            # label_shape = (len(saved_labels), 2)
            # if feedback_type in ['attribute']:
            #     label_shape = saved_labels.shape + (2,)
            # human_labels = np.zeros(label_shape)
            # human_labels[np.array(saved_labels) == 0, 0] = 1.
            # human_labels[np.array(saved_labels) == 1, 1] = 1.
            # human_labels[np.array(saved_labels) == -1] = 0.5
            # human_labels = human_labels[query_range]
            # batch['labels'] = human_labels

    return batch


def train(cfg):
    # set seed
    set_seed(cfg.seed)
    # get work dir
    last_name = 'epoch_' + str(cfg.n_epochs) + '_seed_' + str(cfg.seed)
    # last_name = 'epoch_' + str(cfg.n_epochs) + '_query_' + str(cfg.num_query) +\
    #             '_len_' + str(cfg.len_query) + '_seed_' + str(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.env / cfg.exp_name / last_name
    print("work directory:", work_dir)
    L = logger.Logger(work_dir, cfg)

    # setup environments
    if cfg.domain == "atari":
        import d4rl_atari
        gym_env = gym.make(cfg.env, stack=cfg.stack)
        dataset = get_atari_dataset(gym_env)
        # action extension
        dataset['actions'] = dataset['actions'].reshape(-1, 1)
        # transform to onehot type
        observation_dim = gym_env.observation_space.shape  # (84, 84)
        action_dim = gym_env.action_space.n  # 6
        dataset["actions"] = np.eye(action_dim)[dataset["actions"].reshape(-1)]
    elif cfg.domain in ["mujoco", "antmaze", "adroit", "d4rl"]:
        import d4rl
        gym_env = gym.make(cfg.env)
        dataset = get_d4rl_dataset(gym_env.unwrapped)
        dataset['actions'] = np.clip(dataset['actions'], -cfg.clip_action, cfg.clip_action)
        observation_dim = gym_env.observation_space.shape[0]
        action_dim = gym_env.action_space.shape[0]
    else:
        raise ValueError("Domain not found!")
    print(f"Load env {cfg.env} successfully!")
    
    # load human labels or fake labels
    if cfg.fake_label:
        suffix = 'fake_labels'
    else:
        suffix = 'human_labels'

    if 'feedback_type' in cfg and cfg.feedback_type:
        for feedback_type in cfg.feedback_type:
            _train(L, action_dim, cfg, dataset, _load_data(cfg, suffix, feedback_type), observation_dim, feedback_type)
    else:
        _train(L, action_dim, cfg, dataset, _load_data(cfg, suffix), observation_dim)

def _train(L, action_dim, cfg, dataset, label_data, observation_dim, feedback_type='comparative'):
    if feedback_type in ['keypoint']:
        warnings.warn('The feedback type "' + feedback_type + '" does not require a reward model, so training is skipped.')
        return
    
    if feedback_type in ['comparative', 'attribute']:
        human_indices_1, human_indices_2, human_labels, num_query, len_query = label_data
    elif feedback_type in ['evaluative']:
        human_indices, human_labels, num_query, len_query = label_data
    elif feedback_type in ['keypoint', 'visual']:
        human_labels, num_query, len_query = label_data
    else:
        raise ValueError("Invalid feedback type:", feedback_type)

    # train reward model
    if cfg.modality == "state":
        if cfg.structure == 'mlp':
            reward_model = RewardModel(cfg.env, observation_dim, action_dim, ensemble_size=cfg.ensemble_size, lr=3e-4,
                                       activation="tanh", logger=L)
        elif "transformer" in cfg.structure:
            reward_model = TransformerRewardModel(
                cfg.env, observation_dim, action_dim, ensemble_size=cfg.ensemble_size, lr=5e-5,
                structure_type=cfg.structure, d_model=cfg.d_model, num_layers=cfg.num_layers, nhead=cfg.nhead,
                max_seq_len=cfg.max_seq_len,
                activation="tanh", logger=L)
        
        if feedback_type in ["comparative", "attribute"]:
            saved_indices = [human_indices_1, human_indices_2]
        elif feedback_type in ["evaluative", "visual"]:
            saved_indices = [human_indices]

        # if we use attribute feedback, learn attribute strength mapping first and generate pseudo labels
        if feedback_type in ['attribute']:
            pref_dataset = load_queries_with_indices(
                dataset, num_query, len_query, saved_indices, saved_labels=human_labels, 
                scripted_teacher=cfg.fake_label, modality=cfg.modality, feedback_type=feedback_type)
            
            attribute_dim = pref_dataset['labels'].shape[1]
            normalizer = GaussianNormalizer(torch.FloatTensor(dataset['observations']).to(cfg.device))
            nor_raw_obs = normalizer.normalize(torch.FloatTensor(dataset['observations']).to(cfg.device))
            episode_boundaries = get_episode_boundaries(dataset['dones'])

            # Create attribute function
            attr_func = AttrFunc(observation_dim, attribute_dim, ensemble_size=cfg.attr_map_ensemble_size).to(cfg.device)
            attr_func.train()

            attr_func_file_name = f'{cfg.env}_{cfg.attr_map_n_gradient_steps}'
            if cfg.attr_map_load and os.path.exists(os.path.join('results', 'attr_func', cfg.attr_func_file_name + '.pt')):
                attr_func.load(attr_func_file_name)
            else:
                # Preprocessing
                if cfg.attr_map_test_size and cfg.attr_map_test_size > 0.0:
                    train_idx, test_idx = train_test_split(
                        np.arange(len(pref_dataset['observations'])), 
                        test_size=cfg.attr_map_test_size, 
                        random_state=cfg.seed
                    )
                else:
                    train_idx, test_idx = np.arange(len(pref_dataset['observations'])), np.array([])
                obs1 = normalizer.normalize(torch.FloatTensor(pref_dataset['observations'][:]).to(cfg.device))
                obs2 = normalizer.normalize(torch.FloatTensor(pref_dataset['observations_2'][:]).to(cfg.device))
                pref = torch.FloatTensor(pref_dataset['labels'][:]).to(cfg.device)
                train_obs1, train_obs2, train_pref = obs1[train_idx], obs2[train_idx], pref[train_idx]
                test_obs1, test_obs2, test_pref = obs1[test_idx], obs2[test_idx], pref[test_idx]

                # Train attribute function
                print(f"Start training {cfg.env} attribute function with test size {cfg.attr_map_test_size}.")
                mean_sr = 0.
                pbar = tqdm(range(cfg.attr_map_n_gradient_steps))
                for n in range(cfg.attr_map_n_gradient_steps):
                    log = {"loss": np.zeros(attr_func.ensemble_size)}
                    for k in range(attr_func.ensemble_size):
                        idx = torch.randint(len(train_pref), (cfg.attr_map_batch_size,), device=cfg.device)
                        obs1, obs2, pref = train_obs1[idx], train_obs2[idx], train_pref[idx]
                        loss = attr_func.update(obs1, obs2, pref, k)
                        log["loss"][k] += loss
                        
                    if (n + 1) % cfg.attr_map_test_interval == 0 and cfg.attr_map_test_size > 0:
                        with torch.no_grad():
                            attr_func.eval()
                            prob = attr_func.predict_pref_prob(test_obs1, test_obs2)
                            mean_sr = ((prob >= 0.5) == test_pref).float().mean(0).cpu().numpy()
                            attr_func.train()
                        log["mean_sr"] = mean_sr
                        pbar.set_description(f"Loss {log['loss']} Mean success rate: {mean_sr}")
                    else:
                        pbar.set_description(f"Loss {log['loss']}")
                    pbar.update(1)
                attr_func.save(attr_func_file_name)
                print(f"Finish training.")

            # Normalize the attribute mapping
            print(f"Normalizing attribute mapping for {cfg.env} behavior dataset.")
            attr_func.eval()
            with torch.no_grad():
                attr_strength = []
                for start_idx, end_idx in tqdm(episode_boundaries):
                    episode_traj = torch.unsqueeze(torch.FloatTensor(dataset['observations']).to(cfg.device)[start_idx:end_idx], 0)
                    attr_strength.append(attr_func.predict_attr(episode_traj, None))
                attr_strength = torch.cat(attr_strength, dim=0).cpu().numpy()
            attr_strength_normalizer = MinMaxScaler(np.min(attr_strength, axis=0), np.max(attr_strength, axis=0))
            attr_func.add_output_normalization(attr_strength_normalizer)

            # Generate pseudo labels
            print(f"Generating {cfg.attr_map_n_pseudo_labels} pseudo labels for {cfg.env} behavior dataset.")
            attr_func.eval()
            with torch.no_grad():
                # generate trajectories
                trajectory_boundaries = generate_trajectory_boundaries(cfg.attr_map_n_pseudo_labels * 2, len_query, episode_boundaries, cfg.device)
                
                # generate trajectory attributes with trained attribute mapping function
                trajectory_attributes = np.zeros((cfg.attr_map_n_pseudo_labels * 2, attribute_dim))
                for i in range(len(trajectory_boundaries)):
                    start_idx, end_idx = trajectory_boundaries[i][0], trajectory_boundaries[i][1]
                    trajectory = torch.unsqueeze(torch.FloatTensor(dataset['observations']).to(cfg.device)[start_idx:end_idx], 0)
                    trajectory_attributes[i] = attr_func.predict_attr(trajectory, None)
                
                # perform pairwise comparisons and generate pseudo labels
                saved_indices_1, saved_indices_2, human_labels = np.zeros((cfg.attr_map_n_pseudo_labels)), np.zeros((cfg.attr_map_n_pseudo_labels)), np.zeros((cfg.attr_map_n_pseudo_labels))
                threshold = cfg.attr_map_threshold * np.sqrt(attribute_dim) # use a threshold that depends on the number of attributes
                for i in tqdm(range(cfg.attr_map_n_pseudo_labels)):
                    saved_indices_1[i] = trajectory_boundaries[2 * i][0]
                    saved_indices_2[i] = trajectory_boundaries[2 * i + 1][0]
                    norm_1 = np.linalg.norm(trajectory_attributes[2 * i] - cfg.attr_map_target)
                    norm_2 = np.linalg.norm(trajectory_attributes[2 * i + 1] - cfg.attr_map_target)
                    if abs(norm_1 - norm_2) < threshold: # if norms are too similar make label equal preference
                        human_labels[i] = 0.5
                    elif norm_1 <= norm_2:
                        human_labels[i] = 0
                    else:
                        human_labels[i] = 1
                saved_indices = [saved_indices_1, saved_indices_2]

            # treat the reward learning as comparative, but use the pseudo labels
            feedback_type = 'comparative'

        pref_dataset = load_queries_with_indices(
            dataset, num_query, len_query, saved_indices, saved_labels=human_labels, 
            scripted_teacher=cfg.fake_label, modality=cfg.modality, feedback_type=feedback_type)

        reward_model.train(n_epochs=cfg.n_epochs, dataset=pref_dataset,
                           data_size=pref_dataset["observations"].shape[0],
                           batch_size=cfg.batch_size, feedback_type=feedback_type)
    else:
        reward_model = CNNRewardModel(cfg.env, observation_dim, action_dim, ensemble_size=cfg.ensemble_size, lr=5e-4,
                                      activation=None, logger=L)
        N_DATASET_PARTITION = 5
        pref_dataset = [load_queries_with_indices(
            dataset, num_query // N_DATASET_PARTITION, len_query,
            saved_indices=[human_indices_1, human_indices_2],
            saved_labels=human_labels, scripted_teacher=cfg.fake_label, modality=cfg.modality, partition_idx=p_idx, feedback_type=feedback_type)
            for p_idx in range(N_DATASET_PARTITION)]
        # data_size = None means computing data size in function.
        reward_model.split_train(n_epochs=cfg.n_epochs, dataset=pref_dataset,
                                 data_size=None,
                                 batch_size=cfg.batch_size)
    L.finish(reward_model)
    print('Training completed successfully')


def _load_data(cfg, suffix, feedback_type=None):
    if feedback_type:
        data_dir = os.path.join(cfg.data_dir, f"{cfg.env}_{suffix}", feedback_type)
    else:
        data_dir = os.path.join(cfg.data_dir, f"{cfg.env}_{suffix}")
    print(f"Load saved indices from {data_dir}.")
    if os.path.exists(data_dir):
        suffix = f"domain_{cfg.domain}_env_{cfg.env}"
        # suffix = f"domain_{cfg.domain}_env_{cfg.env}_num_{cfg.num_query}_len_{cfg.len_query}"
        matched_file = []
        for file_name in os.listdir(data_dir):
            print(suffix)
            print(file_name)
            if suffix in file_name:
                matched_file.append(file_name)
        
        # unpickle transformed human labels
        unpickled_data = {}
        for file in matched_file:
            file_name = os.path.splitext(os.path.basename(file))[0]
            data_type = file_name.split('_domain_')[0]
            identifier = file_name.split('_')[-1]

            if identifier not in unpickled_data:
                unpickled_data[identifier] = {}
            with open(os.path.join(data_dir, file), "rb") as fp:  # Unpickling
                unpickled_data[identifier][data_type] = pickle.load(fp)
            
            if 'query_length' not in unpickled_data[identifier]:
                unpickled_data[identifier]['query_length'] = int(file_name.split('_len_')[1].split('_')[0])

        # verify that all datasets have the same number of queries
        query_length = next(iter(unpickled_data.values()))['query_length']
        for identifier in unpickled_data:
            assert unpickled_data[identifier]['query_length'] == query_length
            unpickled_data[identifier].pop('query_length')

        # # add end indices if necessary
        # for identifier in unpickled_data:
        #     for data_type in ['indices', 'indices_1', 'indices_2']:
        #         if data_type in unpickled_data[identifier]:
        #             unpickled_data[identifier]['start_' + data_type] = unpickled_data[identifier].pop(data_type)
        #             unpickled_data[identifier]['end_' + data_type] = unpickled_data[identifier]['start_' + data_type] + unpickled_data[identifier]['query_length']
        #     unpickled_data[identifier].pop('query_length')

        # concat data if multiple datasets are given
        concatenated_unpickled_data = {}
        for identifier in unpickled_data:
            for data_type in unpickled_data[identifier]:
                if data_type not in concatenated_unpickled_data:
                    if isinstance(unpickled_data[identifier][data_type], dict):
                        concatenated_unpickled_data[data_type] = {}
                    else:
                        initial_shape = (0,)
                        if len(unpickled_data[identifier][data_type].shape) > 1:
                            initial_shape += unpickled_data[identifier][data_type].shape[1:]
                        concatenated_unpickled_data[data_type] = np.empty(initial_shape)
                if isinstance(unpickled_data[identifier][data_type], dict):
                    concatenated_unpickled_data[data_type] = {
                        **concatenated_unpickled_data[data_type],
                        **unpickled_data[identifier][data_type]
                    }
                else:
                    concatenated_unpickled_data[data_type] = np.concatenate((
                        concatenated_unpickled_data[data_type],
                        unpickled_data[identifier][data_type]
                    ))

        # verify that the entries of all data types have the same length
        assert all(len(value) == len(next(iter(concatenated_unpickled_data.values()))) for value in concatenated_unpickled_data.values())

        # add query length and query number to the output
        concatenated_unpickled_data['num_query'] = len(next(iter(concatenated_unpickled_data.values())))
        concatenated_unpickled_data['len_query'] = query_length

        out = ()
        for data_type in [
            'indices', 'indices_1', 'indices_2', 
            # 'start_indices', 'start_indices_1', 'start_indices_2', 
            # 'end_indices', 'end_indices_1', 'end_indices_2', 
            'human_label',
            'num_query', 'len_query'
        ]:
            if data_type in concatenated_unpickled_data:
                out = out + (concatenated_unpickled_data[data_type],)
        out = out[0] if len(out) == 1 else out

        # human_labels, indices_1_file, indices_2_file = sorted(matched_file)
        # 
        # with open(os.path.join(data_dir, human_labels), "rb") as fp:  # Unpickling
        #     human_labels = pickle.load(fp)
        # with open(os.path.join(data_dir, indices_1_file), "rb") as fp:  # Unpickling
        #     human_indices_1 = pickle.load(fp)
        # with open(os.path.join(data_dir, indices_2_file), "rb") as fp:  # Unpickling
        #     human_indices_2 = pickle.load(fp)
    else:
        raise ValueError(f"Label not found")
    return out
    # return human_indices_1, human_indices_2, human_labels


if __name__ == '__main__':
    train(parse_cfg(Path().cwd() / __CONFIG__))
