import os
import time
import warnings
import numpy as np
import gym

from reward_model import RewardModel, CNNRewardModel, TransformerRewardModel
# from utils import AttrFunc, GaussianNormalizer, get_episode_boundaries, MinMaxScaler, generate_trajectory_pairs
from utils import *
from keypoint_prediction_model import KeypointPredictorModel
from dataclasses import asdict, dataclass, field
import pyrallis
from tqdm import tqdm
import logger
import random
import torch
import pickle
from pathlib import Path
from cfg import parse_cfg
from sklearn.model_selection import train_test_split


warnings.filterwarnings('ignore')

__LOGS__ = 'model_logs'

@dataclass
class TrainConfig:
    domain: str = "d4rl"
    env: str = "kitchen-complete-v0"
    modality: str = "state"  # state/pixel
    structure: str = "mlp" # mlp/transformer1/transformer2/transformer3
    clip_action: float = 0.999 # only d4rl
    stack: bool = False # stack frame in pixel benchmark, only atari
    feedback_type: list = field(default_factory=lambda: [
        "comparative", 
        # "attribute",
        # "evaluative",
        # "keypoint"
    ])

    # scripted teacher
    fake_label: bool = False # use scripted teacher to label queries instead of human feedback
    relabel_human_labels: bool = False # if true load human label data and adjust labels with scripted teacher, if false load generated fake label data
    n_evaluation_categories: int = 5
    comparison_equivalency_threshold: int = 0
    fake_label_data_dir: str = "../generated_fake_labels/"

    # learning
    ensemble_size: int = 1
    batch_size: int = 64
    n_epochs: int = 1000
    num_query: int = 200 # original 2000
    len_query: int = 100
    human_label_data_dir: str = "../crowdsource_human_labels/"

    # misc
    seed: int = 0
    save_model: bool = True
    use_wandb: bool = True
    wandb_project: str = "Uni-RLHF"
    wandb_group: str = "AuxiliaryModel"
    wandb_name: str = ""
    device: str = "cpu"

    # transformer structure
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 1
    max_seq_len: int = 200

    # attribute strength mapping
    attr_map_ensemble_size: int = 3 # original 3
    attr_map_n_gradient_steps: int = 3000 # original 3000
    attr_map_batch_size: int = 256 # original 256
    attr_map_test_size: float = 0.2
    attr_map_test_interval: int = 20 # original 100
    attr_map_target: list = field(default_factory=lambda: [1, 1]) # the desired target attributes
    attr_map_load: bool = True # load the attribute mapping if it already exists
    attr_map_n_pseudo_labels: int = 10 # number of generated pseudo labels
    attr_map_threshold: float = 0.1 # threshold to determine when trajectories have equal preference
    def __post_init__(self):
        if len(self.feedback_type) >= 1:
            self.wandb_name = "_".join([
                self.env, 
                self.wandb_group, 
                "_".join([feedback_type.capitalize() for feedback_type in self.feedback_type]), 
                str(self.seed)
            ])
        else:
            self.wandb_name = "_".join([self.env, self.wandb_group, str(self.seed)])

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


def load_queries_with_indices(
        env, dataset, num_query, len_query, saved_indices, saved_labels=None,
        scripted_teacher=False, relabel_human_labels=False, comparison_equivalence_threshold=0, n_evaluation_categories=5, 
        modality="state", partition_idx=None, feedback_type="comparative"):    
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

    if scripted_teacher:
        # scripted labels
        if relabel_human_labels:
            # replace human labels with scripted ones
            if feedback_type in ['comparative']:
                r_t_1 = batch['rewards']
                r_t_2 = batch['rewards_2']
                # reset trajectory rewards
                if "kitchen" in env:
                    r_t_1 = r_t_1 - r_t_1[:,0][:,np.newaxis]
                    r_t_2 = r_t_2 - r_t_2[:,0][:,np.newaxis]
                sum_r_t_1 = np.sum(r_t_1, axis=1)
                sum_r_t_2 = np.sum(r_t_2, axis=1)
                binary_label = 1 * (sum_r_t_1 < sum_r_t_2)
                rational_labels = np.zeros((len(binary_label), 2))
                rational_labels[np.arange(binary_label.size), binary_label] = 1.0
                if comparison_equivalence_threshold > 0.0:
                    margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) <= comparison_equivalence_threshold).reshape(-1)
                    rational_labels[margin_index] = 0.5
                batch['labels'] = rational_labels
            elif feedback_type in ['evaluative']:
                r_t = batch['rewards']
                # reset trajectory rewards
                if "kitchen" in env:
                    r_t = r_t - r_t[:,0][:,np.newaxis]
                sum_r_t = np.sum(r_t, axis=1)
                min_sum_r_t, max_sum_r_t = sum_r_t.min(), sum_r_t.max()
                sum_r_t = (sum_r_t - min_sum_r_t) / (max_sum_r_t - min_sum_r_t) # normalize summed rewards
                evaluation_upper_bounds = np.array([category / (n_evaluation_categories) for category in range(1, n_evaluation_categories + 1)])
                categories = np.clip(np.sum(sum_r_t[:, np.newaxis] >= evaluation_upper_bounds, axis=1), 0, n_evaluation_categories - 1) # category is highest index that is smaller than upper bound
                rational_labels = np.zeros((len(sum_r_t), n_evaluation_categories))
                for i in range(len(rational_labels)):
                    rational_labels[i][categories[i]] = 1
                batch['labels'] = rational_labels
            else:
                raise NotImplementedError('Scripted labels are not supported for "' + feedback_type + '" feedback.')
        else:
            # use already generated fake labels
            batch['labels'] = saved_labels
    else:
        # human labels   
        if feedback_type in ['comparative']:    
            label_shape = (len(saved_labels), 2)
            human_labels = np.zeros(label_shape)
            human_labels[np.array(saved_labels) == 0, 0] = 1.
            human_labels[np.array(saved_labels) == 1, 1] = 1.
            human_labels[np.array(saved_labels) == -1] = 0.5
            human_labels = human_labels[query_range]
            batch['labels'] = human_labels
        elif feedback_type in ['attribute']:
            human_labels = np.array(saved_labels)
            human_labels[np.array(saved_labels) == -1] = 0.5
            human_labels = human_labels[query_range]
            batch['labels'] = human_labels
        elif feedback_type in ['keypoint']:
            human_labels = []
            for i in range(num_query):
                keypoints = []
                for keypoint in saved_labels[saved_indices[0][i]]:
                    keypoints.append(dataset['observations'][keypoint])
                human_labels.append(keypoints)
            batch['labels'] = np.array(human_labels, dtype=object)
        else:
            batch['labels'] = saved_labels

    return batch

@pyrallis.wrap()
def train(cfg: TrainConfig):
    # set seed
    set_seed(cfg.seed)

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
    
    if len(cfg.feedback_type) >= 1:
        for feedback_type in cfg.feedback_type:
            _train(action_dim, cfg, dataset, _load_data(cfg, feedback_type), observation_dim, feedback_type)
    else:
        _train(action_dim, cfg, dataset, _load_data(cfg), observation_dim)

def _train(action_dim, cfg, dataset, label_data, observation_dim, feedback_type='comparative'):
    if feedback_type not in ['comparative', 'attribute', 'evaluative', 'keypoint']:
        raise NotImplementedError('Learning from the "' + feedback_type + '" feedback type is not supported yet.')
    
    if feedback_type in ['comparative', 'attribute']:
        human_indices_1, human_indices_2, human_labels, num_query, len_query = label_data
    elif feedback_type in ['evaluative']:
        human_indices, human_labels, num_query, len_query = label_data
    elif feedback_type in ['keypoint', 'visual']:
        human_labels, num_query, len_query = label_data
    else:
        raise ValueError("Invalid feedback type:", feedback_type)
    
    # initialize logger
    log_dir = os.path.join(
        Path().cwd(), __LOGS__, cfg.env, 
        'epoch_' + str(cfg.n_epochs) + '_query_' + str(num_query) + '_len_' + str(len_query) + '_seed_' + str(cfg.seed)
    )
    model_dir = os.path.join(log_dir, 'models')
    if cfg.fake_label:
        if cfg.relabel_human_labels:
            label_type = 'relabeled'
        else:
            label_type = 'scripted'
    else:
        label_type = 'human'
    model_name = f"{label_type}_{feedback_type}_reward_{cfg.structure}"
    print("Logging directory:", log_dir)
    L = logger.Logger(log_dir, cfg)

    # train reward model
    if cfg.modality == "state":
        if feedback_type in ["comparative", "attribute"]:
            saved_indices = [human_indices_1, human_indices_2]
        elif feedback_type in ["evaluative", "visual"]:
            saved_indices = [human_indices]
        elif feedback_type in ['keypoint']:
            saved_indices = [np.array(list(human_labels.keys()))]
            
        pref_dataset = load_queries_with_indices(
            cfg.env, dataset, num_query, len_query, saved_indices, saved_labels=human_labels, scripted_teacher=cfg.fake_label, 
            relabel_human_labels=cfg.relabel_human_labels, comparison_equivalence_threshold=cfg.comparison_equivalency_threshold, 
            n_evaluation_categories=cfg.n_evaluation_categories, modality=cfg.modality, feedback_type=feedback_type)
        
        # if we use attribute feedback, learn attribute strength mapping first and afterwards generate pseudo labels
        if feedback_type in ['attribute']:
            attribute_dim = pref_dataset['labels'].shape[1]
            normalizer = GaussianNormalizer(torch.FloatTensor(dataset['observations']).to(cfg.device))
            nor_raw_obs = normalizer.normalize(torch.FloatTensor(dataset['observations']).to(cfg.device))
            episode_boundaries = get_episode_boundaries(dataset['dones'])

            # Create attribute function
            attr_func = AttrFunc(observation_dim, attribute_dim, ensemble_size=cfg.attr_map_ensemble_size).to(cfg.device)
            attr_func.train()

            attr_func_model_name = f"{label_type}_{feedback_type}_mapping_transformer"
            if cfg.attr_map_load and os.path.exists(os.path.join(model_dir, attr_func_model_name + '.pt')):
                attr_func.load(model_dir, attr_func_model_name)
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
                attr_func.save(model_dir, attr_func_model_name)
                print(f"Finish training.")

            # Normalize the attribute mapping
            print(f"Normalizing attribute mapping for {cfg.env} behavior dataset.")
            attr_func.eval()
            with torch.no_grad():
                attr_strength = []
                for start_idx, end_idx in tqdm(episode_boundaries):
                    episode_traj = torch.unsqueeze(torch.FloatTensor(nor_raw_obs).to(cfg.device)[start_idx:end_idx], 0)
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
                    trajectory = torch.unsqueeze(torch.FloatTensor(nor_raw_obs).to(cfg.device)[start_idx:end_idx], 0)
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

            # load data for the pseudo labels
            pref_dataset = load_queries_with_indices(
                cfg.env, dataset, num_query, len_query, saved_indices, saved_labels=human_labels, scripted_teacher=cfg.fake_label, 
                relabel_human_labels=cfg.relabel_human_labels, comparison_equivalence_threshold=cfg.comparison_equivalency_threshold, 
                n_evaluation_categories=cfg.n_evaluation_categories, modality=cfg.modality, feedback_type='comparative')
        
        # # if we use evaluative feedback, we first have to determine rating boundaries
        # if feedback_type in ['evaluative']:
        #     print(f"Generating rating boundaries for {cfg.env} training dataset.")
        #     trajectory_reward_sums = sorted(list(np.sum(pref_dataset['rewards'], axis=1)))
            
        #     ratings = np.argmax(pref_dataset['labels'], axis=1)

        #     # get frequencies of rating categories
        #     n_rating_categories = max(ratings.astype(int)) + 1
        #     rating_categories, counts = np.unique(ratings.astype(int), return_counts=True)
        #     rating_category_counts = np.zeros(n_rating_categories)
        #     rating_category_counts[rating_categories] = counts

        #     # generate rating boundaries
        #     rating_boundaries = np.zeros(n_rating_categories + 1)
        #     rating_boundaries[0] = trajectory_reward_sums[0]
        #     rating_boundaries[n_rating_categories] = trajectory_reward_sums[-1]
        #     for i in range(1, n_rating_categories):
        #         cumulative_count = np.sum(rating_category_counts[:i]).astype(int) - 1
        #         rating_boundaries[i] = (trajectory_reward_sums[cumulative_count] + trajectory_reward_sums[cumulative_count + 1]) / 2
            
        #     # normalize rewards by mapping them to 0 to 1
        #     normalizer = MinMaxScaler(rating_boundaries[0], rating_boundaries[-1])
        #     pref_dataset['context'] = {
        #         'boundaries': normalizer.transform(rating_boundaries),
        #         'min_unnormalized_reward_sum': rating_boundaries[0],
        #         'max_unnormalized_reward_sum': rating_boundaries[-1]
        #     }

        # train the models
        if feedback_type in ['comparative', 'attribute', 'evaluative']:
            if cfg.structure == 'mlp':
                model = RewardModel(cfg.env, observation_dim, action_dim, ensemble_size=cfg.ensemble_size, lr=3e-4,
                                        activation="tanh", logger=L)
            elif "transformer" in cfg.structure:
                model = TransformerRewardModel(
                    cfg.env, observation_dim, action_dim, ensemble_size=cfg.ensemble_size, lr=5e-5,
                    structure_type=cfg.structure, d_model=cfg.d_model, num_layers=cfg.num_layers, nhead=cfg.nhead,
                    max_seq_len=cfg.max_seq_len,
                    activation="tanh", logger=L)
                
        elif feedback_type in ['keypoint']:
            model = KeypointPredictorModel(cfg.env, observation_dim, action_dim, 
                ensemble_size=cfg.ensemble_size, lr=3e-4, activation="tanh", logger=L)
            model_name = f"{label_type}_{feedback_type}_predictor_{cfg.structure}"
            
        model.train(n_epochs=cfg.n_epochs, dataset=pref_dataset,
            data_size=pref_dataset["observations"].shape[0],
            batch_size=cfg.batch_size, feedback_type=feedback_type)
            
    else:
        model = CNNRewardModel(cfg.env, observation_dim, action_dim, ensemble_size=cfg.ensemble_size, lr=5e-4,
                                      activation=None, logger=L)
        N_DATASET_PARTITION = 5
        pref_dataset = [load_queries_with_indices(
            cfg.env, dataset, num_query // N_DATASET_PARTITION, len_query,
            saved_indices=[human_indices_1, human_indices_2],
            saved_labels=human_labels, scripted_teacher=cfg.fake_label, 
            relabel_human_labels=cfg.relabel_human_labels, 
            comparison_equivalence_threshold=cfg.comparison_equivalency_threshold, 
            n_evaluation_categories=cfg.n_evaluation_categories, 
            modality=cfg.modality, partition_idx=p_idx, feedback_type=feedback_type)
            for p_idx in range(N_DATASET_PARTITION)]
        # data_size = None means computing data size in function.
        model.split_train(n_epochs=cfg.n_epochs, dataset=pref_dataset,
            data_size=None, batch_size=cfg.batch_size)
        
    L.finish(model, model_name)
    print('Training completed successfully')


def _load_data(cfg, feedback_type=None):
    if not cfg.fake_label or cfg.relabel_human_labels:
        data_dir = cfg.human_label_data_dir
        suffix = "_human_labels"
    else:
        data_dir = cfg.fake_label_data_dir
        suffix = "_fake_labels"

    if feedback_type:
        data_dir = os.path.join(data_dir, f"{cfg.env}" + suffix, feedback_type)
    else:
        data_dir = os.path.join(data_dir, f"{cfg.env}" + suffix)

    print(f"Load saved indices from {data_dir}.")
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Label not found")
    
    suffix = f"domain_{cfg.domain}_env_{cfg.env}_num_{cfg.num_query}_len_{cfg.len_query}"
    matched_file = []
    for file_name in os.listdir(data_dir):
        print(suffix)
        print(file_name)
        if suffix in file_name:
            matched_file.append(file_name)

    if len(matched_file) == 0:
        raise ValueError(f"No matching labels found.")
    
    # unpickle transformed labels
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
        'human_label', 'fake_label',
        'num_query', 'len_query'
    ]:
        if data_type in concatenated_unpickled_data:
            out = out + (concatenated_unpickled_data[data_type],)
    out = out[0] if len(out) == 1 else out

    return out


if __name__ == '__main__':
    train()
