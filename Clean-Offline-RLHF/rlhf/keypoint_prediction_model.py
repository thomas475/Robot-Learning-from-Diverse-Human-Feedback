import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision.models as models
import utils


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        if key == 'context':
            indexed[key] = batch[key]
        else:
            indexed[key] = batch[key][indices, ...]
    return indexed


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        pass

    return net


class KeypointPredictorModel(object):
    def __init__(self, task, observation_dim, action_dim, ensemble_size=3, lr=3e-4, activation="tanh", logger=None,
                 device="cpu"):
        self.task = task
        self.observation_dim = observation_dim  # state: env.observation_space.shape[0]
        self.action_dim = action_dim  # state: env.action_space.shape[0]
        self.ensemble_size = ensemble_size  # ensemble_size
        self.lr = lr  # learning rate
        self.logger = logger  # logger
        self.device = torch.device(device)

        # build network
        self.opt = None
        self.activation = activation
        self.ensemble = []
        self.paramlst = []
        self.construct_ensemble()

    def construct_ensemble(self):
        for i in range(self.ensemble_size):
            model = nn.Sequential(*gen_net(in_size=self.observation_dim,
                                           out_size=self.observation_dim, H=256, n_layers=3,
                                           activation=self.activation)).float().to(self.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def save_model(self, path):
        state_dicts = [model.state_dict() for model in self.ensemble]
        torch.save(state_dicts, path)

    def load_model(self, path):
        state_dicts = torch.load(path, map_location='cpu')
        for model, state_dict in zip(self.ensemble, state_dicts):
            model.load_state_dict(state_dict)
            model.to(self.device)

    def train(self, n_epochs, dataset, data_size, batch_size, feedback_type='keypoint'):
        if feedback_type not in ['keypoint']:
            raise ValueError('Training a keypoint predictor is not possible for the "' + feedback_type + '" feedback type.')
          
        interval = int(data_size / batch_size) + 1
        prev_mse = float('inf')

        for epoch in range(1, n_epochs + 1):
            ensemble_losses = [[] for _ in range(self.ensemble_size)]
            ensemble_mse = [[] for _ in range(self.ensemble_size)]

            batch_shuffled_idx = []
            for _ in range(self.ensemble_size):
                batch_shuffled_idx.append(np.random.permutation(dataset["observations"].shape[0]))

            for i in range(interval):
                self.opt.zero_grad()
                total_loss = 0
                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, dataset["observations"].shape[0])
                for member in range(self.ensemble_size):
                    # get batch
                    batch = index_batch(dataset, batch_shuffled_idx[member][start_pt:end_pt])
                    # compute loss
                    curr_loss, mse = self._train(batch, member, feedback_type)
                    total_loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    ensemble_mse[member].append(mse)
                total_loss.backward()
                self.opt.step()

            train_metrics = {"epoch": epoch,
                             "avg_loss": np.mean(ensemble_losses),
                             "avg_mse": np.mean(ensemble_mse)}
            for i in range(self.ensemble_size):
                train_metrics.update({f"ensemble_{i}_loss": np.mean(ensemble_losses[i])})
                train_metrics.update({f"ensemble_{i}_mse": np.mean(ensemble_mse[i])})
            self.logger.log(train_metrics)

            # early stop if change in mse is lower tham 0.1 %
            if (1 - np.mean(ensemble_mse) / prev_mse) < 0.001 and "antmaze" not in self.task:
                break

            prev_mse = ensemble_mse

    def _train(self, batch, member, feedback_type='keypoint'):  
        # get batch
        obs = batch['observations']  # batch_size * len_query * obs_dim

        # get closest keypoint for each state
        labels = []
        for i in range(len(obs)):
            trajectory = obs[i]
            keypoints = np.array(batch['labels'][i])
            # compute euclidean distance between each state and each keypoint
            dists = np.sum((trajectory[:, np.newaxis, :] - keypoints[np.newaxis, :, :]) ** 2, axis=2)
            # find the index of the minimum distance for each state
            labels.append(keypoints[np.argmin(dists, axis=1)])
        labels = torch.from_numpy(np.array(labels)).to(self.device) # batch_size * len_query * obs_dim
        
        # get keypoint predictions
        g = self.predict_by_member(obs, member)  # batch_size * len_query * obs_dim

        # compute loss
        curr_loss = self.regression_loss(g, labels)

        # compute mse
        mse = np.mean((g - labels).detach().numpy() ** 2)

        return curr_loss, mse

    def predict_by_member(self, x, member):
        return self.ensemble[member](torch.from_numpy(x).float().to(self.device))

    def regression_loss(self, input, target):
        trajectory_loss = torch.linalg.norm((input - target), axis=2).sum(axis=1)
        return trajectory_loss.sum() / input.shape[0]
    
    def predict(self, x):
        keypoint_predictions_by_member = []
        for member in range(self.ensemble_size):
            keypoint_predictions_by_member.append(self.predict_by_member(x, member=member).detach().cpu().numpy())
        keypoint_predictions_by_member = np.array(keypoint_predictions_by_member)
        return np.mean(keypoint_predictions_by_member, axis=0)
