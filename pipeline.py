"""Example code adapted from https://imitation.readthedocs.io/en/latest/getting-started/first_steps.html#first-steps
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env


class Pipeline:
    def __init__(self, env_name="seals:seals/CartPole-v0"):
        self.env_name = env_name
        self.rng = np.random.default_rng(0)
        self.env = make_vec_env(
            self.env_name,
            rng=self.rng,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
        )

    def train_expert(self):
        # note: use `download_expert` instead to download a pretrained, competent expert
        print("Training a expert.")
        expert = PPO(
            policy=MlpPolicy,
            env=self.env,
            seed=0,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=10,
            n_steps=64,
        )
        expert.learn(1_000)  # Note: change this to 100_000 to train a decent expert.
        return expert

    def download_expert(self):
        print("Downloading a pretrained expert.")
        expert = load_policy(
            "ppo-huggingface",
            organization="HumanCompatibleAI",
            env_name=self.env_name,
            venv=self.env,
        )
        return expert

    def sample_expert_transitions(self):
        # expert = train_expert()  # uncomment to train your own expert
        expert = self.download_expert()

        print("Sampling expert transitions.")
        rollouts = rollout.rollout(
            expert,
            self.env,
            rollout.make_sample_until(min_timesteps=None, min_episodes=50),
            rng=self.rng,
        )
        return rollout.flatten_trajectories(rollouts)

    def run(self):
        transitions = self.sample_expert_transitions()
        bc_trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            demonstrations=transitions,
            rng=self.rng,
        )

        evaluation_env = make_vec_env(
            self.env_name,
            rng=self.rng,
            env_make_kwargs={"render_mode": "human"},  # for rendering
        )

        print("Evaluating the untrained policy.")
        reward, _ = evaluate_policy(
            bc_trainer.policy,  # type: ignore[arg-type]
            evaluation_env,
            n_eval_episodes=3,
            render=True,  # comment out to speed up
        )
        print(f"Reward before training: {reward}")

        print("Training a policy using Behavior Cloning")
        bc_trainer.train(n_epochs=1)

        print("Evaluating the trained policy.")
        reward, _ = evaluate_policy(
            bc_trainer.policy,  # type: ignore[arg-type]
            evaluation_env,
            n_eval_episodes=3,
            render=True,  # comment out to speed up
        )
        print(f"Reward after training: {reward}")
