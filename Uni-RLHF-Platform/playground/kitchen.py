import gym
import d4rl  # Import required to register environments, you may need to also import the submodule
import time

from d4rl.kitchen.adept_envs.mujoco_env import MujocoEnv


def run_kitchen():
    # Create the environment
    env = gym.make('kitchen-complete-v0')

    # d4rl abides by the OpenAI gym interface
    env.reset()
    env.step(env.action_space.sample())

    # Each task is associated with a dataset
    # dataset contains observations, actions, rewards, terminals, and infos
    dataset = env.get_dataset()
    num_steps = 1500

    obs = env.reset()

    for step in range(num_steps):
        # take random action, but you can also do something more intelligent
        # action = my_intelligent_agent_fn(obs)
        action = env.action_space.sample()

        # apply the action
        obs, reward, done, info = env.step(action)

        # Render the env
        env.render()
        print(step)

        # Wait a bit before the next frame unless you want to see a crazy fast video
        time.sleep(0.001)

        # If the epsiode is up, then start another one
        if done:
            env.reset()

    # Close the env
    env.close()


def run(environment):
    env = gym.make(environment)

    # Observation and action space
    obs_space = env.observation_space
    action_space = env.action_space
    print("The observation space: {}".format(obs_space))
    print("The action space: {}".format(action_space))

    # reset the environment and see the initial observation
    obs = env.reset()
    print("The initial observation is {}".format(obs))

    # Sample a random action from the entire action space
    random_action = env.action_space.sample()

    # # Take the action and get the new observation space
    new_obs, reward, done, info = env.step(random_action)
    print("The new observation is {}".format(new_obs))

    import time

    # Number of steps you run the agent for
    loop = True
    num_steps = 1500

    obs = env.reset()

    for step in range(num_steps):
        # take random action, but you can also do something more intelligent
        # action = my_intelligent_agent_fn(obs)
        action = env.action_space.sample()

        # apply the action
        obs, reward, done, info = env.step(action)

        # Render the env
        env.render()

        # Wait a bit before the next frame unless you want to see a crazy fast video
        time.sleep(0.001)

        # If the epsiode is up, then start another one
        if done:
            env.reset()

    # Close the env
    env.close()


# run('MountainCar-v0')
# run('CartPole-v1')
# run('HalfCheetah-v2')
# run('Walker2d-v3')

run_kitchen()







# Create the environment
# env = gym.make('kitchen-complete-v0')
#
# # d4rl abides by the OpenAI gym interface
# env.reset()
# env.step(env.action_space.sample())
#
# # Each task is associated with a dataset
# # dataset contains observations, actions, rewards, terminals, and infos
# dataset = env.get_dataset()
# print(dataset['observations']) # An N x dim_observation Numpy array of observations
#
# # Alternatively, use d4rl.qlearning_dataset which
# # also adds next_observations.
# dataset = d4rl.qlearning_dataset(env)
