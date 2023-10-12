# If your environment is not registered, you may optionally pass a module
# to import, that would register your environment before creating it like
# this - ``env = gymnasium.make('module:Env-v0')``, where ``module``
# contains the registration code. For the GridWorld env, the registration
# code is run by importing ``gym_examples`` so if it were not possible to
# import gym_examples explicitly, you could register while making by
# ``#    from gym_examples.envs.grid_world import GridWorldEnv``. This
# is especially useful when you’re allowed to pass only the environment ID
# into a third-party codebase (eg. learning library). This lets you
# register your environment without needing to edit the library’s source
# code.

import gymnasium as gym
import Rob1
env = gym.make('Rob1/Rob-v0', render_mode='human')
observation, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
# observation = env.reset()
# env.render()

# while True:
#     # Take a random action for demonstration purposes
#     action = env.action_space.sample()

#     # Perform the action in the environment
#     observation, reward, done, info = env.step(action)

#     if (done).any():  # Check if any element in 'done' array is True
#         break

#     # Render the updated environment
#     env.render()

#     # Check if the episode is done
#     if done:
#         break

# # Close the environment when done
# env.close()