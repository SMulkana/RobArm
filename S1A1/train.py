import gymnasium as gym
import Rob1
from stable_baselines3 import A2C
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt

env = gym.make('Rob1/Rob-v0', render_mode='human')

model = A2C("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")

# #log results
# results_plotter.save_results("Rob1", results)

# #plot results
# log_dir = "Rob1"
# results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "A2C Rob")
# plt.show()

# Create the Gym environment
# env = gym.make('Rob1/Rob-v0', render_mode='human')

# # Create the A2C model
# model = A2C("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=100)  # Corrected total_timesteps

# # log results
# results = model.get_env().get_attr("monitor")[0].get_results()
# results_plotter.save_results("Rob1", results)

# # plot results
# log_dir = "C:\\MyProjects\\RobArm\\S1A1"
# results_plotter.plot_results([log_dir], 100, results_plotter.X_TIMESTEPS, "A2C Rob")
# plt.show()
