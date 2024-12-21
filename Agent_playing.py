# play_cartpole.py
import gym
import os
import numpy as np
from agent import Agent  # Import the Agent class
import tensorflow as tf
np.bool8 = np.bool_
# tf.get_logger().setLevel('ERROR')

# Set up the environment
env = gym.make("CartPole-v0", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

output_dir = "./Models/"

# Initialize the agent

agent = Agent(state_size, action_size)

agent.model.load_weights("./models/Weights_1000.weights.h5")

state, _ = env.reset()

done = False
total_reward = 0

while not done:
    env.render()

    state = np.reshape(state, [1, state_size])
    action = agent.act(state)

    next_state, reward, done, _, __ = env.step(action)

    total_reward += reward

    state = next_state

    if done:
        print(f"Game over! Total reward: {total_reward}")
        break

env.close()
