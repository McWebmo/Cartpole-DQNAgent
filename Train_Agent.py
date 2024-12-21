import gym
import random 
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
from collections import deque
from agent import Agent

np.bool8 = np.bool_
tf.get_logger().setLevel('ERROR')

env = gym.make("CartPole-v0", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

batch_size = 32
n_episodes = 1000

output_dir = "./Models/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


agent = Agent(state_size, action_size)

done = False
for e in range(n_episodes):

    state, _ = env.reset()

    state = np.reshape(state, [1, state_size])

    for t in range(5000):
        env.render()
        
        action = agent.act(state)
        # print(env.step(action))
        next_state, reward, done, _, __= env.step(action)

        reward = reward if not done else -10

        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print(f"Episode: {e+1}/{n_episodes}, score: {t}, epsilon: {agent.epsilon:.2}")
            break

    if len(agent.memory) >= batch_size:
        agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(output_dir + "Weights_" + "{:04d}".format(e) + ".weights.h5")