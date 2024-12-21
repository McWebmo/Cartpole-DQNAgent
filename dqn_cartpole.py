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

class Agent:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95 # Discount factor

        self.epsilon = 1.0 # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        

        model.compile(loss="mse", optimizer=Adam(learning_rate=0.01))
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state.reshape(1, -1), verbose=0))
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

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