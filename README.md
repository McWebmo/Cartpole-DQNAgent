# Deep Q-Learning with CartPole! 🚀

Welcome to my Deep Q-Learning (DQN) project! In this repository, I've implemented a reinforcement learning agent that masters the classic **CartPole-v0** environment from OpenAI Gym. 🏋️‍♂️ The agent leverages a neural network to approximate the Q-value function and learns to balance the pole through trial and error.

## Key Features ✨
- **Environment**: OpenAI Gym's CartPole-v0.
- **Deep Q-Learning**: Combines Q-learning with a neural network to handle continuous state spaces.
- **Experience Replay**: Efficient learning by sampling random batches of past experiences.
- **Exploration vs. Exploitation**: Balances random exploration with strategic exploitation using an adaptive epsilon-greedy policy.

---

## Why These Choices? 🤔

### 1. **Environment**: CartPole-v0
The **CartPole-v0** environment is a perfect playground for reinforcement learning:
- Simple yet non-trivial dynamics.
- Continuous state space (pole angle, cart position, etc.).
- Binary action space (left or right).

This makes it an excellent starting point for testing RL algorithms. 💡

### 2. **Model Architecture** 🧠
I designed a **fully connected neural network** with the following layers:
- **Input Layer**: Matches the size of the state space (4 features).
- **Two Hidden Layers**: Each with 24 neurons and ReLU activation to capture complex patterns.
- **Output Layer**: Produces Q-values for each action (2 outputs).

Why ReLU?
- Efficient and well-suited for deep networks.
- Helps the network learn non-linear state-to-action mappings.

### 3. **Exploration Strategy** 🎲
- **Epsilon-Greedy**: Encourages exploration during early training by choosing random actions with probability \( \epsilon \), which gradually decays.
- **Decay Mechanism**: \( \epsilon \) starts at 1.0 and decreases by 0.5% per episode until it reaches a minimum of 0.01.

This ensures the agent explores early but shifts to exploiting its learned policy as training progresses. 🚴

### 4. **Discount Factor (\( \gamma \))** 📉
\( \gamma = 0.95 \): Encourages the agent to focus on longer-term rewards without entirely ignoring immediate rewards. A good balance for CartPole dynamics! ⚖️

### 5. **Experience Replay** 🔄
- Stores up to 2000 past experiences in a memory buffer.
- Samples random minibatches of size 32 for training.
- Breaks correlation between consecutive experiences, improving stability and convergence.

### 6. **Training Process** 🏋️
The agent trains over 1000 episodes, saving its weights every 50 episodes. This modular approach allows us to resume training or analyze intermediate models. 💾

---

## Code Walkthrough 🖥️

### Agent Class
The `Agent` class encapsulates the core DQN functionality:
- **Neural Network**: Built using TensorFlow/Keras with Adam optimizer and Mean Squared Error (MSE) loss.
- **Memory Buffer**: A `deque` to store experiences.
- **Key Methods**:
  - `act`: Chooses actions using the epsilon-greedy policy.
  - `remember`: Adds experiences to memory.
  - `replay`: Trains the model on sampled experiences.
  - `save` & `load`: Saves and loads model weights.

### Training Loop
The training loop interacts with the environment:
1. Resets the environment for each episode.
2. Executes actions based on the policy and collects rewards.
3. Trains the agent using `replay` after each episode.
4. Periodically saves model weights.

---

## Results 🎉
After training, the agent consistently balances the pole for extended periods. You can visualize its performance by running the code with `render_mode="human"` enabled. Watching the agent in action is super satisfying! 🏆

---

## How to Run 🛠️
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install gym tensorflow numpy
   ```
3. Run the script:
   ```bash
   python dqn_cartpole.py
   ```
4. Sit back and watch the magic unfold! ✨

---

## Challenges & Learnings 🧗
- **Balancing Exploration and Exploitation**: Tuning the epsilon decay rate was critical to ensure the agent didn't get stuck in suboptimal policies.
- **Overfitting to Early Experiences**: Experience replay mitigated this by introducing diverse training samples.
- **Hyperparameter Sensitivity**: Finding the right learning rate and discount factor required experimentation.

---

## What’s Next? 🚀
- Implement **Double DQN** to reduce overestimation bias.
- Add a **Target Network** for more stable training.
- Test on more complex environments like MountainCar or LunarLander.
- Visualize training progress using TensorBoard.

---

## Acknowledgments 🙌
- OpenAI Gym for providing the environment.
- TensorFlow/Keras for the amazing deep learning tools.
- The RL research community for inspiring this project.

---

### Let’s Collaborate! 🤝
If you’d like to improve this project or have ideas to share, feel free to fork, star, or open an issue. Contributions are always welcome! 🌟

---

Thanks for checking out my project. Happy coding! 😄

