import gym
import random
import time
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout

"""
    mountianCar-v0 Problem Description:
        The agent (Car) has to reach the flag up the hill by choosing the actions described down bellow.
        Now the agent can't just go forward until it gets to the flag, Because of the gravitaional pull (which is 0.0025 
        in this case). So it has to go left in order to build up the momentum to be able to reach the flag up the hill.

        Observation (state):
            0 --> Car position (min: -1.2    , max: 0.6)
            1 --> Car velocity (min: -0.07   , max: 0.07)

        Actions:
            0 --> Accelerate to the Left
            1 --> Don't accelerate
            2 --> Accelerate to the Right

        Reward:
            0 if agent reaches the flag (position = 0.5)
            -1 if agent's position is less than 0.5

"""

"""
    Code Description:
        This code is using Double Deep-Q-Network to solve the problem.
        First the main model starts learning every episode and then the second
        model learns every 'num' episode (num is your choice).

        Note: The second model will learn the task with the main model weights.

        Let's Get To Coding...

"""

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.epsilon = 1.0
        self.epsilon_decay = 0.04
        self.epsilon_min = 0.001
        self.gamma = 0.99
        self.learning_rate = 0.1
        self.memory = deque(maxlen=20000)

        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.n

        self.model = self.create_model()
        self.target_model = self.create_model()

        self.update_count = 0
    # Creating the net q model
    def create_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_shape, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.action_shape, activation='relu'))

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    # Training the model
    def train_model(self, batch_size):
        if len(self.memory) < batch_size:
            return 
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            curr_q = self.model.predict(state)
            if done:
                curr_q[0][action] = reward
            else:
                max_future_q = max(self.target_model.predict(new_state)[0])
                curr_q[0][action] = reward + (max_future_q * self.gamma)
            self.model.fit(state, curr_q, epochs=1, verbose=0)

        # Update the count for each episode
        self.update_count += 1
        # Update the target model every 7th episode
        if self.update_count %7==0:
            self.train_target_model()
        
        # Update the Epsilon every 35th episode
        if self.update_count %35==0:
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon - (self.epsilon * self.epsilon_decay)
            self.update_count = 0

    # Deciding whether to go for exploration or exploitaion based on epsilon
    def take_action(self, state):
        if np.random.random() < max(self.epsilon, self.epsilon_min):
            return self.env.action_space.sample() # Exploration
        else:
            return np.argmax(self.model.predict(state)[0]) # Exploitation

    # Sets the main model weights to target model
    def train_target_model(self):
        main_model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(target_model_weights)):
            target_model_weights[i] = main_model_weights[i]
        self.target_model.set_weights(target_model_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Saves the model (pretty obvious!)
    def save_model(self, file_name):
        self.model.save(file_name)


def main(): 
    env = gym.make('MountainCar-v0')
    dqn_agent = DQNAgent(env)

    EPISODES=10000
    LEN_EPISODES=200
    
    model_time_start = time.time()

    for episode in range(EPISODES):
        ep_start = time.time()

        episode_reward = 0
        curr_state = env.reset().reshape(1,2)
        for step in range(LEN_EPISODES):
            action = dqn_agent.take_action(curr_state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            if next_state[0] >= 0.5:
                print('########################################## TARGET REACHED ################################################')
            next_state = next_state.reshape(1,2)
            dqn_agent.remember(curr_state, action, reward, next_state, done)
            episode_reward += reward
            curr_state = next_state

            ep_end = time.time()
            ep_time = ep_end - ep_start
            if done:
                print(f'Episode count: {episode:>05} ||| Reward: {episode_reward} ||| Epsilon: {dqn_agent.epsilon:.2f} ||| Episode Exec time: {ep_time:.1f} seconds')
                break
        # Train the model based on batch size
        dqn_agent.train_model(64)

        # This is to get the total runtime of the program
        if episode == EPISODES-1:
            model_time_end = time.time()
            model_time = (model_time_end - model_time_start) / 3600
            print(f'Total Model Runtime: {model_time:.1f} Hours ... ')

        # Saving the model every 200th episode
        if episode %200==0:
            fn = "MountainCar-DQN_BestModel.h5"
            print(f'Saving the model at "{fn}"')
            dqn_agent.save_model(fn)
    

if __name__ == '__main__': main()
