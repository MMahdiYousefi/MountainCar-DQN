import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from collections import deque
from keras.models import load_model

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
######################################################################## REMEMBER ################################################################
######################################################################## REMEMBER ################################################################
######################################################################## REMEMBER ################################################################
######################################################################## REMEMBER ################################################################
######################################################################## REMEMBER ################################################################

class DQNAgent:
    def __init__(self, env):
        self.epsilon = 1.0
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.001
        self.gamma = 0.8
        self.learning_rate = 0.1
        self.memory = deque(maxlen=20000)

        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.n

        self.model = self.create_model()
        self.target_model = self.create_model()

        self.update_count = 0

    def create_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_shape, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_shape, activation='linear'))

        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        return model

    def train_model(self, batch_size):
        if len(self.memory) < batch_size:
            return 
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            curr_q = self.model.predict(state)
            print('curr_q: ',curr_q)
            if done:
                curr_q[0][action] = reward
                print('curr_q[0][action]: ',curr_q[0][action])
            else:
                max_future_q = max(self.target_model.predict(new_state)[0])
                curr_q[0][action] = reward + (max_future_q * self.gamma)
                print('curr_q[0][action]: ',curr_q[0][action])
            self.model.fit(state, curr_q, epochs=1, verbose=0)

        # Update the count for each episode
        self.update_count += 1
        # Update the target model every 7th episode
        if self.update_count %7==0:
            self.train_target_model()
        
        # Update the Epsilon every 30th episode
        if self.update_count %30==0:
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon - (self.epsilon * self.epsilon_decay)
            self.update_count = 0

    def take_action(self, state):
        if np.random.random() < max(self.epsilon, self.epsilon_min):
            return self.env.action_space().sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    def train_target_model(self):
        main_model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(target_model_weights)):
            target_model_weights[i] = main_model_weights[i]
        self.target_model.set_weights(target_model_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def save_model(self, file_name):
        self.model.save(file_name)

def main(): 
    env = gym.make('MountainCar-v0')
    dqn_agent = DQNAgent(env)

if __name__ == '__main__': main()

