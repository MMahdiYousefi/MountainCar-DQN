import time
import gym
import numpy as np
from tensorflow.keras.models import load_model

# loading the saved model
model = load_model('MountainCar-DQN_BestModel.h5')
env = gym.make('MountainCar-v0').env

EPISODES=5

# create the loop that shows the result
for e in range(EPISODES):
    state = env.reset()
    done = False
    start = time.time()
    while not done:
        env.render()
        state = np.reshape(state, (1,2))
        action = model.predict(state)
        action = np.argmax(action)
        next_state, reward, done, info = env.step(action)
        state = next_state
        end = time.time()
        ep_time = end - start
    print(f'Episode time: {ep_time} seconds')
