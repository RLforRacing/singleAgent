import sys
sys.path.insert(0, '../')
import gym
import numpy as np
from utils.image_proc import detect_edge, locate_car, dist_from_edge_ahead, strip_indicators
import matplotlib.pyplot as plt

env = gym.make('CarRacing-v0')
observation = env.reset()
total_reward = 0
T_max = 1000

# check the action and observation spaces
print('Action space:{}'.format(env.action_space))
print('Lowest value for actions: {}'.format(env.action_space.low))
print('Highest value for actions: {}'.format(env.action_space.high))

print('Observation space:{}'.format(env.observation_space))
print('Lowest value for observation: {}'.format(np.min(env.observation_space.low)))
print('Highest value for observation: {}'.format(np.min(env.observation_space.high)))

for cur_time in range(T_max):
    env.render()

    cur_action = env.action_space.sample() # take a random action
    observation, reward, is_done, info = env.step(cur_action)
    observation = strip_indicators(observation)  # Get rid of indicator bar on the bottom
    filtered = detect_edge(observation)  # Detect edges for vis
    car_loc = locate_car(observation)
    dist = dist_from_edge_ahead(observation)
    print('distance from road edge:', dist)
    print('car loc: ', car_loc)
    if cur_time % 10 == 0:
        plt.figure(0)
        plt.imshow(filtered, cmap='gray')
        plt.show()
        plt.pause(0.1)
    total_reward = total_reward + reward

    if is_done:
        print('Episode is finised after {}-th timesteps'.format(cur_time))
        print('Total reward: {}'.format(total_reward))
        break
env.close()
