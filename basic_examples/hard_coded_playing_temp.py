import os, sys
sys.path.append(os.path.dirname(sys.path[0]))
import gym
import numpy as np
from utils.image_proc import detect_edge, locate_car, dist_from_edge_ahead, strip_indicators, dist_from_right_and_left
import matplotlib.pyplot as plt

env = gym.make('CarRacing-v0')
observation = env.reset()
total_reward = 0
T_max = 2000

# check the action and observation spaces
print('Action space:{}'.format(env.action_space))
print('Lowest value for actions: {}'.format(env.action_space.low))
print('Highest value for actions: {}'.format(env.action_space.high))

print('Observation space:{}'.format(env.observation_space))
print('Lowest value for observation: {}'.format(np.min(env.observation_space.low)))
print('Highest value for observation: {}'.format(np.min(env.observation_space.high)))

# some discrete actions below
left_soft_speed_soft = [-0.8, 0.3, 0.0]
right_soft_speed_soft = [0.8, 0.3, 0.0]
accelarate = [0.0, 1.0, 0.2]
brake = [0.0, 0.0, 0.8]

for cur_time in range(T_max):
    print('Time: {}'.format(cur_time))
    env.render()

    if cur_time < 50:
        print('Taking random action')
        cur_action = env.action_space.sample() # take a random action
    observation, reward, is_done, info = env.step(cur_action)
    observation = strip_indicators(observation)  # Get rid of indicator bar on the bottom
    filtered = detect_edge(observation)  # Detect edges for vis
    car_loc = locate_car(observation)
    dist = dist_from_edge_ahead(observation)
    left_dist, right_dist = dist_from_right_and_left(observation)
    print('distance from road edge:', dist)
    print('left distance: {}, right distance: {}'.format(left_dist, right_dist))
    print('car loc: ', car_loc)
    # if cur_time % 10 == 0:
    #     plt.figure(0)
    #     plt.imshow(filtered, cmap='gray')
    #     plt.show()
    #     plt.pause(0.1)
    
    # if dist > 1 and dist < 30:
    #     print('Braking')
    #     cur_action = brake

    if dist > 1 and dist < 20 and right_dist > left_dist + 1:
        print('CURVE AHEAD, Turning right')
        cur_action = right_soft_speed_soft
    elif dist > 1 and dist < 20 and left_dist > right_dist + 1:
        print('CURVE AHEAD, Turning left')
        cur_action = left_soft_speed_soft
    elif right_dist > 0 and right_dist < 2:
        print('Getting close to Right, Turning left')
        cur_action = left_soft_speed_soft
    elif left_dist > 0 and left_dist < 2:
        print('Getting close to Left, Turning Right')
        cur_action = right_soft_speed_soft
    else:
        print('Accelerate')
        cur_action = accelarate
    
    
    total_reward = total_reward + reward

    if is_done:
        print('Episode is finised after {}-th timesteps'.format(cur_time))
        print('Total reward: {}'.format(total_reward))
        break
env.close()
