import gym
import numpy as np

env = gym.make('CarRacing-v0')
init_observation = env.reset()
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
    total_reward = total_reward + reward
    print('Current reward: {}'.format(reward))

    if is_done:
        print('Episode is finised after {}-th timesteps'.format(cur_time))
        print('Total reward: {}'.format(total_reward))
        break
env.close()
