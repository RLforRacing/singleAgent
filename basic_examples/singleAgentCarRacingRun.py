import gym
env = gym.make('CarRacing-v0')
init_observation = env.reset()
total_reward = 0
for cur_time in range(1000):
    env.render()

    cur_action = env.action_space.sample() # take a random action
    observation, reward, is_done, info = env.step(cur_action)
    total_reward = total_reward + reward
    print('Current reward: {}'.format(reward))

    if is_done:
        print('Episode is finised after {}th timesteps'.format(cur_time))
        print('Total reward: {}'.format(total_reward))

env.close()
