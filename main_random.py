#!/usr/bin/env Python
# coding=utf-8
"""
DQN visualization and save video to local
The RL is in DQN_keras.py.
利用win32gui带的截图功能，截取屏幕特定像素的图片。
如果你的笔记本的分辨率做了放大，那么截到的图就不全
跑程序之前，最好将分辨率调到100%
"""
import numpy as np
import matplotlib.pyplot as plt
from DQN_keras import DQNAgent
from maze_step_random import Maze

# run dqn main loop
def play_qlearning(env, agent, episode=0,
                   train=False, render=True,
                   max_step=200,
                   episodes=120,
                   key_episode=30,
                   ):
    episode_reward = 0
    # change gamma when the key episode defined.
    if episode > key_episode:
        agent.gamma = 1.0
    observation = env.reset(agent, episode=episode, key_episode=key_episode)
    step = 0
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, = env.step(action, agent, episode=episode, key_episode=key_episode)
        episode_reward += reward
        done = False
        if train:
            agent.learn(observation, action,
                        reward, next_observation,
                        done)
        # if done or step > max_step:
        if step > max_step:
            break
        observation = next_observation
        step += 1
    return episode_reward


# exp_info is video pre-name, to save some info for this exp hyper-parameters.
env = Maze(unit=80,
           rect_size=25,
           maze_h=4,
           maze_w=4,
           goal_h=2,
           goal_w=3,
           scale=1.2,
           exp_info="DS-10-10",
           reward_type="Tanh2sparse",
           goal_reward=10.0)
net_kwargs = {'hidden_sizes': [200, ],
              'learning_rate': 0.001}

agent = DQNAgent(state_dim=env.n_features,
                 action_dim=env.n_actions,
                 epsilon=0.4,
                 gamma=0.99,
                 net_kwargs=net_kwargs)

episodes = 80
episode_rewards = []
from progressbar import *
pbar = ProgressBar(maxval=episodes).start()

try:
    for episode in range(episodes):
        episode_reward = play_qlearning(env, agent,
                                        episode=episode,
                                        train=True,
                                        max_step=80,
                                        episodes=episodes,
                                        key_episode=40,
                                        )
        episode_rewards.append(episode_reward)
        pbar.update(episode)
        # print("episode_reward:", episode_reward)
    pbar.finish()
    for img in env.capture_images:
        env.video.write(img)
    env.video.release()
    import cv2
    cv2.destroyAllWindows()
    # plt.plot(episode_rewards)
except Exception as e:
    print(e)
    for img in env.capture_images:
        env.video.write(img)
    env.video.release()
    # import cv2
    # cv2.destroyAllWindows()

