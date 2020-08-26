"""
这个不用加注释了吧？
标准好使的DQN~
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import gym
import matplotlib.pyplot as plt
from progressbar import *


class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward',
                                            'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)


class DQNAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 net_kwargs={},
                 gamma=0.99, epsilon=0.001,
                 replayer_capacity=10000, batch_size=64):
        observation_dim = state_dim
        self.action_n = action_dim
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.replayer = DQNReplayer(replayer_capacity)  # 经验回放

        self.evaluate_net = self.build_network(input_size=observation_dim,
                                               output_size=self.action_n, **net_kwargs)  # 评估网络
        self.target_net = self.build_network(input_size=observation_dim,
                                             output_size=self.action_n, **net_kwargs)  # 目标网络

        self.target_net.set_weights(self.evaluate_net.get_weights())

    def build_network(self, input_size,
                      hidden_sizes, output_size,
                      activation=tf.nn.relu,
                      output_activation=None,
                      learning_rate=0.01):  # 构建网络
        model = keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = dict(input_shape=(input_size,)) if not layer else {}
            model.add(keras.layers.Dense(units=hidden_size,
                                         activation=activation, **kwargs))
        model.add(keras.layers.Dense(units=output_size,
                                     activation=output_activation))  # 输出层
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def learn(self,
              observation,
              action,
              reward,
              next_observation,
              done,
              train_flag=True,
              ):
        self.replayer.store(observation, action, reward, next_observation,
                            done)  # 存储经验
        if train_flag:
            observations, actions, rewards, next_observations, dones = \
                self.replayer.sample(self.batch_size)  # 经验回放

            next_qs = self.target_net.predict(next_observations)
            next_max_qs = next_qs.max(axis=-1)
            us = rewards + self.gamma * (1. - dones) * next_max_qs
            targets = self.evaluate_net.predict(observations)
            targets[np.arange(us.shape[0]), actions] = us
            self.evaluate_net.fit(observations, targets, verbose=0)

            if done:  # 更新目标网络
                self.target_net.set_weights(self.evaluate_net.get_weights())

    def decide(self, observation):  # epsilon贪心策略
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        qs = self.evaluate_net.predict(observation[np.newaxis])
        return np.argmax(qs)

    def predict_target_q(self, observation):
        q_predict = self.target_net.predict(observation[np.newaxis])
        return q_predict

    def predict_eval_q(self, observation):
        q_predict = self.evaluate_net.predict(observation[np.newaxis])
        return q_predict


def play_qlearning(env, agent, train=False, render=True):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action,
                        reward, next_observation,
                        done)
        if done:
            break
        observation = next_observation
    return episode_reward


def main():
    env = gym.make('MountainCar-v0')
    env.seed(0)
    net_kwargs = {'hidden_sizes': [200, ],
                  'learning_rate': 0.01}

    agent = DQNAgent(state_dim=env.observation_space.shape[0],
                     action_dim=env.action_space.n,
                     epsilon=0.4,
                     net_kwargs=net_kwargs)

    # 训练
    episodes = 12
    episode_rewards = []

    pbar = ProgressBar(maxval=episodes).start()
    for episode in range(episodes):
        episode_reward = play_qlearning(env, agent, train=True)
        episode_rewards.append(episode_reward)
        pbar.update(episode)
        # print("episode_reward:", episode_reward)
    pbar.finish()
    plt.plot(episode_rewards)

    agent.epsilon = 0.

    episode_rewards = []
    pbar = ProgressBar()
    for i in pbar(range(10)):
        episode_reward = play_qlearning(env, agent)
        # print(episode_reward)
        episode_rewards.append(episode_reward)
    print('平均回合奖励 = {} / {} = {}'.format(sum(episode_rewards),
            len(episode_rewards), np.mean(episode_rewards)))


if __name__ == "__main__":
    main()
