#!/usr/bin/env Python
# coding=utf-8
"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
"""
Reinforcement learning maze example
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = 1].
All other states:       ground      [reward = 0].
The observation is the dist between self.oval and self.rect
This script is the environment part of this example.
And I modified the script to custom hells.

Modified by lyl for 2020/08/26
花里胡哨的加了以下功能:
1.可以随便修改格子的大小，数量
2.可以设置格子的奖励值
3.用箭头可视化了每个格子的最大q值
4.存了每一个step的图片

"""

import numpy as np
import tkinter as tk
import time
import cv2
import win32gui
from PIL import ImageGrab


class Maze(tk.Tk, object):
    def __init__(self,
                 #### maze init params ####
                 unit=100,
                 rect_size=35,
                 maze_h=10,
                 maze_w=10,
                 goal_h=7,
                 goal_w=6,
                 scale=1.2,
                 exp_info="",
                 reward_type="sparse",
                 goal_reward=1.0):
        super(Maze, self).__init__()
        #### Maze init parameters ####
        # the pixel of pane
        self.unit = unit
        self.rect_size = rect_size
        # the num of panes
        self.maze_h = maze_h
        self.maze_w = maze_w
        self.goal_h = goal_h
        self.goal_w = goal_w
        self.image_size = self.unit * self.maze_h
        self.title_size = self.rect_size
        self.scale = scale
        #### RL params ####
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.reward_type = reward_type
        self.arrow_list = []
        self.q_value_text_list = []
        self.reward_text_list = []
        self.goal_reward = goal_reward
        #### visual params ####
        self.title("My_Maze_Visualize_DQN_Update")
        self.exp_info = exp_info
        fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
        video_name = self.exp_info + self.reward_type + ".avi"
        self.video = cv2.VideoWriter(video_name, fourcc, 15, (
        int(self.image_size / self.scale), int((self.image_size + self.title_size) / self.scale)))
        self.capture_images = []
        self.title_str = ""
        self.font = cv2.FONT_HERSHEY_COMPLEX

        self.reach_time = 0
        self.geometry('{0}x{1}'.format(self.maze_h * self.unit, self.maze_w * self.unit))
        self._built_maze()

    def reset(self, net, episode=0, key_episode=30):
        if episode > 0:
            HWND = self.canvas.winfo_id()
            rect = win32gui.GetWindowRect(HWND)
            img = ImageGrab.grab(rect)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            title_image = np.ones((self.title_size, self.image_size, 3)) * 255.0
            title_image = np.uint8(title_image)
            cv2.putText(title_image, self.title_str,
                        (int(self.rect_size / 2), int(self.rect_size / 2)),
                        self.font, 0.6, (0, 0, 0), 1)
            img = np.concatenate((title_image, img), axis=0)
            img = cv2.resize(img, (self.image_size, self.image_size + self.title_size))
            self.capture_images.append(img)

        if self.exp_info:
            self.title_str = self.exp_info + "RewardType:" + self.reward_type + "-Gamma:" + str(
                net.gamma) + "-Episode:" + str(episode) + "-ReachTime:" + str(self.reach_time)
        else:
            self.title_str = "Reward_type:" + self.reward_type + "-Gamma:" + str(net.gamma) + "-Episode:" + str(
                episode) + "-ReachTime:" + str(self.reach_time)

        self.title(self.title_str)
        self.update()
        self.canvas.delete(self.rect)
        for arrow in self.arrow_list:
            self.canvas.delete(arrow)
        for q_value_text in self.q_value_text_list:
            self.canvas.delete(q_value_text)
        for reward_text in self.reward_text_list:
            self.canvas.delete(reward_text)
        self.rect = self.create_rect(0, 0)

        target = np.array(self.canvas.coords(self.oval))
        agent = np.array(self.canvas.coords(self.rect))
        dist = (target - agent) / self.unit
        self.last_hmd_dist = abs(dist[0]) + abs(dist[1])

        self.draw_all_q_value(net=net, episode=episode, key_episode=key_episode)
        self.update()
        # the observation is the dist between self.oval and self.rect
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(self.maze_h*self.unit)

    def create_hell(self, x, y):
        hell_center = self.origin + np.array([self.unit * x, self.unit * y])
        hell = self.canvas.create_rectangle(
            hell_center[0] - self.rect_size, hell_center[1] - self.rect_size,
            hell_center[0] + self.rect_size, hell_center[1] + self.rect_size,
            fill='black'
        )
        return hell

    def create_rect(self, x, y):
        center = self.origin + np.array([self.unit * x, self.unit * y])
        rect = self.canvas.create_rectangle(
            center[0] - self.rect_size, center[1] - self.rect_size,
            center[0] + self.rect_size, center[1] + self.rect_size,
            fill='red'
        )
        return rect

    def _built_maze(self):
        self.canvas = tk.Canvas(self,
                                bg='white',
                                height=self.maze_h * self.unit,
                                width=self.maze_w * self.unit
                                )
        # create grids
        for c in range(0, self.maze_h * self.unit, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.maze_h * self.unit
            self.canvas.create_line(x0, y0, x1, y1,
                                    width=1,
                                    fill="yellow",
                                    tags="line")
        for r in range(0, self.maze_w * self.unit, self.unit):
            x0, y0, x1, y1 = 0, r, self.maze_w * self.unit, r
            self.canvas.create_line(x0, y0, x1, y1, width=1,
                                    fill="red", tags="line")

        # create origin
        self.origin = np.array([self.unit/2, self.unit/2])

        # create active rect
        self.rect = self.create_rect(0, 0)

        # create paradise
        oval_center = self.origin + np.array([self.unit * self.goal_h, self.unit * self.goal_w])
        self.oval = self.canvas.create_oval(
            oval_center[0] - self.rect_size, oval_center[1] - self.rect_size,
            oval_center[0] + self.rect_size, oval_center[1] + self.rect_size,
            fill='yellow'
        )
        # 这一段我忘了有啥用了...
        # oval_coords = self.canvas.coords(self.oval)
        # self.goal_coords = [(oval_coords[0] + oval_coords[2]) / 2.0,
        #                     (oval_coords[1] + oval_coords[3]) / 2.0,
        #                     ]
        # self.reward_txt = self.canvas.create_text(oval_coords[:2],
        #                                           text="R "+str(self.goal_reward))
        # create hells
        # self.hell_1 = self.create_hell(2,3)
        # self.hell_2 = self.create_hell(4,2)

        # pick all
        self.canvas.pack()

    """
    可视化Q值，思路：
    遍历每个格子，拿到当前格子的观察值obs，obs的单位需要和的DQN网络的一致.
    然后将obs传到DQN中，拿到所有动作（4个）的q_value。
    取得最大的q值的动作方向，作为箭头方向
    后面花里胡哨的，我也不知道当初是用做啥的了...
    """
    def draw_all_q_value(self, net, episode=0, key_episode=30):
        self.maze_h = self.maze_h
        self.maze_w = self.maze_w
        for i in range(self.maze_h):
            for j in range(self.maze_w):
                center_pos = np.array([self.unit * i + self.unit/2.0, self.unit * j + self.unit/2.0])
                oval_coords = self.canvas.coords(self.oval)
                goal_coords = [(oval_coords[0]+oval_coords[2]) / 2.0,
                               (oval_coords[1] + oval_coords[3]) / 2.0,
                               ]
                obs = center_pos - goal_coords
                s = obs / (self.maze_h * self.unit)
                # 获取Q值
                q_value = net.predict_eval_q(s)
                action = np.argmax(q_value)
                arrow_pos = None
                q_value_pos_list = []
                bias = self.rect_size
                # 根据Q值绝对值，绘制箭头长短，具体计算公式忘了...
                if q_value[0][action] > 0:
                    arrow_length = abs(int(q_value[0][action] / (sum(q_value[0]) + 1e-5) * bias))
                else:
                    min_v = np.min(q_value[0])
                    arrow_length = abs(q_value[0][action]-min_v)/abs(np.mean(q_value[0]) - min_v + 1e-5)*bias/4
                    # arrow_length = abs(int(q_value[0][action] / (sum(q_value[0]) + 1e-5) * bias))
                if arrow_length >= bias:
                    arrow_length = bias

                if action == 0:
                    arrow_pos = [center_pos[0], center_pos[1],
                                 center_pos[0], center_pos[1] - arrow_length]
                if action == 1:
                    arrow_pos = [center_pos[0], center_pos[1],
                                 center_pos[0], center_pos[1] + arrow_length]
                if action == 2:
                    arrow_pos = [center_pos[0], center_pos[1],
                                 center_pos[0]-arrow_length, center_pos[1]]
                if action == 3:
                    arrow_pos = [center_pos[0], center_pos[1],
                                 center_pos[0]+arrow_length, center_pos[1]]

                if arrow_pos:
                    arrow = self.canvas.create_line(arrow_pos,
                                                    arrow="last",
                                                    fill="blue",
                                                    # arrow_shape=(10, 40, 10),
                                                    )
                    self.arrow_list.append(arrow)
                else:
                    print("arrow_pos:", arrow_pos)

                if arrow_pos:
                    center_pos = [arrow_pos[0], arrow_pos[1]]
                    txt_bias = bias - 5
                    q_value_pos_list.append([center_pos[0],
                                             center_pos[1] - txt_bias])
                    q_value_pos_list.append([center_pos[0],
                                             center_pos[1] + txt_bias])
                    q_value_pos_list.append([center_pos[0] - txt_bias,
                                             center_pos[1]])
                    q_value_pos_list.append([center_pos[0] + txt_bias,
                                             center_pos[1]])
                    next_state = [center_pos[0]-self.unit/2+2,
                                  center_pos[1]-self.unit/2+8,
                                  center_pos[0]+self.unit/2,
                                  center_pos[1]+self.unit/2,
                                  ]
                    if self.reward_type != "Reletively_dist":
                        current_reward = self.reward(next_state=next_state,
                                                     episode=episode,
                                                     key_episode=key_episode,
                                                     )
                        current_reward = np.round(current_reward, 3)
                        current_reward_str = str(current_reward[0])
                        if obs[0] == 0. and obs[1] == 0.0:
                            reward_txt = self.canvas.create_text(next_state[:2],
                                                                 text="R " + str(self.goal_reward),
                                                                 anchor="w",
                                                                 justify="left", )
                        else:
                            reward_txt = self.canvas.create_text(next_state[:2],
                                                                 text="R " + current_reward_str,
                                                                 anchor="w",
                                                                 justify="left",)
                        self.reward_text_list.append(reward_txt)
                    for d in range(4):
                        q_value_str = str(np.round(q_value[0][d], 2))

                        q_value_text = self.canvas.create_text(q_value_pos_list[d],
                                                               text=q_value_str)
                        self.q_value_text_list.append(q_value_text)

    # 环境动态迭代函数，下一步要加状态转移概率
    def step(self, action, net, step=0, episode=0, key_episode=30):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        if step % 20 == 0:
            for arrow in self.arrow_list:
                self.canvas.delete(arrow)
            for q_value_text in self.q_value_text_list:
                self.canvas.delete(q_value_text)
            for reward_text in self.reward_text_list:
                self.canvas.delete(reward_text)
            self.draw_all_q_value(net=net, episode=episode, key_episode=key_episode)
            self.update()

            HWND = self.canvas.winfo_id()  # get the handle of the canvas
            rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
            img = ImageGrab.grab(rect)  # get image of the current location
            # img = np.array(img)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # cv2.imshow("grab", img)
            # cv2.waitKey(1)
            # cv2.destroyAllWindows()
            # print("img:", img.shape)
            title_image = np.ones((self.title_size, self.image_size, 3)) * 255.0
            title_image = np.uint8(title_image)
            # print("title_image:", title_image.shape)
            cv2.putText(title_image, self.title_str,
                        (int(self.rect_size/2), int(self.rect_size/2)),
                        self.font, 0.5, (0, 0, 0), 1)
            # cv2.imshow("title_image", title_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img = np.concatenate((title_image, img), axis=0)
            # print("img:", img.shape)
            img = cv2.resize(img, (int(self.image_size/self.scale), int((self.image_size+self.title_size)/self.scale)))
            img = np.uint8(img)
            # print("img:", img.shape)
            self.capture_images.append(img)

        if action == 0:
            if s[1] > self.unit:
                base_action[1] -= self.unit
        if action == 1:
            if s[1] < self.maze_h * self.unit - self.unit:
                base_action[1] += self.unit
        if action == 2:
            if s[0] > self.unit:
                base_action[0] -= self.unit
        if action == 3:
            if s[0] < self.maze_w * self.unit - self.unit:
                base_action[0] += self.unit

        # base_action[0] control the horizontal direction moving
        # base_action[1] control the vertical direction moving
        self.canvas.move(self.rect, base_action[0], base_action[1])

        # next_state
        next_state = self.canvas.coords(self.rect)

        reward, done = self.reward(next_state=next_state, episode=episode, key_episode=key_episode)
        # dist was normalized
        s_ = (np.array(next_state[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/self.unit/self.maze_h
        self.render()
        return s_, reward, done

    # 设置了非常多的奖励函数形式，可以用于探究不同reward shaping对策略收敛的影响
    def reward(self, next_state, now_state=None, episode=0, key_episode=30):
        if self.reward_type == "Sparse":
            return self.sparse_reward(next_state=next_state)
        elif self.reward_type == "Linear_increase":
            return self.linear_increase_reward(next_state=next_state)
        elif self.reward_type == "Negative_Euclidean":
            return self.negative_Euclidean_reward(next_state=next_state)
        elif self.reward_type == "Tanh":
            return self.tanh_reward(next_state=next_state)
        elif self.reward_type == "Reletively_dist":
            return self.reletively_dist_reward(next_state=next_state,
                                               now_state=now_state
                                               )
        elif self.reward_type == "Tanh2sparse":
            if episode < key_episode:
                return self.tanh_reward(next_state=next_state)
            else:
                return self.sparse_reward(next_state=next_state)
        elif self.reward_type == "Linear2sparse":
            if episode < key_episode:
                return self.linear_increase_reward(next_state=next_state)
            else:
                return self.sparse_reward(next_state=next_state)
        elif self.reward_type == "NE2sparse":
            if episode < key_episode:
                return self.negative_Euclidean_reward(next_state=next_state)
            else:
                return self.sparse_reward(next_state=next_state)

    def reletively_dist_reward(self,
                               next_state,
                               now_state=None,
                               ):
        r = 0.0
        if next_state == self.canvas.coords(self.oval):
            r = self.goal_reward
            self.reach_time += 1
            done = True

        else:
            if now_state:
                pass
            else:
                dense_reward = 0.0
                target = np.array(self.canvas.coords(self.oval))
                agent = np.array(next_state)
                dist = (target - agent) / self.unit
                dist_hmd = abs(dist[0]) + abs(dist[1])

                # print("dist_hmd:", dist_hmd)
                max_dist = self.maze_h + self.maze_w - 1
                delta_dist = self.last_hmd_dist - dist_hmd
                dense_reward = delta_dist / max_dist
                self.last_hmd_dist = dist_hmd
                done = False
                r = dense_reward
        return r, done

    def tanh_reward(self, next_state):
        r = 0.0
        target_coords = self.canvas.coords(self.oval)
        if next_state == self.canvas.coords(self.oval):
            r = self.goal_reward
            self.reach_time += 1
            done = True
        else:
            dense_reward = 0.0
            target = np.array(self.canvas.coords(self.oval))
            agent = np.array(next_state)
            dist = (target - agent)/self.unit
            origin_dist = np.array([dist[0], dist[1]])
            dist = np.linalg.norm(origin_dist)
            reaching_reward = 1 - np.tanh(dist/10.0)
            dense_reward = reaching_reward * 10
            done = False
            r = dense_reward
        return r, done

    def negative_Euclidean_reward(self, next_state):
        r = 0.0
        if next_state == self.canvas.coords(self.oval):
            r = self.goal_reward
            self.reach_time += 1
            done = True
        else:
            dense_reward = 0.0
            target = np.array(self.canvas.coords(self.oval))
            agent = np.array(next_state)
            dist = (target - agent)/self.unit
            dist_euc = (dist[0]**2 + dist[1]**2)/2.0
            dense_reward = -dist_euc
            done = False
            r = dense_reward
        return r, done

    def sparse_reward(self, next_state):
        r = 0.0
        done = False
        if next_state == self.canvas.coords(self.oval):
            r = self.goal_reward
            self.reach_time += 1
            done = True
        else:
            dense_reward = 0.0
            done = False
            r = dense_reward
        return r, done

    def linear_increase_reward(self, next_state):
        r = 0.0
        if next_state == self.canvas.coords(self.oval):
            r = self.goal_reward
            self.reach_time += 1
            done = True
        else:
            dense_reward = 0.0
            target = np.array(self.canvas.coords(self.oval))
            agent = np.array(next_state)
            dist = (target - agent)/self.unit
            dist_hmd = abs(dist[0]) + abs(dist[1])
            # print("dist_hmd:", dist_hmd)
            max_dist = self.maze_h + self.maze_w - 1
            dense_reward = (max_dist-dist_hmd)/max_dist
            done = False
            r = dense_reward
        return r, done

    def render(self):
        self.update()
        time.sleep(0.001)


def main():
    maze = Maze()
    maze.reset()
    done = False
    # while not done:
    for i in range(100):
        action = np.random.randint(0,4,1)
        print(action)
        obs, reward, done = maze.step(action)
        print(obs, reward, done)
    maze.reset()


if __name__ == "__main__":
    main()
