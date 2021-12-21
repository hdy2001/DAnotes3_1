import numpy as np
import time
import sys
import copy
if sys.version_info.major == 2:
    import Tkinter as tk
    from Tkinter import PhotoImage
else:
    import tkinter as tk
    from tkinter import PhotoImage

UNIT = 100  # 迷宫中每个格子的像素大小
MAZE_H = 5  # 迷宫的高度（格子数）
MAZE_W = 5  # 迷宫的宽度（格子数）


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']  # 决策空间
        self.n_actions = len(self.action_space)
        self.title('Value Iteration')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        """迷宫初始化
        """
        self.canvas = tk.Canvas(self,
                                bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([UNIT / 2, UNIT / 2])
        self.bm_trap = PhotoImage(file="trap.png")
        self.trap1 = self.canvas.create_image(origin[0] + UNIT * 2,
                                              origin[1] + UNIT,
                                              image=self.bm_trap)
        self.trap2 = self.canvas.create_image(origin[0] + UNIT,
                                              origin[1] + UNIT * 2,
                                              image=self.bm_trap)

        self.bm_mouse = PhotoImage(file="mouse.png")
        self.mouse = self.canvas.create_image(origin[0],
                                              origin[1],
                                              image=self.bm_mouse)

        self.bm_cheese = PhotoImage(file="cheese.png")
        self.cheese = self.canvas.create_image(origin[0] + 2 * UNIT,
                                               origin[1] + 2 * UNIT,
                                               image=self.bm_cheese)

        self.canvas.pack()

    def reset(self):
        """重置迷宫，并返回老鼠初始位置
        """
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.mouse)
        origin = np.array([UNIT / 2, UNIT / 2])

        self.mouse = self.canvas.create_image(origin[0],
                                              origin[1],
                                              image=self.bm_mouse)
        # 返回当前老鼠所在的位置
        cords = self.canvas.coords(self.mouse)
        return cords

    def step(self, action, move=True):
        """给定行动action，在迷宫中移动老鼠位置

        参数:
            action ([int]): 移动方向，例如0-向上移动；1-向右移动；2-向下移动；3-向左移动

        返回值:
            s_ ([int, int]): 画布上老鼠移动后的坐标
            done (bool): 老鼠是否已经吃到奶酪
        """
        s = self.canvas.coords(self.mouse)
        base_action = np.array([0, 0])
        if action == 0:  # 向上移动
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 2:  # 向下移动
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 1:  # 向右移动
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # 向左移动
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.mouse, base_action[0],
                         base_action[1])  # 在图形化界面上移动老鼠
        s_ = self.canvas.coords(self.mouse)

        if not move:
            self.canvas.move(self.mouse, -base_action[0],
                             -base_action[1])  # 在图形化界面上移动老鼠

        # 判断游戏是否结束
        if s_ == self.canvas.coords(self.cheese):
            done = True
            s_ = 'terminal'
        else:
            done = False

        return s_, done

    def render(self):
        """渲染函数，更新画布
        """
        time.sleep(0.1)
        self.update()
        time.sleep(2)


# ============== 以下是Maze使用示例 ==============


def update(env, policy=None):
    # 更新图形化界面
    env.reset()
    time.sleep(5)
    state = [0, 0]
    while True:
        env.render()
        s, done = env.step(policy[state[0]][state[1]])
        if done:
            env.render()
            break
        state = [int((s[0] - 50) / 100), int((s[1] - 50) / 100)]
        time.sleep(1)
    env.destroy()


# TODO: 完善强化学习算法
"""
选择价值迭代算法
"""
r = [[-1, -1, -1, -1, -1], [-1, -1, -10000000, -1, -1],
     [-1, -100000000, 10000, -1, -1], [-1, -1, -1, -1, -1],
     [-1, -1, -1, -1, -1]]


def moveOnestep(action, state):
    if action == 0:  # 向左移动
        if state[1] > 0:
            state[1] -= 1
    elif action == 2:  # 向右移动
        if state[1] < 4:
            state[1] += 1
    elif action == 1:  # 向下移动
        if state[0] < 4:
            state[0] += 1
    elif action == 3:  # 向上移动
        if state[0] > 0:
            state[0] -= 1

    return state


# 用现有策略更新价值函数
def policy_evaluate(v, policy):
    for _ in range(100):
        now_v = np.copy(v)
        for i in range(0, 5):
            for j in range(0, 5):
                s = [i, j]
                # 如果状态结束，就跳过
                if s == [1, 2] or s == [2, 1] or s == [2, 2]:
                    v[i][j] = r[i][j]
                    continue
                # 未结束找到最佳更新
                action = policy[i][j]
                next_state = moveOnestep(action, s[:])
                # 如果动作合法则更新矩阵
                if next_state != s:
                    v[i][j] = r[next_state[0]][next_state[1]] + 0.1 * now_v[
                        next_state[0]][next_state[1]]

        print(v)

    return v


# TODO: de这个bug
# 用现有价值函数更新策略
def policy_improve(v, policy):
    for i in range(0, 5):
        for j in range(0, 5):
            rewards = {}
            s = [i, j]
            for action in range(0, 4):
                # 合法动作则更新state
                # next_state, _ = env.step(action)
                next_state = moveOnestep(action, s[:])
                # 如果原地不动则不更新
                if s == [1, 2] or s == [2, 1] or s == [2, 2
                                                       ] or next_state == s:
                    next_state = s[:]
                    continue
                reward = r[next_state[0]][
                    next_state[1]] + 0.1 * v[next_state[0]][next_state[1]]
                rewards.update({action: reward})
            # 选出最优策略，如果没有就不更新
            if len(rewards) > 0:
                print(max(rewards, key=rewards.get))
                policy[i][j] = max(rewards, key=rewards.get)

    return policy


if __name__ == '__main__':
    env = Maze()
    # 随机生成初始策略
    policy = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    v = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    for _ in range(50):
        v = policy_evaluate(v, policy)
        # 开始运行
        policy = policy_improve(v, policy)

    print(policy)

    # 开始执行程序
    update(env, policy)