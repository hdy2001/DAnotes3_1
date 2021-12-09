import numpy as np
import time
import sys
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

        self.bm_trap = PhotoImage(file="人工智能/编程/hw5/trap.png")
        self.trap1 = self.canvas.create_image(origin[0] + UNIT * 2,
                                              origin[1] + UNIT,
                                              image=self.bm_trap)
        self.trap2 = self.canvas.create_image(origin[0] + UNIT,
                                              origin[1] + UNIT * 2,
                                              image=self.bm_trap)

        self.bm_mouse = PhotoImage(file="人工智能/编程/hw5/mouse.png")
        self.mouse = self.canvas.create_image(origin[0],
                                              origin[1],
                                              image=self.bm_mouse)

        self.bm_cheese = PhotoImage(file="人工智能/编程/hw5/cheese.png")
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


def update(env):
    # 更新图形化界面
    env.reset()
    time.sleep(5)
    for t in range(4):
        env.render()
        a = 1
        s, done = env.step(a)
        time.sleep(1)
    env.destroy()


# TODO: 完善强化学习算法
"""
选择价值迭代算法
"""
v = [[0, 0, 0, 0, 0], [0, 0, -100, 0, 0], [0, -100, 100, 0, 0],
     [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

r = [[-1, -1, -1, -1, -1], [-1, -1, -100, -1, -1], [-1, -100, 100, -1, -1],
     [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]]

gamma = 0.5


def max_action(state, env):
    max_value = r[state[0]][state[1]] + gamma * v[state[0]][state[1]]
    max_action = 0
    for action in range(0, 4):
        reward = r[state[0]][state[1]]
        state, _ = env.step(action, move=False)
        state = [int((state[0] - 50) / 100), int((state[1] - 50) / 100)]
        # print(state)
        new_value = reward + gamma * v[state[0]][state[1]]
        max_value = max(new_value, max_value)
        if max_value == new_value: max_action = action
    return max_value, max_action


def train(env):
    env.reset()
    time.sleep(3)
    s = [0, 0]
    for _ in range(100):
        for i in range(0, 5):
            for j in range(0, 5):
                # env.render()
                # value = v[s[0]][s[1]]
                s = [i, j]
                if s == [1, 2] or s == [2, 2] or s == [2, 3]:
                    v[i][j], a = max_action(s, env)
                # s, done = env.step(a)
                # s = [int(s[0] / 50 - 1), int(s[1] / 50 - 1)]
                # if done: s = [0, 0]
                # time.sleep(1)
        print(v)


def test():
    pass


if __name__ == '__main__':
    env = Maze()
    train(env)
    # 开始运行
    test()
# ============== Maze使用示例结束 ==============