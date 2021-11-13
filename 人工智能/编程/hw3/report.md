# 第三次编程

<center>何东阳 2019011462</center>

## 1 题目介绍

MNIST 数据库是美国国家标准与技术研究院收集整理的大型手写数字数据库，在机器学习领域被广泛使用。数据库中的每张图片由 28 × 28 个像素 点构成，每个像素点用一个灰度值表示，原始数据中将 28 × 28 的像素展开为一个一维的行向量（每行 784 个 值）。图片标签为 one-hot 编码：0-9。

## 2 编程要求

1. **参看课件，推导用随机梯度下降法求解一元 Logistic 回归的过程**：

​		一元Logistic回归的预测函数为：
$$
P(y=1) = \frac{1}{e^{-(wx+b)}+1}
$$
​		其损失函数为：
$$
F(w, b) =-\sum_{n=1}^{N}\left(y_{n} \ln (p)+\left(1-y_{n}\right) \ln (1-p)\right)
$$
​		其中$y_n$是标签，$p$是预测概率，$w$梯度更新公式为：
$$
\begin{aligned}
\frac{\partial F(w,b)}{\partial w}
& = \frac{\partial F(w,b)}{\partial p}\frac{\partial p}{\partial w} \\
& = -(\sum_{n=1}^{N}\left(\frac{y_{n}}{p}-\frac{\left(1-y_{n}\right)}{1-p} \right)\frac{\partial p}{\partial w}) \\
& = -(\sum_{n=1}^{N}\frac{y_n-p}{p(1-p)}\frac{\partial p}{\partial w})\\
\end{aligned}
$$
​		又，
$$
&\begin{aligned}
\frac{\partial p}{\partial w}
& = -\frac{1}{\left(1+e^{-w x-b}\right)^{2}} \cdot\left(1+e^{-w x - b}\right)^{\prime}\\
& = \frac{1}{\left(1+e^{-w x-b}\right)^{2}} \cdot e^{-w x - b}\cdot x\\
& = p(1-p)x
\end{aligned}
&\begin{aligned}
\frac{\partial p}{\partial b}
& = -\frac{1}{\left(1+e^{-w x-b}\right)^{2}} \cdot\left(1+e^{-w x - b}\right)^{\prime}\\
& = \frac{1}{\left(1+e^{-w x-b}\right)^{2}} \cdot e^{-w x - b}\\
& = p(1-p)
\end{aligned}
$$
​		带入得到
$$
\frac{\partial F(w,b)}{\partial w} = \sum_{n=1}^{N}(p-y_n)x_n
$$
​		同理
$$
\begin{aligned}
\frac{\partial F(w,b)}{\partial b}
& = \frac{\partial F(w,b)}{\partial p}\frac{\partial p}{\partial b} \\
& = -(\sum_{n=1}^{N}\left(\frac{y_{n}}{p}-\frac{\left(1-y_{n}\right)}{1-p} \right)\frac{\partial p}{\partial b}) \\
& = -(\sum_{n=1}^{N}\frac{y_n-p}{p(1-p)}\frac{\partial p}{\partial b})\\
& = \sum_{n=1}^{N}(p-y_n)
\end{aligned}
$$
​		因此使用随机梯度下降时梯度更新公式为：

​		

