# 第二次数值分析大作业

<center>何东阳 2019011462 自96</center>

## 1 求解$ W_0 (z)$ **—** **数值法解微分方程**

### (1) 证明

$$
已知we^w = z，可以写作：\\
w(z)e^{w(z)} = z，两边同时对z求导(z\neq -\frac{1}{e}) \\
(w(z)e^{w(z)})^{\prime} = 1\\
w(z)^{\prime}e^{w(z)}+w(z)w(z)^{\prime}e^{w(z)} = 1\\
又w(z)e^{w(z)} = z，所以可得：\\
w(z)^{\prime} = \frac{1}{z+e^{w(z)}}，得证
$$

### (2) 编程求解与误差分析

我编程求解使用的方法是改进的欧拉法，预测使用的是欧拉二步公式，校正使用的是梯形公式。
$$
\left\{\begin{array}{l}
\bar{y}_{n+1}=y_{n}+h f\left(x_{n}, y_{n}\right) \\
y_{n+1}=y_{n}+\frac{h}{2}\left[f\left(x_{n}, y_{n}\right)+f\left(x_{n+1}, \bar{y}_{n+1}\right)\right]
\end{array}\right.
$$
分析**方法误差**

$$
\begin{aligned}
&\left\{\begin{array}{l}
\bar{\Delta}_{n+1} \leq(1+h M) \Delta_{n}+\frac{L}{2} h^{2} \\
\Delta_{n+1} \leq \Delta_{n}+\frac{h}{2} M \Delta_{n}+\frac{h}{2} M \bar{\Delta}_{n+1}+\frac{T \cdot h^{3}}{12}
\end{array}\right.\\
\Rightarrow & \Delta_{n+1} \leq\left(1+h M+\frac{h^{2}}{2} M^{2}\right) \Delta_{n}+\left(\frac{L M}{4}+\frac{T}{12}\right) h^{3} \\
&\text { 其中 }\left|\frac{\partial f}{\partial y}(x, y)\right| \leq M,\left|y^{(2)}(x)\right| \leq L,\left|y^{(3)}(x)\right| \leq T\\
& \Delta_{n+1}+\frac{\frac{LM}{4}+\frac{T}{12}h^3}{hM+\frac{h^2}{2}M^2}\leq
(1+hM+\frac{h^2}{2}M^2)^{n+1}\cdot\frac{\frac{LM}{4}+\frac{T}{12}h^3}{hM+\frac{h^2}{2}M^2}
\end{aligned}
$$
又
$$
\left|\frac{\partial f}{\partial y}(x, y)\right| = \frac{1}{(z+e^{W(z)})^2}\cdot e^{W(z)}\leq 1\\
\left|y^{(2)}(x)\right| = \frac{1}{(z+e^{W(z)})^2}(1+W(z)^{\prime}e^{W(z)})\leq2\\
\left|y^{(3)}(x)\right| =  \frac{2}{(z+e^{W(z)})^3}(1+W(z)^{\prime}e^{W(z)})\leq9
$$
因此可以得到
$$
M=1 , L=2 , T=9
$$
带入得到方法误差为：
$$
\Delta_{n+1}\leq0.25\times 10^{-9}
$$
分析**舍入误差**
$$
\begin{gathered}
\begin{cases}\bar{\delta}_{n+1} & \leq(1+h M) \delta_{n}+\frac{1}{2} \cdot 10^{-m} \\
\delta_{n+1} & \leq \delta_{n}+\frac{h}{2} M \delta_{n}+\frac{h}{2} M \bar{\delta}_{n+1}+\frac{1}{2} \cdot 10^{-m}\end{cases} \\
\Rightarrow \delta_{n+1} \leq\left(1+h M+\frac{h^{2}}{2} M^{2}\right) \delta_{n}+\left(1+\frac{h M}{2}\right) \cdot \frac{1}{2} \cdot 10^{-m}\\
\delta_{n+1} \leq\left(\left(1+h M+\frac{h^{2}}{2} M^{2}\right)^{n+1}-1\right) \frac{\left(1+\frac{h M}{2}\right) \cdot \frac{1}{2} \cdot 10^{-m}}{hM+\frac{h^2}{2}M^2}
\end{gathered}
$$
代入$M, L,T$得到：
$$
\delta_{n+1}\leq 0.25\times 10^{-9}
$$
以$z=2$为例解出来的得到



## 3 求解定积分

### (1) 



### (2)

