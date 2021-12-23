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
\Delta_{n+1}\leq0.25\times 10^{-m}
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
\delta_{n+1}\leq 0.25\times 10^{-m}
$$
以$z=2$为例解出来的得到（精确到小数点后六位）

$$
W_0(2) = 0.852606
$$

## 3 求解定积分

本报告中**a = 1**：

### (1) 

$$
\begin{aligned}
\int_{0}^{a} W_{0}(z) d z 
& = \int_{w_0(0)}^{w_0(a)} w d (we^w)\\
& = \int_{w_0(0)}^{w_0(a)} w(w+1)e^w d w\\
& = \left.w(w+1)e^w\right|_{w_0(0)}^{w_0(a)} -  \left.(2w+1)e^w\right|_{w_0(0)}^{w_0(a)} + \left.2e^w\right|_{w_0(0)}^{w_0(a)}\\
\end{aligned}
$$

分析方法误差：

此题使用解析解直接带求值，因此没有方法误差。

分析舍入误差：
$$
|\Delta \mathrm{A}| \leq \max \left|\left(\frac{\partial I}{\partial x}\right)\right|\left|\Delta x\right|\\
 = \max \left|\left(\frac{\partial I}{\partial x}\right)\right|\times\frac{1}{2}\times10^{-m}\\
 = M \times\frac{1}{2}\times 10^{-m}
$$
其中$M$是当 $z=1$时的积分一次导数的最大值。

### (2)

因为此题的积分无法计算出确切的解析表达式，因此我使用复化梯形公式进行了积分的估算：

首先，将原积分进行换元为
$$
W_0(z) = t^2\\
z = t^2e^{t^2}\\
I=\int_{0}^{\sqrt{W_0(a)}}\left(2 t^{2}+2 t^{4}\right) e^{t^{2}} d t
$$
又复化梯形公式为：
$$
I=\frac{h}{2}\left[f(a)+2 \sum_{k=1}^{n-1} f\left(x_{k}\right)+f(b)\right]\\
f(t) = (2 t^{2}+2 t^{4})e^{t^2}
$$
误差分析：

分析方法误差：
$$
|R[f]|=\frac{n \cdot h^{3}}{12} |f^{\prime \prime}(\eta)|=\frac{ah^{2}}{12}| f^{\prime \prime}(\eta)|
$$
又，
$$
f^{\prime \prime}(t)=\left(8 t^{6}+44 t^{4}+44 t^{2}+4\right) e^{t^{2}},0\leq t \leq\sqrt{W_0(a)}
$$
故，
$$
f^{\prime \prime}(\eta)\leq\left(8 W_0^{3}(a)+44 W_0^{2}(a)+44 W_0(a)+4\right) e^{W_0(a)}
$$
所以方法误差：
$$
|R[f]|\leq\frac{a h^{2}}{12}\left(8 W_0^{3}(a)+44 W_0^{2}(a)+44 W_0(a)+4\right) e^{W_0(a)}
$$
分析舍入误差：
$$
\delta \leq\left|\frac{\partial I}{\partial h}\right|\delta h+\sum_{0 \leq k \leq n}\left|\frac{\partial I}{\partial f\left(x_{k}\right)}\right| \delta f\left(x_{k}\right)+\frac{1}{2} \times 10^{-m}\\
\delta f\left(x_{k}\right) \leq\left|f^{\prime}\left(x_{k}\right)\right| \delta x_{k}+\frac{1}{2} \times 10^{-\mathrm{m}}\\
\delta h ,\delta x_{k} \leq \frac{1}{2} \times 10^{-\mathrm{m}}
$$
代入以下值的上界即可，
$$
\left|\frac{\partial I}{\partial h}\right|,\left|f^{\prime}\left(x_{k}\right)\right|,\sum_{0 \leq k \leq n}\left|\frac{\partial I}{\partial f\left(x_{k}\right)}\right|
$$

### (3)

代入$a = 1$，第一问得到**0.330366**，第二问得到**0.550832**
