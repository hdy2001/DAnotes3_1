import numpy as np

Y = np.array([1,2,3])
pred = np.array([1,1,1])

TP = np.where((Y == pred)&(Y > 0), np.ones_like(Y), np.zeros_like(Y))

print(TP)

data = np.array([[0, 2, 0], [3, 1, 2], [0, 4, 0]])
new_data = np.where((data >= 0) & (data <= 2), np.ones_like(data),
                    np.zeros_like(data))
print(new_data)