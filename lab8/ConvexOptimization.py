import matplotlib.pyplot as plt
import numpy as np

a = np.array([2, -6, 2])  # y=2*(x-1.5)^2-2.5
p = np.poly1d(a)
x0 = -5  # -5
x1 = 5  # 5
delta = 0.1  # 0.1
s = 10  # 10
fmin = 10000000000000  # 10000000000000
threshold = 0.001  # 0.001


def convex_optimization(p, x0, x1, delta, s, fminPre, threshold, n=0):
    n = n+1
    x = np.arange(x0, x1, delta)
    y = p(x)
    # print(y)
    # print(x)
    for i in range(1, len(y)-1):
        if y[i] < y[i-1] and y[i] < y[i+1]:
            fmin = y[i]
            x0 = x[i-1]
            x1 = x[i+1]
            # print(x0)
            # print(x1)
            if fminPre - fmin < threshold:
                return x[i], fmin
            break
    delta = delta/s
    return convex_optimization(p, x0, x1, delta, s, fmin, threshold, n)


x, y = convex_optimization(p, x0, x1, delta, s, fmin, threshold)
print(x, y)

plt.plot(np.linspace(-30, 30, 61), p(np.linspace(-30, 30, 61)))
plt.show()
